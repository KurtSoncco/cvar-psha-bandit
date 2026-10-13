"""Continuous marked point-process decoder for EQ-JEPA L0 (section 9.1 redesign).

Replaces the gridded softmax allocation head with an intensity

    lambda(t, x, y, z, m | latent, horizon)
        = N(latent, horizon) * f_t(t) * f_xy(x, y) * f_z(z) * f_m(m)

conditioned on the same JEPA latent. Every factor is closed-form and exactly
normalized over the modelling domain, which is what makes three things possible that
the gridded head could not do:

* the likelihood is the standard inhomogeneous-Poisson log-likelihood of the actual
  event locations, so training needs no grid and no binning;
* the forecast can be simulated as catalogs with real coordinates, depths and
  magnitudes, which is what pycsep's catalog-based test track consumes;
* the magnitude factor is a truncated Gutenberg-Richter law with a learned b and a
  learned (global) upper bound, so the implied seismic moment is finite and
  well-posed -- unlike a gridded forecast with an open top magnitude bin, whose
  Kostrov integral is dominated by the bin edge.

Spatial and depth components are axis-aligned Gaussians, which keeps the truncated
normalization and the per-cell integrals exact (products of error functions) rather
than approximate. Time is uniform within the forecast window: the CSEP window tests
do not resolve within-window timing, and an Omori-style f_t is the natural next
extension.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)

def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))

def _trunc_norm_logpdf(x, mu, log_sigma, lo, hi):
    """log density of N(mu, sigma) truncated to [lo, hi], broadcast over components."""
    sigma = log_sigma.exp()
    zc = (x - mu) / sigma
    log_phi = -0.5 * zc.pow(2) - math.log(math.sqrt(2 * math.pi)) - log_sigma
    mass = (_norm_cdf((hi - mu) / sigma) - _norm_cdf((lo - mu) / sigma)).clamp_min(1e-8)
    return log_phi - mass.log()

def _trunc_norm_cell_mass(lo_edge, hi_edge, mu, log_sigma, lo, hi):
    """probability mass of the truncated normal inside [lo_edge, hi_edge]."""
    sigma = log_sigma.exp()
    num = _norm_cdf((hi_edge - mu) / sigma) - _norm_cdf((lo_edge - mu) / sigma)
    den = (_norm_cdf((hi - mu) / sigma) - _norm_cdf((lo - mu) / sigma)).clamp_min(1e-8)
    return (num / den).clamp_min(0.0)


@dataclass
class PPConfig:
    dim: int = 128
    n_components: int = 24
    min_lat: float = 32.0
    max_lat: float = 42.0
    min_lon: float = -125.0
    max_lon: float = -114.0
    min_depth: float = -5.0
    max_depth: float = 60.0
    mag_min: float = 2.8
    horizon_dim: int = 16


class PointProcessHead(nn.Module):
    """Decodes a latent into total count plus normalized space/depth/magnitude densities."""
    def __init__(self, cfg: PPConfig):
        super().__init__()
        self.cfg = cfg
        K = cfg.n_components
        self.rate = nn.Linear(cfg.dim, 1)
        self.horizon = nn.Sequential(nn.Linear(1, cfg.horizon_dim), nn.GELU(),
                                     nn.Linear(cfg.horizon_dim, cfg.horizon_dim))
        self.horizon_scale = nn.Sequential(nn.Linear(cfg.dim + cfg.horizon_dim, cfg.dim),
                                           nn.GELU(), nn.Linear(cfg.dim, 1))
        self.mix = nn.Linear(cfg.dim, K)
        self.loc = nn.Linear(cfg.dim, 4 * K)          # lon mu, lat mu, log sigmas
        self.dep = nn.Linear(cfg.dim, 2 * K)          # depth mu, log sigma
        self.bval = nn.Linear(cfg.dim, 1)
        self.mmax_raw = nn.Parameter(torch.tensor(0.0))   # global upper magnitude bound
        # initialize component means spread over the region, sigmas ~ 1 degree
        with torch.no_grad():
            self.loc.bias[:K].uniform_(cfg.min_lon + 1, cfg.max_lon - 1)
            self.loc.bias[K:2*K].uniform_(cfg.min_lat + 1, cfg.max_lat - 1)
            self.loc.bias[2*K:].fill_(math.log(0.7))
            self.dep.bias[:K].uniform_(4.0, 12.0)
            self.dep.bias[K:].fill_(math.log(5.0))
            self.bval.bias.fill_(0.0)
            self.loc.weight.mul_(0.01); self.dep.weight.mul_(0.01); self.mix.weight.mul_(0.1)

    def mmax(self) -> torch.Tensor:
        return 6.5 + 2.0 * torch.sigmoid(self.mmax_raw)      # in [6.5, 8.5]

    def forward(self, z: torch.Tensor, horizon_days: torch.Tensor) -> dict:
        c, K = self.cfg, self.cfg.n_components
        h = self.horizon(torch.log(horizon_days.clamp_min(1e-3))[:, None])
        total = (F.softplus(self.rate(z)).squeeze(-1) * horizon_days
                 * F.softplus(self.horizon_scale(torch.cat([z, h], -1))).squeeze(-1)).clamp_min(1e-6)
        logw = F.log_softmax(self.mix(z), -1)
        p = self.loc(z)
        mu_lon = p[:, :K].clamp(c.min_lon, c.max_lon)
        mu_lat = p[:, K:2*K].clamp(c.min_lat, c.max_lat)
        ls_lon = p[:, 2*K:3*K].clamp(math.log(0.08), math.log(6.0))
        ls_lat = p[:, 3*K:4*K].clamp(math.log(0.08), math.log(6.0))
        d = self.dep(z)
        mu_dep = d[:, :K].clamp(c.min_depth, c.max_depth)
        ls_dep = d[:, K:].clamp(math.log(0.5), math.log(25.0))
        beta = (F.softplus(self.bval(z)).squeeze(-1) + 0.2) * math.log(10.0)   # b in (0.2, inf)
        return dict(total=total, logw=logw, mu_lon=mu_lon, mu_lat=mu_lat, ls_lon=ls_lon,
                    ls_lat=ls_lat, mu_dep=mu_dep, ls_dep=ls_dep, beta=beta, mmax=self.mmax())

    # ---------- densities ----------
    def log_space_depth(self, par: dict, lon, lat, dep) -> torch.Tensor:
        """log f_xy,z at event coordinates; lon/lat/dep are (B, n_ev)."""
        c = self.cfg
        lp = (_trunc_norm_logpdf(lon[..., None], par["mu_lon"][:, None], par["ls_lon"][:, None],
                                 c.min_lon, c.max_lon)
              + _trunc_norm_logpdf(lat[..., None], par["mu_lat"][:, None], par["ls_lat"][:, None],
                                   c.min_lat, c.max_lat)
              + _trunc_norm_logpdf(dep[..., None], par["mu_dep"][:, None], par["ls_dep"][:, None],
                                   c.min_depth, c.max_depth))
        return torch.logsumexp(par["logw"][:, None] + lp, -1)

    def log_magnitude(self, par: dict, mag) -> torch.Tensor:
        beta, mmax, m0 = par["beta"][:, None], par["mmax"], self.cfg.mag_min
        norm = 1.0 - torch.exp(-beta * (mmax - m0))
        return torch.log(beta) - beta * (mag - m0) - torch.log(norm.clamp_min(1e-8))

    def log_likelihood(self, par: dict, lon, lat, dep, mag, mask, horizon_days) -> torch.Tensor:
        """Inhomogeneous marked-Poisson log-likelihood per batch element."""
        ll_ev = (self.log_space_depth(par, lon, lat, dep) + self.log_magnitude(par, mag)
                 + torch.log(par["total"])[:, None] - torch.log(horizon_days)[:, None])
        return (ll_ev * mask).sum(-1) - par["total"]

    # ---------- grid integration (exact, for the gridded CSEP tests) ----------
    def grid_rates(self, par: dict, lon_edges, lat_edges, mag_edges) -> torch.Tensor:
        """Integrate the intensity over a lat/lon/magnitude grid -> (B, n_cells, n_mag)."""
        c = self.cfg
        mlon = _trunc_norm_cell_mass(lon_edges[:-1][None, :, None], lon_edges[1:][None, :, None],
                                     par["mu_lon"][:, None], par["ls_lon"][:, None],
                                     c.min_lon, c.max_lon)
        mlat = _trunc_norm_cell_mass(lat_edges[:-1][None, :, None], lat_edges[1:][None, :, None],
                                     par["mu_lat"][:, None], par["ls_lat"][:, None],
                                     c.min_lat, c.max_lat)
        w = par["logw"].exp()
        # (B, n_lat, n_lon) = sum_k w_k * mlat[:, i, k] * mlon[:, j, k]
        cell = torch.einsum("bk,bik,bjk->bij", w, mlat, mlon).reshape(w.shape[0], -1)
        beta, mmax, m0 = par["beta"][:, None], par["mmax"], c.mag_min
        cdf = (1.0 - torch.exp(-beta * (mag_edges[None, :].clamp(max=mmax.detach()) - m0))) / \
              (1.0 - torch.exp(-beta * (mmax - m0))).clamp_min(1e-8)
        pm = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
        return par["total"][:, None, None] * cell[:, :, None] * pm[:, None, :]

    # ---------- simulation (for CatalogForecast) ----------
    @torch.no_grad()
    def simulate(self, par: dict, b: int, n_cat: int, gen: torch.Generator):
        """Draw n_cat catalogs for batch element b; returns list of (lon, lat, dep, mag)."""
        c = self.cfg
        total = float(par["total"][b])
        n = torch.poisson(torch.full((n_cat,), total), generator=gen).long()
        w = par["logw"][b].exp()
        out = []
        for i in range(n_cat):
            k = torch.multinomial(w, int(n[i]), replacement=True, generator=gen)
            def tn(mu, ls, lo, hi):
                m, s = mu[k], ls[k].exp()
                a, bb = _norm_cdf((lo - m) / s), _norm_cdf((hi - m) / s)
                u = torch.rand(int(n[i]), generator=gen) * (bb - a) + a
                return m + s * SQRT2 * torch.erfinv((2 * u - 1).clamp(-1 + 1e-7, 1 - 1e-7))
            lon = tn(par["mu_lon"][b], par["ls_lon"][b], c.min_lon, c.max_lon)
            lat = tn(par["mu_lat"][b], par["ls_lat"][b], c.min_lat, c.max_lat)
            dep = tn(par["mu_dep"][b], par["ls_dep"][b], c.min_depth, c.max_depth)
            beta, mmax = float(par["beta"][b]), float(par["mmax"])
            u = torch.rand(int(n[i]), generator=gen)
            mag = c.mag_min - torch.log1p(-u * (1 - math.exp(-beta * (mmax - c.mag_min)))) / beta
            out.append((lon.numpy(), lat.numpy(), dep.numpy(), mag.numpy()))
        return out
