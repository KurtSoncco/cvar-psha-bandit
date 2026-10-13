"""Encoder-side fixes for EQ-JEPA L0, plus a sketched isotropy penalty.

Two changes, both additive -- ``model.py`` is untouched so the old arm stays runnable:

1. ``EventFeatureEmbedding`` replaces the raw ``nn.Linear(5, dim)`` that consumed
   unnormalized [dt_days, lat, lon, depth_km, magnitude] floats. Features are put on
   comparable scales, hypocentre coordinates get sinusoidal (Fourier) features so the
   encoder can resolve structure finer than a linear map allows, and the inter-event
   time gets a Time2Vec encoding (one linear term plus learned periodic terms).
   ``LongHistoryEncoderV2`` additionally log-compresses the monthly count/moment
   channels, which spanned several orders of magnitude unnormalized.

2. ``sigreg_epps_pulley`` is a differentiable Gaussianity penalty in the spirit of
   LeJEPA's SIGReg (arXiv:2511.08544): project the batch of latents onto random unit
   directions and penalize the weighted L2 distance between each projection's empirical
   characteristic function and the standard normal's. The closed form is the Epps-Pulley
   statistic; it is zero (up to an O(1/n) bias) iff every projection is N(0, 1), which
   is the isotropic-Gaussian target. Unlike EMA + stop-gradient alone, this constrains
   the latent *distribution* rather than merely making collapse a non-optimum.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig, negative_binomial_nll


def fourier_features(x: torch.Tensor, n_bands: int) -> torch.Tensor:
    """sin/cos at geometrically spaced frequencies for x already scaled to ~[-1, 1]."""
    bands = 2.0 ** torch.arange(n_bands, device=x.device, dtype=x.dtype) * math.pi
    a = x[..., None] * bands
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Time2Vec(nn.Module):
    """One linear term plus k learned periodic terms (Kazemi et al. 2019)."""
    def __init__(self, k: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(k) * 0.5); self.b = nn.Parameter(torch.zeros(k))
        self.w0 = nn.Parameter(torch.ones(1)); self.b0 = nn.Parameter(torch.zeros(1))
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.w0 * t[..., None] + self.b0,
                          torch.sin(self.w * t[..., None] + self.b)], -1)


class EventFeatureEmbedding(nn.Module):
    """Normalized + Fourier/Time2Vec embedding of one event token."""
    def __init__(self, dim: int, min_lat=32.0, max_lat=42.0, min_lon=-125.0, max_lon=-114.0,
                 n_bands=6, mag_bands=3, t2v_k=8, mag_ref=2.8, depth_scale=50.0,
                 dt_scale=30.0):
        super().__init__()
        self.register_buffer("lat_c", torch.tensor((min_lat + max_lat) / 2))
        self.register_buffer("lat_h", torch.tensor((max_lat - min_lat) / 2))
        self.register_buffer("lon_c", torch.tensor((min_lon + max_lon) / 2))
        self.register_buffer("lon_h", torch.tensor((max_lon - min_lon) / 2))
        self.n_bands, self.mag_bands = n_bands, mag_bands
        self.mag_ref, self.depth_scale, self.dt_scale = mag_ref, depth_scale, dt_scale
        self.t2v = Time2Vec(t2v_k)
        n_feat = 5 + (t2v_k + 1) + 3 * 2 * n_bands + 2 * mag_bands
        self.proj = nn.Sequential(nn.Linear(n_feat, dim), nn.GELU(), nn.LayerNorm(dim))
        self.n_feat = n_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt, lat, lon, dz, mag = (x[..., i] for i in range(5))
        dt_n = dt / self.dt_scale
        lat_n = (lat - self.lat_c) / self.lat_h
        lon_n = (lon - self.lon_c) / self.lon_h
        dz_n = (dz / self.depth_scale).clamp(-1.0, 3.0)
        mag_n = mag - self.mag_ref
        parts = [torch.stack([dt_n, lat_n, lon_n, dz_n, mag_n], -1),
                 self.t2v(dt_n),
                 fourier_features(lat_n, self.n_bands),
                 fourier_features(lon_n, self.n_bands),
                 fourier_features(dz_n, self.n_bands),
                 fourier_features(mag_n / 3.0, self.mag_bands)]
        return self.proj(torch.cat(parts, -1))


class EventHistoryEncoderV2(nn.Module):
    def __init__(self, cfg: EQJEPAConfig):
        super().__init__()
        self.input = EventFeatureEmbedding(cfg.dim)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4 * cfg.dim, batch_first=True,
                                           activation="gelu")
        self.net = nn.TransformerEncoder(layer, cfg.layers)
        self.empty = nn.Parameter(torch.zeros(cfg.dim))
    def forward(self, x, mask):
        h = self.net(self.input(x), src_key_padding_mask=~mask)
        denom = mask.sum(1, keepdim=True).clamp_min(1)
        pooled = (h * mask[..., None]).sum(1) / denom
        return torch.where(mask.any(1, keepdim=True), pooled, self.empty.expand_as(pooled))


class LongHistoryEncoderV2(nn.Module):
    """Monthly (count, moment) summaries, log-compressed before the GRU."""
    def __init__(self, cfg: EQJEPAConfig):
        super().__init__()
        self.net = nn.GRU(2 * cfg.long_dim, cfg.dim, num_layers=2, batch_first=True)
    def forward(self, x):
        lg = torch.log1p(x.clamp_min(0))
        return self.net(torch.cat([lg, lg - lg.mean(1, keepdim=True)], -1))[1][-1]


def sigreg_epps_pulley(z: torch.Tensor, n_dirs: int = 64,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Sketched isotropic-Gaussian penalty; 0 iff every 1-D projection is N(0, 1).

    Weighted L2 distance between the empirical characteristic function of each random
    projection and exp(-t^2/2), with Gaussian weight -- the Epps-Pulley closed form.
    """
    n, d = z.shape
    u = torch.randn(n_dirs, d, device=z.device, dtype=z.dtype, generator=generator)
    u = u / u.norm(dim=1, keepdim=True)
    y = z @ u.T                                              # (n, n_dirs)
    diff = y[:, None, :] - y[None, :, :]                     # (n, n, n_dirs)
    term1 = torch.exp(-0.5 * diff.pow(2)).mean((0, 1))
    term2 = (2.0 / math.sqrt(2.0)) * torch.exp(-0.25 * y.pow(2)).mean(0)
    return (term1 - term2 + 1.0 / math.sqrt(3.0)).mean()


def build_model(cfg: EQJEPAConfig, v2_embedding: bool) -> EQJEPA:
    """EQJEPA with the original or the fixed event/long embeddings; heads unchanged."""
    model = EQJEPA(cfg)
    if v2_embedding:
        model.events = EventHistoryEncoderV2(cfg)
        model.long = LongHistoryEncoderV2(cfg)
        model.target_events = copy.deepcopy(model.events)
        model.target_long = copy.deepcopy(model.long)
        model.target_fuse = copy.deepcopy(model.fuse)
        for mod in (model.target_events, model.target_long, model.target_fuse):
            mod.requires_grad_(False)
    return model


def loss_v2(model: EQJEPA, batch: dict, sigreg_weight: float = 0.0, n_dirs: int = 64,
            generator: torch.Generator | None = None):
    """JEPA latent-prediction loss + NB count loss + optional SIGReg isotropy penalty."""
    z = model.encode(batch)
    pred = F.normalize(model.predictor(z), dim=-1)
    target_view = {"events": batch["future_events"], "event_mask": batch["future_mask"],
                   "long_history": batch["future_long_history"]}
    with torch.no_grad():
        target = F.normalize(model.encode(target_view, target=True), dim=-1)
    jepa = F.mse_loss(pred, target)
    rates, dispersion = model.forecast_rates(batch)
    nll = negative_binomial_nll(batch["counts"], rates, dispersion).sum((-1, -2)).mean()
    reg = sigreg_epps_pulley(z, n_dirs, generator) if sigreg_weight > 0 else z.new_zeros(())
    total = jepa + nll + sigreg_weight * reg
    return total, {"loss": float(total.detach()), "jepa": float(jepa.detach()),
                   "nb_nll": float(nll.detach()), "sigreg": float(reg.detach())}
