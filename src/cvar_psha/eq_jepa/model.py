"""Hierarchical JEPA with a CSEP-ready, horizon-conditioned negative-binomial rate head.

Design (see EQ-JEPA_sequential_definition.md, L0): the representation ``z`` produced by
``encode()`` is horizon-agnostic -- it is the same latent regardless of what forecast
horizon is later queried. Horizon conditioning happens only at decode time, in
``forecast_rates()``: a daily baseline rate is scaled by the requested horizon length and
by a learned multiplicative correction for the well-known non-linearity of near-term
seismicity (elevated short-horizon rates from ongoing aftershock decay that a naive
linear-in-horizon Poisson rate would miss). This is what makes the field
"horizon-coherent" -- forecasts at different horizons from the same issue time are
decodes of one latent, not independently-fit models that can silently disagree.

The count head is Negative-Binomial rather than Poisson: earthquake counts in a
cell/magnitude bin are over-dispersed (variance >> mean) because of spatial-temporal
clustering, so a Poisson head under-penalizes confidently-wrong high-density predictions.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class EQJEPAConfig:
    n_cells: int; n_mag: int; event_dim: int = 5; long_dim: int = 2; dim: int = 128; heads: int = 4; layers: int = 3
    ema_tau: float = 0.996
    horizon_dim: int = 16          # width of the horizon-length embedding
    min_dispersion: float = 1e-2   # floor on NB dispersion r, avoids a degenerate/unstable fit

class HorizonEncoder(nn.Module):
    """Embeds the requested forecast horizon (in days) on a log scale.

    Log scale because the domain of interest spans 1-day nowcasts to multi-decade
    hazard horizons, and the near-term non-linearity being modeled (aftershock decay)
    is itself power-law-in-time, i.e. linear in log-time.
    """
    def __init__(self, cfg: "EQJEPAConfig"):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, cfg.horizon_dim), nn.GELU(), nn.Linear(cfg.horizon_dim, cfg.horizon_dim))
    def forward(self, horizon_days: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log(horizon_days.clamp_min(1e-3))[:, None])

class EventHistoryEncoder(nn.Module):
    def __init__(self, cfg: EQJEPAConfig):
        super().__init__(); self.input = nn.Sequential(nn.Linear(cfg.event_dim, cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim))
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4*cfg.dim, batch_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, cfg.layers); self.empty = nn.Parameter(torch.zeros(cfg.dim))
    def forward(self, x, mask):
        h = self.net(self.input(x), src_key_padding_mask=~mask)
        denom = mask.sum(1, keepdim=True).clamp_min(1); pooled = (h*mask[..., None]).sum(1)/denom
        return torch.where(mask.any(1, keepdim=True), pooled, self.empty.expand_as(pooled))

class LongHistoryEncoder(nn.Module):
    def __init__(self, cfg): super().__init__(); self.net = nn.GRU(cfg.long_dim, cfg.dim, num_layers=2, batch_first=True)
    def forward(self, x): return self.net(x)[1][-1]

def negative_binomial_nll(counts: torch.Tensor, mu: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-element NB negative log-likelihood, mean/dispersion parametrization.

    ``mu`` is the predicted mean count, ``r`` the dispersion (r -> inf recovers Poisson;
    small r means heavy over-dispersion, i.e. clustered/bursty counts).
    """
    mu = mu.clamp_min(eps)
    return -(torch.lgamma(counts + r) - torch.lgamma(r) - torch.lgamma(counts + 1)
             + r * torch.log(r / (r + mu) + eps) + counts * torch.log(mu / (r + mu) + eps))

class EQJEPA(nn.Module):
    def __init__(self, cfg: EQJEPAConfig):
        super().__init__(); self.cfg = cfg; self.events = EventHistoryEncoder(cfg); self.long = LongHistoryEncoder(cfg)
        self.fuse = nn.Sequential(nn.Linear(2*cfg.dim, cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim))
        self.predictor = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.rate_parent = nn.Linear(cfg.dim, 1); self.rate_allocation = nn.Linear(cfg.dim, cfg.n_cells*cfg.n_mag)
        self.rate_dispersion = nn.Linear(cfg.dim, cfg.n_cells*cfg.n_mag)
        self.horizon = HorizonEncoder(cfg); self.horizon_scale = nn.Sequential(
            nn.Linear(cfg.dim + cfg.horizon_dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.target_events, self.target_long, self.target_fuse = copy.deepcopy(self.events), copy.deepcopy(self.long), copy.deepcopy(self.fuse)
        for module in (self.target_events, self.target_long, self.target_fuse):
            module.requires_grad_(False)
    def encode(self, batch, target=False):
        ev, lo, fu = (self.target_events, self.target_long, self.target_fuse) if target else (self.events, self.long, self.fuse)
        return fu(torch.cat([ev(batch["events"], batch["event_mask"]), lo(batch["long_history"])], -1))
    def forecast_rates(self, batch, horizon_days: torch.Tensor | None = None):
        """Decode the horizon-agnostic latent into a (rate, dispersion) pair at a chosen horizon.

        ``horizon_days`` overrides ``batch["horizon_days"]`` when given, so a single
        trained model can be queried at any horizon post-hoc (e.g. floatCSEP's 1-day and
        30-day windows from the same L0 forward pass) without retraining.
        """
        z = self.encode(batch)
        if horizon_days is None:
            horizon_days = batch["horizon_days"]
        daily_rate = F.softplus(self.rate_parent(z)).squeeze(-1)          # events/day baseline, horizon-independent
        h_embed = self.horizon(horizon_days)
        correction = F.softplus(self.horizon_scale(torch.cat([z, h_embed], -1))).squeeze(-1)  # near-term non-linearity
        total = daily_rate * horizon_days * correction
        allocation = F.softmax(self.rate_allocation(z), -1).view(-1, self.cfg.n_cells, self.cfg.n_mag)
        rates = total[:, None, None] * allocation
        dispersion = F.softplus(self.rate_dispersion(z)).view(-1, self.cfg.n_cells, self.cfg.n_mag) + self.cfg.min_dispersion
        return rates, dispersion
    @torch.no_grad()
    def update_target(self):
        for online, target in ((self.events,self.target_events),(self.long,self.target_long),(self.fuse,self.target_fuse)):
            for p, q in zip(online.parameters(), target.parameters()): q.mul_(self.cfg.ema_tau).add_(p, alpha=1-self.cfg.ema_tau)
    def loss(self, context, future):
        z = self.encode(context); pred = F.normalize(self.predictor(z), dim=-1)
        target_view = {"events": future["future_events"], "event_mask": future["future_mask"], "long_history": future["future_long_history"]}
        with torch.no_grad(): target = F.normalize(self.encode(target_view, target=True), dim=-1)
        jepa = F.mse_loss(pred, target)
        rates, dispersion = self.forecast_rates(context); counts = context["counts"]
        nll = negative_binomial_nll(counts, rates, dispersion).sum((-1,-2)).mean()
        return jepa+nll, {"loss": float((jepa+nll).detach()), "jepa": float(jepa.detach()), "nb_nll": float(nll.detach())}
