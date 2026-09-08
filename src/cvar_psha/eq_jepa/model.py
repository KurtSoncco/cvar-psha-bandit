"""Hierarchical JEPA with a CSEP-ready, additive rate head."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class EQJEPAConfig:
    n_cells: int; n_mag: int; event_dim: int = 5; long_dim: int = 2; dim: int = 128; heads: int = 4; layers: int = 3; ema_tau: float = 0.996

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

class EQJEPA(nn.Module):
    def __init__(self, cfg: EQJEPAConfig):
        super().__init__(); self.cfg = cfg; self.events = EventHistoryEncoder(cfg); self.long = LongHistoryEncoder(cfg)
        self.fuse = nn.Sequential(nn.Linear(2*cfg.dim, cfg.dim), nn.GELU(), nn.LayerNorm(cfg.dim))
        self.predictor = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.rate_parent = nn.Linear(cfg.dim, 1); self.rate_allocation = nn.Linear(cfg.dim, cfg.n_cells*cfg.n_mag)
        self.target_events, self.target_long, self.target_fuse = copy.deepcopy(self.events), copy.deepcopy(self.long), copy.deepcopy(self.fuse)
        for module in (self.target_events, self.target_long, self.target_fuse):
            module.requires_grad_(False)
    def encode(self, batch, target=False):
        ev, lo, fu = (self.target_events, self.target_long, self.target_fuse) if target else (self.events, self.long, self.fuse)
        return fu(torch.cat([ev(batch["events"], batch["event_mask"]), lo(batch["long_history"])], -1))
    def forecast_rates(self, batch):
        z = self.encode(batch); total = F.softplus(self.rate_parent(z)).squeeze(-1)
        allocation = F.softmax(self.rate_allocation(z), -1).view(-1, self.cfg.n_cells, self.cfg.n_mag)
        return total[:, None, None]*allocation
    @torch.no_grad()
    def update_target(self):
        for online, target in ((self.events,self.target_events),(self.long,self.target_long),(self.fuse,self.target_fuse)):
            for p, q in zip(online.parameters(), target.parameters()): q.mul_(self.cfg.ema_tau).add_(p, alpha=1-self.cfg.ema_tau)
    def loss(self, context, future):
        z = self.encode(context); pred = F.normalize(self.predictor(z), dim=-1)
        target_view = {"events": future["future_events"], "event_mask": future["future_mask"], "long_history": future["future_long_history"]}
        with torch.no_grad(): target = F.normalize(self.encode(target_view, target=True), dim=-1)
        jepa = F.mse_loss(pred, target)
        rates = self.forecast_rates(context); counts = context["counts"]
        nll = (rates-counts*torch.log(rates.clamp_min(1e-8))).sum((-1,-2)).mean()
        return jepa+nll, {"loss": float((jepa+nll).detach()), "jepa": float(jepa.detach()), "poisson_nll": float(nll.detach())}
