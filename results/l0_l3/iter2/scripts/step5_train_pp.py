"""Step 5b: train the point-process L0 and emit both forecast forms.

Encoder: the fixed (normalized + Fourier/Time2Vec) embedding from step 3, which is the
arm that de-collapsed the latent. Decoder: the continuous marked intensity from
model_pp. Objective: JEPA latent prediction + inhomogeneous marked-Poisson
log-likelihood of the actual independent-event coordinates (no grid, no binning).

Emits, for every window: the exactly grid-integrated rate array (so the same model can
be scored by the gridded N/S/M/CL battery) and simulated catalogs with real
coordinates, depths and magnitudes (for pycsep's catalog-based track and the depth
test in step 6).
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "repo/src")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from cvar_psha.eq_jepa.model import EQJEPAConfig
from cvar_psha.eq_jepa.model_v2 import build_model
from cvar_psha.eq_jepa.model_pp import PointProcessHead, PPConfig

EPOCHS, BATCH, LR, SEED, NCAT = 40, 32, 3e-4, 0, 500
cache = torch.load("samples_cache.pt", weights_only=False)
pp = np.load("pp_targets.npz", allow_pickle=False)
TR, EV = cache["train"], cache["eval"]
n_tr = TR["events"].shape[0]
tr_ev = torch.tensor(pp["train_events"]); tr_mask = torch.tensor(pp["train_mask"])
ev_ev = torch.tensor(pp["eval_events"]); ev_mask = torch.tensor(pp["eval_mask"])
times = pd.to_datetime([t for t in pp["eval_times"]], utc=True)
hor = torch.tensor(pp["eval_horizons"], dtype=torch.float32)
cal = np.where(times < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(times >= pd.Timestamp("2019-01-01", tz="UTC"))[0]

torch.manual_seed(SEED); np.random.seed(SEED)
cfg = EQJEPAConfig(440, 4, dim=128)
model = build_model(cfg, v2_embedding=True)
head = PointProcessHead(PPConfig(dim=cfg.dim))
params = list(model.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=LR)
perm = torch.Generator().manual_seed(SEED + 2)

def batch_loss(idx):
    b = {k: v[idx] for k, v in TR.items() if torch.is_tensor(v)}
    z = model.encode(b)
    pred = F.normalize(model.predictor(z), dim=-1)
    tv = {"events": b["future_events"], "event_mask": b["future_mask"],
          "long_history": b["future_long_history"]}
    with torch.no_grad():
        tgt = F.normalize(model.encode(tv, target=True), dim=-1)
    jepa = F.mse_loss(pred, tgt)
    e, m = tr_ev[idx], tr_mask[idx]
    par = head(z, b["horizon_days"])
    ll = head.log_likelihood(par, e[..., 0], e[..., 1], e[..., 2], e[..., 3], m,
                             b["horizon_days"])
    return jepa - ll.mean(), dict(jepa=float(jepa), pp_ll=float(ll.mean()))

hist = []
for ep in range(EPOCHS):
    model.train(); order = torch.randperm(n_tr, generator=perm); last = None
    for i in range(0, n_tr, BATCH):
        idx = order[i:i + BATCH]
        if len(idx) < 4: continue
        loss, met = batch_loss(idx)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); model.update_target(); last = met
    hist.append(dict(epoch=ep + 1, **last))
    if (ep + 1) % 10 == 0: print(f"epoch {ep+1}: {last}", flush=True)
pd.DataFrame(hist).to_csv("step5_training_history.csv", index=False)

model.eval(); head.eval()
all_idx = torch.arange(ev_ev.shape[0])
with torch.no_grad():
    z = model.encode({k: v for k, v in EV.items() if torch.is_tensor(v)})
    par = head(z, hor)
    lon_edges = torch.linspace(-125.0, -114.0, 23)
    lat_edges = torch.linspace(32.0, 42.0, 21)
    mag_edges = torch.tensor([2.8, 3.5, 4.5, 5.5, 8.5])
    grid = head.grid_rates(par, lon_edges, lat_edges, mag_edges)      # (B, n_lat*n_lon, n_mag)
    # grid_rates returns cells ordered (lat, lon) -> matches RegionGrid c = i*n_lon + j
    ll_ev = head.log_likelihood(par, ev_ev[..., 0], ev_ev[..., 1], ev_ev[..., 2],
                               ev_ev[..., 3], ev_mask, hor)

# expected moment per event under the learned truncated GR (now finite and well posed)
beta = par["beta"].numpy(); mmax = float(par["mmax"]); m0 = 2.8
mm = np.linspace(m0, mmax, 6001)
pdf = (beta[:, None] * np.exp(-beta[:, None] * (mm[None, :] - m0))
       / (1 - np.exp(-beta * (mmax - m0)))[:, None])
e_m0 = np.trapezoid(pdf * 10 ** (1.5 * mm + 9.05)[None, :], mm, axis=1)
np.savez_compressed("step5_pp_forecast.npz", grid=grid.numpy(), total=par["total"].numpy(),
                    beta=beta, mmax=np.array([mmax]), e_m0_per_event=e_m0,
                    heldout_ll=ll_ev.numpy(), eval_times=pp["eval_times"],
                    eval_horizons=pp["eval_horizons"])
torch.save({"encoder": model.state_dict(), "head": head.state_dict(), "cfg": cfg},
           "step5_ckpt_pointprocess.pt")

# simulated catalogs for the test windows (saved flat; step 5c/6 consume them)
gen = torch.Generator().manual_seed(123)
sim_store = {}
for w in tst:
    cats = head.simulate(par, int(w), NCAT, gen)
    lens = np.array([len(c[0]) for c in cats])
    flat = np.concatenate([np.stack(c, 1) for c in cats if len(c[0])], 0) if lens.sum() else \
        np.zeros((0, 4), np.float32)
    sim_store[f"len_{w}"] = lens; sim_store[f"ev_{w}"] = flat.astype(np.float32)
np.savez_compressed("step5_pp_simulations.npz", windows=tst, n_cat=np.array([NCAT]), **sim_store)
print(f"mean total (test) = {par['total'][tst].mean():.2f}, mean b = {beta[tst].mean()/np.log(10):.3f}, "
      f"Mmax = {mmax:.2f}, mean held-out PP log-lik = {ll_ev[tst].mean():.2f}", flush=True)
print(f"simulated {NCAT} catalogs for each of {len(tst)} test windows", flush=True)
