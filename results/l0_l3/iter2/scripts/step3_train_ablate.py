"""Step 3b: 2x2 ablation -- {original, fixed} event embedding x {no, with} SIGReg.

All four arms share the cached samples, the same seeds, the same horizon mix (1/7/30 d),
the same EMA + stop-gradient machinery, and the same count target (declustered
independent events, i.e. what floatCSEP scores -- removing the train/score catalog
mismatch that step 1 exposed). Only the two ablated factors differ.

Outputs: step3_arms.npz (per-arm rates + latents), step3_arm_diagnostics.csv,
step3_ckpt_<arm>.pt, step3_training_history.csv
"""
from __future__ import annotations
import sys, math, json
import numpy as np, pandas as pd, torch
from scipy import stats

sys.path.insert(0, "repo/src")
from cvar_psha.eq_jepa.model import EQJEPAConfig, negative_binomial_nll
from cvar_psha.eq_jepa.model_v2 import build_model, loss_v2, sigreg_epps_pulley

EPOCHS, BATCH, LR, SEED = 30, 32, 3e-4, 0
cache = torch.load("samples_cache.pt", weights_only=False)
TR = {k: v for k, v in cache["train"].items()}
EV = {k: v for k, v in cache["eval"].items()}
TR["counts"] = TR["counts_ind"]; EV["counts"] = EV["counts_ind"]
n_tr = TR["events"].shape[0]
eval_times = pd.to_datetime(cache["eval_times"], utc=True)
cal = np.where(eval_times < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(eval_times >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
cfg = EQJEPAConfig(440, 4, dim=128)

def take(d, idx):
    return {k: v[idx] for k, v in d.items() if torch.is_tensor(v)}

def grad_norm(model, batch, which: str, gen):
    model.zero_grad()
    z = model.encode(batch)
    if which == "nll":
        rates, disp = model.forecast_rates(batch)
        obj = negative_binomial_nll(batch["counts"], rates, disp).sum((-1, -2)).mean()
    else:
        obj = sigreg_epps_pulley(z, 64, gen)
    obj.backward()
    g = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None))
    model.zero_grad()
    return g

def latent_diagnostics(z: np.ndarray) -> dict:
    n, d = z.shape
    zc = z - z.mean(0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False); lam = s**2
    zu = z / np.linalg.norm(z, axis=1, keepdims=True)
    cos = (zu @ zu.T)[np.triu_indices(n, 1)]
    rng = np.random.default_rng(0)
    u = rng.normal(size=(512, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pr = zc @ u.T; prs = (pr - pr.mean(0)) / pr.std(0)
    ks = np.array([stats.kstest(prs[:, k], "norm").pvalue for k in range(prs.shape[1])])
    return dict(participation_ratio=float(lam.sum()**2 / (lam**2).sum()),
                frac_var_pc1=float(lam[0] / lam.sum()),
                n_components_99pct_var=int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1),
                mean_pairwise_cosine=float(cos.mean()),
                sketched_gaussianity_frac_rejected=float((ks < 0.05).mean()))

ARMS = {"A_original":        dict(v2=False, sigreg=False),
        "B_original_sigreg": dict(v2=False, sigreg=True),
        "C_fixed_embedding": dict(v2=True,  sigreg=False),
        "D_fixed_sigreg":    dict(v2=True,  sigreg=True)}

rates_out, z_out, rows, hist = {}, {}, [], []
for arm, spec in ARMS.items():
    torch.manual_seed(SEED); np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED + 1)
    model = build_model(cfg, spec["v2"])
    w = 0.0
    if spec["sigreg"]:
        b0 = take(TR, torch.arange(BATCH))
        gn, gr = grad_norm(model, b0, "nll", gen), grad_norm(model, b0, "sigreg", gen)
        w = float(gn / max(gr, 1e-12))          # gradient-norm matched at initialization
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    perm_gen = torch.Generator().manual_seed(SEED + 2)
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(n_tr, generator=perm_gen)
        last = None
        for i in range(0, n_tr, BATCH):
            b = take(TR, order[i:i + BATCH])
            if b["events"].shape[0] < 4: continue
            loss, met = loss_v2(model, b, sigreg_weight=w, generator=gen)
            opt.zero_grad(); loss.backward(); opt.step(); model.update_target()
            last = met
        hist.append(dict(arm=arm, epoch=ep + 1, **last))
        if (ep + 1) % 10 == 0:
            print(f"{arm} epoch {ep+1}: {last}", flush=True)
    model.eval()
    with torch.no_grad():
        z = model.encode(take(EV, torch.arange(EV["events"].shape[0])))
        r, disp = model.forecast_rates(take(EV, torch.arange(EV["events"].shape[0])))
        nll_raw = negative_binomial_nll(EV["counts"], r, disp).sum((-1, -2))
        c0 = float(EV["counts"][cal].sum() / r[cal].sum())
        nll_scaled = negative_binomial_nll(EV["counts"], r * c0, disp).sum((-1, -2))
    rates_out[arm] = r.numpy(); z_out[arm] = z.numpy()
    d = latent_diagnostics(z.numpy())
    rows.append(dict(arm=arm, fixed_embedding=spec["v2"], sigreg=spec["sigreg"],
                     sigreg_weight=w, scale_c0=c0,
                     heldout_nb_nll_raw=float(nll_raw[tst].mean()),
                     heldout_nb_nll_rescaled=float(nll_scaled[tst].mean()),
                     mean_forecast_total=float(r[tst].sum((1, 2)).mean() * c0),
                     total_rate_cv=float((r[tst].sum((1, 2)) / r[tst].sum((1, 2)).mean()).std()),
                     **d))
    torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": arm,
                "sigreg_weight": w}, f"step3_ckpt_{arm}.pt")
    print(f"{arm}: c0={c0:.3f} heldout_nll={float(nll_scaled[tst].mean()):.2f} "
          f"PR={d['participation_ratio']:.2f} cos={d['mean_pairwise_cosine']:.3f}", flush=True)

np.savez_compressed("step3_arms.npz", eval_horizons=np.array(cache["eval_horizons"]),
                    eval_times=np.array(cache["eval_times"]),
                    obs=EV["counts"].numpy(),
                    **{f"rates_{k}": v for k, v in rates_out.items()},
                    **{f"z_{k}": v for k, v in z_out.items()})
pd.DataFrame(rows).to_csv("step3_arm_diagnostics.csv", index=False)
pd.DataFrame(hist).to_csv("step3_training_history.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
