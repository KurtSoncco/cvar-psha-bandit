"""Iteration 3, step 2b (revised): H9 retrain with the gradient-balance fix.

The first H9 run was undertrained and its loss oscillated between -12 and -24 nats across
epochs, because the marked-Poisson log-likelihood of a window scales with the number of
events in it: a 7-day window holding an aftershock burst (up to 1,116 events) contributes
two orders of magnitude more gradient than a quiet 1-day window (2-3 events). Two ways to
fix that, and they are not equivalent, so both are implemented and compared:

  objective="mle"        keep the correct marked-Poisson likelihood and fix the MINI-BATCH
                         composition instead -- stratify by event-count decile so every
                         batch carries a similar mix of quiet and bursty windows. The
                         estimator is unchanged; only its variance drops.
  objective="normalized" divide each window's log-likelihood by (1 + n_events), which
                         equalizes windows directly but is no longer the MLE.

Runs on CPU or CUDA unchanged (`--device auto`). Checkpoints every epoch so an interrupted
run is never lost, and writes the same score tables as the first run so the H9 comparison
against the step-1 benchmarks stays like-for-like.
"""
from __future__ import annotations
import argparse, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in ("repo/src", os.path.join(_HERE, "..", "..", "src"),
           os.path.join(_HERE, "repo", "src"), os.path.join(_HERE, "src")):
    if os.path.isdir(os.path.join(_c, "cvar_psha")):
        sys.path.insert(0, os.path.abspath(_c))
        break
else:
    raise SystemExit("cannot locate the cvar_psha package: looked for src/cvar_psha "
                     "beside this script, two levels up, and under ./repo/src")
import numpy as np, pandas as pd, torch
from math import lgamma

# scipy.special.gammaln on integer counts, without the scipy dependency:
# the Lambda images ship a torch/scipy pair compiled against numpy 1.x, so any
# extra binary-numpy consumer is one more chance of an ABI clash.
_lgamma = np.vectorize(lgamma, otypes=[float])
import torch.nn.functional as F
from cvar_psha.eq_jepa.model import EQJEPAConfig
from cvar_psha.eq_jepa.model_v2 import build_model
from cvar_psha.eq_jepa.model_pp import PointProcessHead, PPConfig

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", choices=["mle", "normalized"], default="mle")
    p.add_argument("--stratify", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    return p.parse_args()

def main():
    a = parse()
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if a.device == "auto" else a.device))
    tag = a.tag or f"{a.objective}{'_strat' if a.stratify else ''}"
    print(f"device={dev} objective={a.objective} stratify={bool(a.stratify)} "
          f"epochs={a.epochs} tag={tag}", flush=True)
    if dev.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

    ck = torch.load("iter3_cache.pt", weights_only=False)
    D = {k: v.to(dev) for k, v in ck["data"].items()}
    meta = pd.DataFrame(ck["meta"])
    is_train = torch.tensor(meta.is_train.to_numpy())
    tr_idx = torch.where(is_train)[0]
    n_ev_tr = meta.n_events.to_numpy()[tr_idx.numpy()]
    print(f"{len(tr_idx)} training windows, {int((~is_train).sum())} test windows; "
          f"training event counts {n_ev_tr.min()}-{n_ev_tr.max()} "
          f"(90th pct {np.percentile(n_ev_tr, 90):.0f})", flush=True)

    # stratified batches: sort by count into deciles, then draw round-robin across deciles
    decile = np.clip((np.argsort(np.argsort(n_ev_tr)) / len(n_ev_tr) * 10).astype(int), 0, 9)
    strata = [tr_idx[np.where(decile == d)[0]] for d in range(10)]

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = EQJEPAConfig(440, 4, dim=128)
    model = build_model(cfg, v2_embedding=True).to(dev)
    head = PointProcessHead(PPConfig(dim=cfg.dim, mag_min=2.8)).to(dev)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed + 2)

    def batches(epoch):
        if not a.stratify:
            order = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
            return [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        per = max(a.batch // 10, 1)
        pools = [s[torch.randperm(len(s), generator=gen)].tolist() for s in strata]
        out, exhausted = [], False
        while not exhausted:
            b = []
            for p in pools:
                b += [p.pop() for _ in range(min(per, len(p)))]
            if len(b) < 8: exhausted = True
            else: out.append(torch.tensor(b))
            if sum(len(p) for p in pools) == 0: exhausted = True
        return out

    def loss_of(idx):
        b = {k: v[idx] for k, v in D.items()}
        z = model.encode(b)
        pred = F.normalize(model.predictor(z), dim=-1)
        tv = {"events": b["future_events"], "event_mask": b["future_mask"],
              "long_history": b["future_long_history"]}
        with torch.no_grad():
            tgt = F.normalize(model.encode(tv, target=True), dim=-1)
        jepa = F.mse_loss(pred, tgt)
        par = head(z, b["horizon_days"])
        nmax = max(int(b["target_mask"].sum(1).max().item()), 1)
        e, m = b["target_events"][:, :nmax], b["target_mask"][:, :nmax]
        ll = head.log_likelihood(par, e[..., 0], e[..., 1], e[..., 2], e[..., 3], m,
                                 b["horizon_days"])
        if a.objective == "normalized":
            ll = ll / (1.0 + m.sum(1))
            scale = 1.0
        else:
            scale = 50.0                      # keeps the two terms comparable in magnitude
        total = jepa - ll.mean() / scale
        return total, dict(jepa=float(jepa.detach()), pp_ll=float(ll.mean().detach()),
                           pp_ll_std=float(ll.detach().std()))

    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train(); last = None; mets = []
        for idx in batches(ep):
            loss, met = loss_of(idx.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); model.update_target(); mets.append(met); last = met
        ep_ll = float(np.mean([m["pp_ll"] for m in mets]))
        ep_sd = float(np.std([m["pp_ll"] for m in mets]))
        hist.append(dict(epoch=ep + 1, pp_ll_epoch_mean=ep_ll, pp_ll_epoch_std=ep_sd,
                         jepa=last["jepa"], minutes=(time.time() - t0) / 60))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}: pp_ll {ep_ll:+.2f} +- {ep_sd:.2f} (batch spread), "
                  f"jepa {last['jepa']:.4f}, {(time.time()-t0)/60:.1f} min", flush=True)
        torch.save({"encoder": model.state_dict(), "head": head.state_dict(), "cfg": cfg,
                    "epoch": ep + 1, "args": vars(a)}, f"iter3_ckpt_{tag}.pt")
    pd.DataFrame(hist).to_csv(f"iter3_history_{tag}.csv", index=False)

    # ---------------- out-of-sample scoring, identical metric to the first run
    model.eval(); head.eval()
    tot = np.zeros(len(meta))
    with torch.no_grad():
        for i in range(0, len(meta), 256):
            idx = torch.arange(i, min(i + 256, len(meta)))
            b = {k: v[idx.to(dev)] for k, v in D.items()}
            tot[idx.numpy()] = head(model.encode(b), b["horizon_days"])["total"].cpu().numpy()
    meta["l0_total"] = tot

    def pll(n, mu):
        mu = np.clip(mu, 1e-12, None)
        return n * np.log(mu) - mu - _lgamma(n + 1.0)

    base = pd.read_csv("iter3_window_forecasts.csv")
    rows = []
    for h, g in meta.groupby("horizon_days"):
        tr, te = g[g.is_train], g[~g.is_train]
        c0 = float(tr.n_events.sum() / tr.l0_total.sum())
        mu_l0 = te.l0_total.to_numpy() * c0
        n_te = te.n_events.to_numpy().astype(float)
        mu_c = np.full(len(te), float(tr.n_events.mean()))
        rec = dict(tag=tag, objective=a.objective, stratify=bool(a.stratify), epochs=a.epochs,
                   horizon_days=h, scale_c0=c0, mean_count=float(n_te.mean()),
                   mean_forecast=float(mu_l0.mean()),
                   gain_per_event=float((pll(n_te, mu_l0) - pll(n_te, mu_c)).sum() / n_te.sum()),
                   corr_forecast_observed=float(np.corrcoef(mu_l0, n_te)[0, 1]))
        bh = base[base.horizon_days == h].set_index("window_start")
        common = [w for w in te.window_start if w in bh.index]
        nb = bh.loc[common, "observed"].to_numpy().astype(float)
        mc_ = np.full(len(common), float(tr.n_events.mean()))
        for bm in ("excitation", "etas"):
            gb = (pll(nb, bh.loc[common, bm].to_numpy()) - pll(nb, mc_)).sum() / nb.sum()
            rec[f"gain_per_event_{bm}"] = float(gb)
            rec[f"l0_minus_{bm}"] = float(rec["gain_per_event"] - gb)
        rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(f"iter3_h9_summary_{tag}.csv", index=False)
    print(out[["horizon_days", "mean_count", "mean_forecast", "gain_per_event",
               "gain_per_event_etas", "l0_minus_etas", "corr_forecast_observed"]]
          .round(4).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
