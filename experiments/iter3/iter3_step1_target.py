"""Iteration 3, step 1: define the L0 target where signal exists, and fix the bar.

Second-iteration result that drives this: monthly counts of DECLUSTERED M2.8+ events are
Poisson with no memory, so no model can show skill there. The target moves to the
undeclustered catalog at 1-, 3- and 7-day horizons, with a magnitude of completeness
estimated per year instead of a flat cutoff.

Nothing is trained here. The step produces the specification plus the out-of-sample
likelihood of the two benchmarks any L0 has to beat:

  * an exponential-decay excitation feature, c0 * (1 + a * sum exp(-dt/tau)), which is
    the cheap stand-in used in the cross-region survey;
  * a temporal ETAS integrated over each window,
    mu*Delta + K * sum_i 10^(alpha*(m_i-m0)) * [(T1-t_i+c)^(1-p) - (T0-t_i+c)^(1-p)]/(1-p),
    with (mu, K) fitted by Poisson MLE on the training period and (alpha, c, p) profiled
    over a grid.

Outputs: iter3_mc_by_year.csv, iter3_baselines.csv, iter3_window_forecasts.csv,
target_spec.md
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import optimize, special

HORIZONS = [1.0, 3.0, 7.0]
TRAIN_START, SPLIT, END = "2010-01-01", "2019-01-01", "2024-12-01"
TAUS = np.concatenate([np.linspace(0.1, 5, 30), np.linspace(5.5, 60, 25)])
LOOKBACK = 730.0
ETAS_GRID = dict(alpha=[0.6, 0.8, 1.0, 1.2], c=[0.003, 0.01, 0.05], p=[1.05, 1.15, 1.3])
DM = 0.1

def aki_utsu_b(m, mc):
    s = m[m >= mc - 1e-9]
    return float(1.0 / (np.log(10) * (s.mean() - (mc - DM / 2)))) if len(s) > 20 else np.nan

def mc_maxc(m):
    """Maximum-curvature estimate plus the conventional +0.2 correction."""
    bins = np.arange(np.floor(m.min() * 10) / 10, m.max() + DM, DM)
    h, e = np.histogram(m, bins=bins)
    return float(e[np.argmax(h)] + 0.2)

def mc_bvalue_stability(m, lo=2.0, hi=4.0):
    """Smallest Mc whose b-value is stable over the next 0.5 magnitude units."""
    grid = np.arange(lo, hi, DM)
    bs = np.array([aki_utsu_b(m, c) for c in grid])
    for i, c in enumerate(grid[:-6]):
        win = bs[i:i + 6]
        if np.all(np.isfinite(win)) and np.nanmax(np.abs(win - bs[i])) <= 0.03:
            return float(c)
    return float(np.nan)

def days(ts):
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy()

def poisson_ll(counts, mu):
    mu = np.clip(mu, 1e-12, None)
    return counts * np.log(mu) - mu - special.gammaln(counts + 1)

def main():
    cat = pd.read_csv("california_m2_catalog.csv")
    cat["time"] = pd.to_datetime(cat["time"], format="mixed", utc=True)
    cat = cat[(cat.lat >= 32) & (cat.lat <= 42) & (cat.lon >= -125) & (cat.lon <= -114)]
    print(f"{len(cat)} events M>={cat.mag.min():.1f}", flush=True)

    rows = []
    for y, g in cat.groupby(cat.time.dt.year):
        m = g.mag.to_numpy()
        rows.append(dict(year=int(y), n_events=len(m), mc_maxc=mc_maxc(m),
                         mc_bstab=mc_bvalue_stability(m),
                         b_at_2p8=aki_utsu_b(m, 2.8), b_at_mc_maxc=aki_utsu_b(m, mc_maxc(m))))
    mc = pd.DataFrame(rows); mc.to_csv("iter3_mc_by_year.csv", index=False)
    mc_max = float(np.nanmax(mc.mc_maxc)); mc_p90 = float(np.nanpercentile(mc.mc_maxc, 90))
    target_mag = max(2.8, float(np.ceil(mc_p90 * 10) / 10))
    print(mc.round(3).to_string(index=False), flush=True)
    print(f"Mc(t) by maximum curvature: max {mc_max:.2f}, 90th pct {mc_p90:.2f} "
          f"-> target threshold M{target_mag:.1f}", flush=True)

    sub = cat[cat.mag >= target_mag].sort_values("time")
    t_ev, m_ev = days(sub.time), sub.mag.to_numpy()
    t_split = float(days(pd.Series([pd.Timestamp(SPLIT, tz="UTC")]))[0])
    t0_all = float(days(pd.Series([pd.Timestamp(TRAIN_START, tz="UTC")]))[0])
    t1_all = float(days(pd.Series([pd.Timestamp(END, tz="UTC")]))[0])

    out_rows, win_rows = [], []
    for h in HORIZONS:
        edges = np.arange(t0_all, t1_all, h)
        counts = np.histogram(t_ev, bins=edges)[0].astype(float)
        starts, ends = edges[:-1], edges[1:]
        tr = starts < t_split; te = ~tr
        # --- excitation feature over the tau grid
        feat = np.zeros((len(starts), len(TAUS)))
        for i, s in enumerate(starts):
            a, b = np.searchsorted(t_ev, [s - 60.0, s])
            if b > a:
                feat[i] = np.exp(-(s - t_ev[a:b])[:, None] / TAUS[None, :]).sum(0)
        best = None
        for j, tau in enumerate(TAUS):
            f = feat[tr, j]; n = counts[tr]
            def nll(la):
                g = 1.0 + np.exp(la) * f
                c0 = n.sum() / g.sum()
                return -poisson_ll(n, c0 * g).sum()
            r = optimize.minimize_scalar(nll, bounds=(-12, 6), method="bounded")
            if best is None or r.fun < best[0]: best = (r.fun, float(np.exp(r.x)), float(tau), j)
        _, amp, tau_hat, jbest = best
        g_all = 1.0 + amp * feat[:, jbest]
        c0 = counts[tr].sum() / g_all[tr].sum()
        mu_exc = c0 * g_all
        # --- ETAS integrated over each window
        best_etas = None
        for alpha in ETAS_GRID["alpha"]:
            w_ev = 10.0 ** (alpha * (m_ev - target_mag))
            for c in ETAS_GRID["c"]:
                for p in ETAS_GRID["p"]:
                    phi = np.zeros(len(starts))
                    for i, (s, e) in enumerate(zip(starts, ends)):
                        a, b = np.searchsorted(t_ev, [s - LOOKBACK, s])
                        if b > a:
                            ti = t_ev[a:b]
                            phi[i] = (w_ev[a:b] * ((e - ti + c) ** (1 - p)
                                                   - (s - ti + c) ** (1 - p)) / (1 - p)).sum()
                    n = counts[tr]; ph = phi[tr]
                    def nll2(theta):
                        mu0, K = np.exp(theta)
                        return -poisson_ll(n, mu0 * h + K * ph).sum()
                    r = optimize.minimize(nll2, x0=np.log([max(n.mean() / h, 1e-3), 0.1]),
                                          method="Nelder-Mead",
                                          options=dict(maxiter=600, xatol=1e-4, fatol=1e-4))
                    if best_etas is None or r.fun < best_etas[0]:
                        best_etas = (r.fun, np.exp(r.x), alpha, c, p, phi)
        _, (mu0, K), alpha_h, c_h, p_h, phi = best_etas
        mu_etas = mu0 * h + K * phi
        mu_const = np.full(len(starts), counts[tr].mean())

        ll = {k: poisson_ll(counts, v) for k, v in
              (("constant", mu_const), ("excitation", mu_exc), ("etas", mu_etas))}
        n_te, n_ev_te = te.sum(), counts[te].sum()
        for k in ll:
            out_rows.append(dict(horizon_days=h, model=k, n_windows_test=int(n_te),
                                 n_events_test=int(n_ev_te), mean_count=float(counts[te].mean()),
                                 ll_per_window=float(ll[k][te].mean()),
                                 ll_per_event=float(ll[k][te].sum() / max(n_ev_te, 1)),
                                 gain_per_window=float((ll[k][te] - ll["constant"][te]).mean()),
                                 gain_per_event=float((ll[k][te] - ll["constant"][te]).sum()
                                                      / max(n_ev_te, 1)),
                                 corr_forecast_observed=float(np.corrcoef(
                                     {"constant": mu_const, "excitation": mu_exc,
                                      "etas": mu_etas}[k][te], counts[te])[0, 1])
                                 if k != "constant" else np.nan))
        for i in np.where(te)[0]:
            win_rows.append(dict(horizon_days=h,
                                 window_start=str(pd.Timestamp("1970-01-01", tz="UTC")
                                                  + pd.Timedelta(days=float(starts[i])))[:10],
                                 observed=int(counts[i]), constant=float(mu_const[i]),
                                 excitation=float(mu_exc[i]), etas=float(mu_etas[i])))
        print(f"h={h:g}d  excitation amp={amp:.4f} tau={tau_hat:.2f}d | "
              f"ETAS mu={mu0:.3f}/d K={K:.4f} alpha={alpha_h} c={c_h} p={p_h} | "
              f"test gain/event exc {out_rows[-2]['gain_per_event']:+.4f} "
              f"etas {out_rows[-1]['gain_per_event']:+.4f}", flush=True)

    bl = pd.DataFrame(out_rows); bl.to_csv("iter3_baselines.csv", index=False)
    pd.DataFrame(win_rows).to_csv("iter3_window_forecasts.csv", index=False)
    print(bl[["horizon_days", "model", "mean_count", "ll_per_event", "gain_per_event",
              "corr_forecast_observed"]].round(4).to_string(index=False), flush=True)
    json.dump(dict(target_magnitude=target_mag, mc_maxc_max=mc_max, mc_maxc_p90=mc_p90,
                   horizons=HORIZONS, train=[TRAIN_START, SPLIT], test=[SPLIT, END]),
              open("iter3_target_spec.json", "w"), indent=2)

if __name__ == "__main__":
    main()
