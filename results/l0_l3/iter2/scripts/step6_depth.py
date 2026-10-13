"""Step 6: a depth consistency test, and what depth does to the risk tail.

(a) D-test. No depth test exists anywhere in pycsep -- even its catalog-based spatial
    test re-grids onto a 2-D region and scores only lon/lat. This builds one by reusing
    pycsep's own simulate-and-rank pattern rather than hand-deriving depth-binned
    likelihood math: bin depth, take the forecast's depth distribution from its
    simulated catalogs, score the observed depth histogram by Poisson joint
    log-likelihood against the forecast's expected histogram rescaled to the observed
    count (exactly how pycsep's magnitude test treats magnitude), and rank that score
    against 1000 synthetic depth histograms drawn from the same forecast.

(b) R_hyp vs R_epi. The point-process forecast now carries real depths, so hypocentral
    distance R_hyp = sqrt(R_epi^2 + z^2) can be propagated through L2 (ground motion)
    and L3/L4 (SDOF ductility -> loss -> CVaR) for a site, and compared against the
    epicentral-distance version the pipeline used before. The ground-motion distance
    and magnitude scaling is fitted from the project's own saved L2 stochastic-GMM
    ensemble; the ductility map is the linear-limit relation mu_d = Sa/eta, so the
    reported CVaR sensitivity is a lower bound (the pilot found a degrading oscillator
    roughly doubles CVaR_99 relative to the linear case).
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "repo/src")
import numpy as np, pandas as pd
from csep_scoring import load_independent_catalog
from cvar_psha.eq_jepa.sdof import loss_from_ductility

DEPTH_EDGES = np.array([-5.0, 2.0, 5.0, 8.0, 11.0, 15.0, 20.0, 60.0])
NSIM_D, ALPHA = 1000, 0.05
SITE_LAT, SITE_LON = 34.05, -118.25          # downtown Los Angeles
T1, ZETA, ETA, ALPHA_CVAR = 0.5, 0.05, 0.25, 0.99

sim = np.load("step5_pp_simulations.npz", allow_pickle=False)
z5 = np.load("step5_pp_forecast.npz", allow_pickle=False)
times = pd.to_datetime([t for t in z5["eval_times"]], utc=True)
z1 = np.load("forecasts_step1.npz", allow_pickle=False)
ends = pd.to_datetime([t for t in z1["ends"]], utc=True)
tst = sim["windows"]
df = load_independent_catalog()
rng = np.random.default_rng(11)

# ---------- (a) D-test ----------
rows = []
for w in tst:
    lens, ev = sim[f"len_{w}"], sim[f"ev_{w}"]
    obs = df[(df.Time >= times[w]) & (df.Time < ends[w])]
    n_obs = len(obs)
    if n_obs == 0 or ev.shape[0] == 0:
        continue
    sim_depth = ev[:, 2]
    p = np.histogram(sim_depth, bins=DEPTH_EDGES)[0].astype(float)
    p = p / p.sum()
    expected = n_obs * p                              # rescaled to the observed count
    obs_hist = np.histogram(obs["Depth/km"].to_numpy().clip(-4.9, 59.9), bins=DEPTH_EDGES)[0]

    def jll(h):
        mu = np.clip(expected, 1e-9, None)
        from scipy.special import gammaln
        return float((h * np.log(mu) - mu - gammaln(h + 1)).sum())

    obs_score = jll(obs_hist)
    sim_scores = np.empty(NSIM_D)
    for s in range(NSIM_D):
        draw = rng.choice(sim_depth, size=n_obs, replace=True)
        sim_scores[s] = jll(np.histogram(draw, bins=DEPTH_EDGES)[0])
    q = float((sim_scores < obs_score).mean())
    rows.append(dict(window_start=times[w].strftime("%Y-%m-%d"), n_observed=n_obs,
                     observed_joint_loglik=obs_score, median_simulated=float(np.median(sim_scores)),
                     quantile=q, passes=bool(q >= ALPHA),
                     observed_median_depth=float(obs["Depth/km"].median()),
                     forecast_median_depth=float(np.median(sim_depth))))
dt = pd.DataFrame(rows); dt.to_csv("step6_dtest.csv", index=False)
print(f"D-test: {100*dt.passes.mean():.1f}% of {len(dt)} windows pass; "
      f"median quantile {dt['quantile'].median():.3f}; "
      f"observed median depth {dt.observed_median_depth.median():.1f} km vs forecast "
      f"{dt.forecast_median_depth.median():.1f} km", flush=True)

# ---------- (b) fit the L2 distance/magnitude scaling from the saved ensemble ----------
l2 = pd.read_csv("l2_ensemble.csv")
fits = {}
for T, g in l2.groupby("period_s"):
    piv = g.groupby(["M", "R_km"]).Sa_g.agg(["mean", "std", "count"]).reset_index()
    X = np.column_stack([np.ones(len(piv)), piv.M, np.log(piv.R_km), piv.R_km])
    y = np.log(piv["mean"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sig_within = float(np.mean(g.groupby(["M", "R_km"]).Sa_g.apply(lambda s: np.std(np.log(s)))))
    fits[float(T)] = dict(coef=beta.tolist(), r2=float(1 - resid.var() / y.var()),
                          sigma_ln=sig_within)
json.dump(fits, open("step6_l2_fit.json", "w"), indent=2)
cf = np.array(fits[0.2]["coef"]); sig = fits[0.2]["sigma_ln"]
print(f"L2 fit at T=0.2 s: R2={fits[0.2]['r2']:.3f}, sigma_ln(Sa)={sig:.3f}, coefs={cf.round(3)}",
      flush=True)

def sa_of(M, R, n_draw, rng):
    lnSa = cf[0] + cf[1] * M + cf[2] * np.log(np.clip(R, 1.0, None)) + cf[3] * R
    return np.exp(lnSa + rng.normal(0.0, sig, size=n_draw))

# ---------- (b) CVaR with and without depth, using the simulated catalogs ----------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

allev = np.concatenate([sim[f"ev_{w}"] for w in tst if sim[f"ev_{w}"].shape[0]], 0)
lon, lat, dep, mag = allev[:, 0], allev[:, 1], allev[:, 2], allev[:, 3]
r_epi = haversine_km(SITE_LAT, SITE_LON, lat, lon)
r_hyp = np.sqrt(r_epi ** 2 + np.clip(dep, 0, None) ** 2)
out = []
for lab, R in (("R_epi (depth ignored)", r_epi), ("R_hyp (real depth)", r_hyp)):
    rg = np.random.default_rng(5)
    sa = sa_of(mag, R, len(R), rg)
    mu_d = sa / ETA                                  # linear-limit ductility
    loss = loss_from_ductility(mu_d)
    v = np.quantile(loss, ALPHA_CVAR)
    cv = float(loss[loss >= v].mean())
    out.append(dict(distance_metric=lab, n_events=len(R), median_R_km=float(np.median(R)),
                    mean_Sa_g=float(sa.mean()), median_Sa_g=float(np.median(sa)),
                    p99_Sa_g=float(np.quantile(sa, 0.99)), mean_loss=float(loss.mean()),
                    VaR99_loss=float(v), CVaR99_loss=cv,
                    frac_events_within_50km=float((R < 50).mean())))
cd = pd.DataFrame(out)
cd["CVaR99_ratio_to_Repi"] = cd.CVaR99_loss / cd.CVaR99_loss.iloc[0]
cd["mean_loss_ratio_to_Repi"] = cd.mean_loss / cd.mean_loss.iloc[0]
cd.to_csv("step6_rhyp_vs_repi.csv", index=False)
print(cd.round(4).to_string(index=False), flush=True)
# depth sensitivity sweep: how the CVaR gap grows with assumed source depth
sweep = []
for zfix in (0, 5, 10, 15, 20, 30):
    R = np.sqrt(r_epi ** 2 + zfix ** 2)
    rg = np.random.default_rng(5)
    sa = sa_of(mag, R, len(R), rg); loss = loss_from_ductility(sa / ETA)
    v = np.quantile(loss, ALPHA_CVAR)
    sweep.append(dict(assumed_depth_km=zfix, median_R_km=float(np.median(R)),
                      mean_loss=float(loss.mean()), CVaR99_loss=float(loss[loss >= v].mean())))
pd.DataFrame(sweep).to_csv("step6_depth_sweep.csv", index=False)
print(pd.DataFrame(sweep).round(5).to_string(index=False), flush=True)
