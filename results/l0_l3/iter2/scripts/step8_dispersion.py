"""Step 8: does count dispersion matter for the tail, and how big is the L2 sigma gap?

(a) Per-cell overdispersion. L0 already carries a negative-binomial dispersion head, so
    this reads it out and tests the Poisson assumption per cell by likelihood ratio with
    the boundary correction (under r -> infinity the LR statistic is a 50:50 mixture of
    a point mass at 0 and chi^2_1), in the manner of EarthquakeNet (arXiv:2605.21437).
(b) What dispersion does to CVaR. Window losses for a site are built from the
    point-process forecast's own locations and magnitudes, with the event count drawn
    from Poisson or from a negative binomial at a range of dispersions.
(c) The L2 ground-motion sigma gap. The within-scenario sigma of the project's
    stochastic GMM is compared against the empirical total sigma range recorded for
    NGA-West2, the missing variance is attributed, and CVaR is recomputed with sigma
    inflated to the empirical range -- documenting the gap, which is as far as it can be
    taken without strong-motion records.

Outputs: step8_cell_dispersion.csv, step8_poisson_lr_test.csv, step8_dispersion_cvar.csv,
step8_sigma_decomposition.csv
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "repo/src")
import numpy as np, pandas as pd, torch
from scipy import stats, optimize, special
from cvar_psha.eq_jepa.model import EQJEPAConfig
from cvar_psha.eq_jepa.model_v2 import build_model
from cvar_psha.eq_jepa.sdof import loss_from_ductility

SITE_LAT, SITE_LON, ETA, A_CVAR = 34.05, -118.25, 0.25, 0.99
NGA_SIGMA_RANGE = (0.5, 0.7)        # empirical total sigma_ln(Y), project's recorded range

cache = torch.load("samples_cache.pt", weights_only=False)
EV = {k: v for k, v in cache["eval"].items() if torch.is_tensor(v)}
times = pd.to_datetime(cache["eval_times"], utc=True)
cal = np.where(times < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(times >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
obs = EV["counts_ind"].numpy()

ck = torch.load("step3_ckpt_C_fixed_embedding.pt", map_location="cpu", weights_only=False)
model = build_model(ck["config"], v2_embedding=True); model.load_state_dict(ck["state_dict"])
model.eval()
with torch.no_grad():
    rates, disp = model.forecast_rates(EV)
rates, disp = rates.numpy(), disp.numpy()
c0 = float(obs[cal].sum() / rates[cal].sum())
rates = rates * c0

# ---------- (a) per-cell dispersion and the Poisson LR test ----------
rows = []
cell_tot = obs[tst].sum(0)                       # (n_cells, n_mag) observed over the test period
for c in range(rates.shape[1]):
    for m in range(rates.shape[2]):
        n = obs[tst, c, m]; lam = rates[tst, c, m]
        if n.sum() < 5:                          # no power below this
            continue
        ll_pois = float((n * np.log(np.clip(lam, 1e-12, None)) - lam - special.gammaln(n + 1)).sum())
        def nll_nb(log_r):
            r = np.exp(log_r)
            return -float((special.gammaln(n + r) - special.gammaln(r) - special.gammaln(n + 1)
                           + r * np.log(r / (r + lam)) + n * np.log(lam / (r + lam))).sum())
        res = optimize.minimize_scalar(nll_nb, bounds=(-6, 12), method="bounded")
        lr = 2 * (-res.fun - ll_pois)
        p = 0.5 * stats.chi2.sf(max(lr, 0.0), 1)             # boundary-corrected
        rows.append(dict(cell=c, mag_bin=m, observed_total=int(n.sum()),
                         mean_forecast=float(lam.mean()), head_dispersion_r=float(disp[tst, c, m].mean()),
                         fitted_r=float(np.exp(res.x)), lr_statistic=float(lr), p_value=float(p),
                         rejects_poisson=bool(p < 0.05)))
cd = pd.DataFrame(rows); cd.to_csv("step8_cell_dispersion.csv", index=False)
print(f"(a) {len(cd)} cell/magnitude bins with >=5 observed events; "
      f"{100*cd.rejects_poisson.mean():.1f}% reject Poisson (boundary-corrected LR, alpha=0.05); "
      f"median fitted r = {cd.fitted_r.median():.2f}; median head r = {cd.head_dispersion_r.median():.2f}",
      flush=True)
# aggregate (whole-grid) test on window totals
n_tot = obs[tst].sum((1, 2)); lam_tot = rates[tst].sum((1, 2))
ll_p = float((n_tot * np.log(lam_tot) - lam_tot - special.gammaln(n_tot + 1)).sum())
def nll_tot(log_r):
    r = np.exp(log_r)
    return -float((special.gammaln(n_tot + r) - special.gammaln(r) - special.gammaln(n_tot + 1)
                   + r * np.log(r / (r + lam_tot)) + n_tot * np.log(lam_tot / (r + lam_tot))).sum())
rt = optimize.minimize_scalar(nll_tot, bounds=(-6, 14), method="bounded")
lr_tot = 2 * (-rt.fun - ll_p)
pd.DataFrame([dict(level="window totals", fitted_r=float(np.exp(rt.x)), lr_statistic=float(lr_tot),
                   p_value=float(0.5 * stats.chi2.sf(max(lr_tot, 0), 1)),
                   observed_variance_over_mean=float(n_tot.var() / n_tot.mean()))]
             ).to_csv("step8_poisson_lr_test.csv", index=False)
print(f"    whole-grid window totals: fitted r = {np.exp(rt.x):.1f}, LR = {lr_tot:.3f}, "
      f"p = {0.5*stats.chi2.sf(max(lr_tot,0),1):.3f} (r -> inf is Poisson)", flush=True)

# ---------- (b) dispersion -> CVaR ----------
sim = np.load("step5_pp_simulations.npz", allow_pickle=False)
fit = json.load(open("step6_l2_fit.json")); cf = np.array(fit["0.2"]["coef"]); sig0 = fit["0.2"]["sigma_ln"]

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0; p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) *
         np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))

pool = np.concatenate([sim[f"ev_{w}"] for w in sim["windows"] if sim[f"ev_{w}"].shape[0]], 0)
p_lon, p_lat, p_dep, p_mag = pool[:, 0], pool[:, 1], pool[:, 2], pool[:, 3]
p_r = np.sqrt(haversine_km(SITE_LAT, SITE_LON, p_lat, p_lon) ** 2 + np.clip(p_dep, 0, None) ** 2)

def window_losses(n_draws, lam, r_disp, sigma, rng):
    """Aggregate loss per window: count ~ Poisson/NB(lam), marks drawn from the forecast."""
    if r_disp is None:
        n = rng.poisson(lam, n_draws)
    else:
        p = r_disp / (r_disp + lam)
        n = rng.negative_binomial(r_disp, p, n_draws)
    tot = int(n.sum())
    idx = rng.integers(0, len(p_mag), tot)
    lnSa = cf[0] + cf[1] * p_mag[idx] + cf[2] * np.log(np.clip(p_r[idx], 1.0, None)) + cf[3] * p_r[idx]
    loss = loss_from_ductility(np.exp(lnSa + rng.normal(0, sigma, tot)) / ETA)
    out = np.zeros(n_draws)
    if tot:
        np.add.at(out, np.repeat(np.arange(n_draws), n), loss)
    return out

lam_mean = float(lam_tot.mean()); N_DRAW = 200_000
rows = []
for r_disp in [0.25, 0.5, 1.0, 2.0, 5.0, 20.0, None]:
    L = window_losses(N_DRAW, lam_mean, r_disp, sig0, np.random.default_rng(42))
    v = np.quantile(L, A_CVAR)
    rows.append(dict(count_law="Poisson" if r_disp is None else f"NB(r={r_disp})",
                     dispersion_r=np.inf if r_disp is None else r_disp,
                     count_variance_over_mean=1.0 if r_disp is None else 1 + lam_mean / r_disp,
                     mean_window_loss=float(L.mean()), VaR99=float(v),
                     CVaR99=float(L[L >= v].mean())))
dc = pd.DataFrame(rows)
dc["CVaR99_ratio_to_Poisson"] = dc.CVaR99 / float(dc.loc[dc.count_law == "Poisson", "CVaR99"].iloc[0])
dc.to_csv("step8_dispersion_cvar.csv", index=False)
print("\n(b)"); print(dc.round(5).to_string(index=False), flush=True)

# ---------- (c) L2 sigma gap ----------
rows = []
for sigma in [sig0, 0.4, 0.5, 0.6, 0.7]:
    L = window_losses(N_DRAW, lam_mean, None, sigma, np.random.default_rng(42))
    v = np.quantile(L, A_CVAR)
    rows.append(dict(sigma_ln_Sa=sigma, is_current_model=bool(abs(sigma - sig0) < 1e-9),
                     missing_sigma_in_quadrature=float(np.sqrt(max(sigma ** 2 - sig0 ** 2, 0.0))),
                     mean_window_loss=float(L.mean()), VaR99=float(v),
                     CVaR99=float(L[L >= v].mean())))
sd = pd.DataFrame(rows)
sd["CVaR99_ratio_to_current"] = sd.CVaR99 / float(sd.loc[sd.is_current_model, "CVaR99"].iloc[0])
sd.to_csv("step8_sigma_decomposition.csv", index=False)
print("\n(c)"); print(sd.round(5).to_string(index=False), flush=True)
print(f"    current within-scenario sigma_ln(Sa) = {sig0:.3f}; to reach the empirical total range "
      f"{NGA_SIGMA_RANGE} the missing component is "
      f"{np.sqrt(NGA_SIGMA_RANGE[0]**2 - sig0**2):.3f}-{np.sqrt(NGA_SIGMA_RANGE[1]**2 - sig0**2):.3f} "
      f"in quadrature (directivity, basin response, model form -- none randomized here)", flush=True)
