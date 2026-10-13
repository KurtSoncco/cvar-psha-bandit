"""Step 2: is EQ-JEPA L0's latent carrying information, or has it collapsed?

Geometric diagnostics on the encoder output that L1-L4 actually consume (no projector),
a behavioural diagnostic on the decoded rate field, and a deliberate-collapse control
scored through the same CSEP tests.

Outputs: step2_collapse_diagnostics.csv, step2_latent_spectrum.csv,
step2_collapsed_control_scores.csv, step2_allocation_similarity.csv
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd, torch
from scipy import stats
from scipy.optimize import minimize_scalar
from csep.core.regions import CartesianGrid2D
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog
from csep.core import poisson_evaluations as pe

sys.path.insert(0, "repo/src")
from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig

ALPHA, NSIM = 0.05, 1000
MAGS = np.array([2.8, 3.5, 4.5, 5.5]); N_LAT, N_LON, DH, MIN_LAT, MIN_LON = 20, 22, 0.5, 32.0, -125.0
CAT = "repo/results/l0_l3/data/california_catalog_declustered_vetted.csv"

z = np.load("forecasts_step1.npz")
zlat, obs, horizons = z["zlat"].astype(np.float64), z["obs"], z["horizons"]
rates_l0, rates_gr = z["rates_l0"], z["rates_gr"]
starts = pd.to_datetime([s for s in z["starts"]], utc=True)
ends = pd.to_datetime([s for s in z["ends"]], utc=True)
cal = np.where(starts < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(starts >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
n, d = zlat.shape

# ---------- geometric diagnostics on z ----------
zc = zlat - zlat.mean(0, keepdims=True)
s = np.linalg.svd(zc, compute_uv=False)
lam = s**2 / max(n - 1, 1)
p = s / s.sum()
eff_rank_entropy = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
part_ratio = float(lam.sum()**2 / (lam**2).sum())
cum = np.cumsum(lam) / lam.sum()
n90, n99 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.99) + 1)
dim_std = zlat.std(0)
# pairwise cosine similarity of raw latents (how distinguishable are two issue times?)
zu = zlat / np.linalg.norm(zlat, axis=1, keepdims=True)
cos = zu @ zu.T
cos_off = cos[np.triu_indices(n, 1)]
# SIGReg-style sketched Gaussianity: random 1-D projections of the whitened-by-scale latent
rng = np.random.default_rng(0)
dirs = rng.normal(size=(512, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
proj = zc @ dirs.T
proj = (proj - proj.mean(0)) / proj.std(0)
ks = np.array([stats.kstest(proj[:, k], "norm").pvalue for k in range(proj.shape[1])])
ep = np.array([stats.anderson(proj[:, k], "norm").statistic for k in range(proj.shape[1])])

diag = dict(n_issue_times=n, latent_dim=d,
            effective_rank_entropy=eff_rank_entropy, participation_ratio=part_ratio,
            n_components_90pct_var=n90, n_components_99pct_var=n99,
            sv_max=float(s[0]), sv_min=float(s[-1]), sv_ratio=float(s[0] / max(s[-1], 1e-30)),
            dim_std_min=float(dim_std.min()), dim_std_median=float(np.median(dim_std)),
            dim_std_max=float(dim_std.max()),
            dims_below_1pct_of_max_std=int((dim_std < 0.01 * dim_std.max()).sum()),
            mean_pairwise_cosine=float(cos_off.mean()), min_pairwise_cosine=float(cos_off.min()),
            frac_pairs_cosine_above_0p99=float((cos_off > 0.99).mean()),
            sketched_gaussianity_frac_rejected_p05=float((ks < 0.05).mean()),
            sketched_gaussianity_median_ks_p=float(np.median(ks)),
            sketched_anderson_median_stat=float(np.median(ep)))

# ---------- behavioural: how much does the decoded field move? ----------
alloc = rates_l0.reshape(n, -1)
alloc = alloc / alloc.sum(1, keepdims=True)          # spatial-magnitude allocation per window
ac = alloc - alloc.mean(0, keepdims=True)
sa = np.linalg.svd(ac, compute_uv=False)
pa = sa / sa.sum()
diag["allocation_effective_rank_entropy"] = float(np.exp(-(pa * np.log(pa + 1e-300)).sum()))
diag["allocation_frac_var_in_first_pc"] = float(sa[0]**2 / (sa**2).sum())
au = alloc / np.linalg.norm(alloc, axis=1, keepdims=True)
acos = (au @ au.T)[np.triu_indices(n, 1)]
diag["allocation_mean_pairwise_cosine"] = float(acos.mean())
diag["allocation_min_pairwise_cosine"] = float(acos.min())
diag["allocation_total_rate_cv"] = float(rates_l0.sum((1, 2)).std() / rates_l0.sum((1, 2)).mean())

# ---------- is the (static) map right at all? ----------
obs_map = obs[tst].sum(0).sum(1)                       # observed events per cell, test period
l0_map = rates_l0[tst].mean(0).sum(1)
gr_map = rates_gr[tst].mean(0).sum(1)
def sim(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    cs = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return dict(cosine=cs, pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(stats.spearmanr(a, b).statistic))
alloc_rows = [dict(map="EQJEPA_L0 mean rate map", **sim(l0_map, obs_map)),
              dict(map="HistoricalRateGR mean rate map", **sim(gr_map, obs_map)),
              dict(map="EQJEPA_L0 vs HistoricalRateGR", **sim(l0_map, gr_map))]
pd.DataFrame(alloc_rows).to_csv("step2_allocation_similarity.csv", index=False)

# ---------- deliberate-collapse control ----------
ck = torch.load("repo/results/l0_l3/model/eq_jepa_model.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]; model = EQJEPA(cfg); model.load_state_dict(ck["state_dict"]); model.eval()

def decode_from_z(zt: torch.Tensor, h: torch.Tensor) -> np.ndarray:
    """Replicates EQJEPA.forecast_rates from a supplied latent (bypasses the encoder)."""
    with torch.no_grad():
        daily = torch.nn.functional.softplus(model.rate_parent(zt)).squeeze(-1)
        corr = torch.nn.functional.softplus(
            model.horizon_scale(torch.cat([zt, model.horizon(h)], -1))).squeeze(-1)
        total = daily * h * corr
        a = torch.softmax(model.rate_allocation(zt), -1).view(-1, cfg.n_cells, cfg.n_mag)
        return (total[:, None, None] * a).numpy()

# sanity: decoding the stored latents must reproduce the stored rates
chk = decode_from_z(torch.tensor(zlat[tst[:5]], dtype=torch.float32),
                    torch.tensor(horizons[tst[:5]], dtype=torch.float32))
assert np.allclose(chk, rates_l0[tst[:5]], rtol=1e-4, atol=1e-6), "decode_from_z mismatch"

z_bar = torch.tensor(zlat[cal].mean(0, keepdims=True), dtype=torch.float32)
rates_col = np.zeros_like(rates_l0)
for w in range(n):
    rates_col[w] = decode_from_z(z_bar, torch.tensor([float(horizons[w])]))[0]

# ---------- score trained vs collapsed with the same prospective rescale ----------
origins = np.array([[MIN_LON + (c % N_LON) * DH, MIN_LAT + (c // N_LON) * DH]
                    for c in range(N_LAT * N_LON)])
region = CartesianGrid2D.from_origins(origins, dh=DH, magnitudes=MAGS)
df = pd.read_csv(CAT); df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
df = df[df["is_mainshock"].astype(bool)]
df = df[(df.Latitude >= MIN_LAT) & (df.Latitude < MIN_LAT + N_LAT * DH) &
        (df.Longitude >= MIN_LON) & (df.Longitude < MIN_LON + N_LON * DH) & (df.Magnitude >= MAGS[0])]

def catalog_for(w):
    sub = df[(df.Time >= starts[w]) & (df.Time < ends[w])]
    arr = np.zeros(len(sub), dtype=CSEPCatalog.dtype)
    arr["longitude"] = sub.Longitude.to_numpy(); arr["latitude"] = sub.Latitude.to_numpy()
    arr["magnitude"] = sub.Magnitude.to_numpy(); arr["depth"] = sub["Depth/km"].to_numpy()
    arr["origin_time"] = (sub.Time.astype("int64") // 10**6).to_numpy()
    arr["id"] = [str(x).encode() for x in sub.EventID]
    return CSEPCatalog(data=arr, region=region)

obs_tot = obs.sum((1, 2))
def rescale(r):
    c0 = obs_tot[cal].sum() / r[cal].sum((1, 2)).sum()
    return r * c0, float(c0)

variants = {}
for nm, r in (("EQJEPA_L0_trained", rates_l0), ("EQJEPA_L0_collapsed_encoder", rates_col)):
    rr, c0 = rescale(r); variants[nm] = rr
    diag[f"{nm}_scale_c0"] = c0

rows = []
for k, w in enumerate(tst):
    cat = catalog_for(w)
    for nm, rr in variants.items():
        f = GriddedForecast(data=rr[w], region=region, magnitudes=MAGS,
                            start_time=starts[w].to_pydatetime(), end_time=ends[w].to_pydatetime())
        nt = pe.number_test(f, cat); d1, d2 = nt.quantile
        rows.append(dict(window_start=starts[w].strftime("%Y-%m-%d"), model=nm, test="N",
                         q1=float(d1), q2=float(d2), passes=bool(d1 >= ALPHA/2 and d2 >= ALPHA/2)))
        if cat.event_count:
            for tn, fn in (("S", pe.spatial_test), ("M", pe.magnitude_test),
                           ("CL", pe.conditional_likelihood_test)):
                res = fn(f, cat, num_simulations=NSIM, seed=1000 + w)
                rows.append(dict(window_start=starts[w].strftime("%Y-%m-%d"), model=nm, test=tn,
                                 q1=float(res.quantile), q2=np.nan,
                                 passes=bool(res.quantile >= ALPHA)))
    if k % 20 == 0: print(f"  scored {k+1}/{len(tst)}", flush=True)

sc = pd.DataFrame(rows); sc.to_csv("step2_collapsed_control_scores.csv", index=False)
pd.DataFrame([diag]).T.rename(columns={0: "value"}).to_csv("step2_collapse_diagnostics.csv")
pd.DataFrame(dict(component=np.arange(1, len(s) + 1), singular_value=s,
                  variance_fraction=lam / lam.sum(), cumulative=cum,
                  dim_std=np.sort(dim_std)[::-1])).to_csv("step2_latent_spectrum.csv", index=False)
print(sc.pivot_table(index="model", columns="test", values="passes", aggfunc="mean").to_string())
print(json.dumps(diag, indent=2), flush=True)
