"""Step 4: the two defensible forms of the energy idea.

(1) Moment-rate budget. Kostrov summation over each forecast's magnitude-frequency
    distribution gives an implied seismic moment rate, compared against the rate the
    catalog actually released. Within-bin magnitudes are integrated under a truncated
    Gutenberg-Richter law with b estimated from the catalog (Aki-Utsu), because
    M0 ~ 10^(1.5m) grows faster than 10^(-bm) decays for b < 1.5: every bin's moment is
    dominated by its upper edge, so the result is sensitive to b and to the top bin's
    truncation. Both sensitivities are reported rather than hidden.

(2) An energy consistency test (E-test), built on pycsep's simulate-and-rank pattern:
    draw 1000 synthetic catalogs from the forecast (Poisson counts per cell/magnitude
    bin, magnitudes within a bin from the same truncated GR law), total the radiated
    energy of each, and rank the observed total against them. This asks whether the
    forecast got the moment/energy release right, which is the quantity a CVaR
    objective is actually exposed to -- unlike the N-test, which weights an M3 the
    same as an M7.

Outputs: step4_moment_rate.csv, step4_etest.csv, step4_etest_summary.csv,
step4_b_value.json
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from csep_scoring import load_independent_catalog, MAGS
MAG_EDGES = np.array([2.8, 3.5, 4.5, 5.5, 8.5])
NSIM = 1000
SEC_PER_YEAR = 365.25 * 86400

def aki_utsu_b(mags: np.ndarray, mc: float, dm: float = 0.1) -> float:
    m = mags[mags >= mc - 1e-9]
    return float(1.0 / (np.log(10) * (m.mean() - (mc - dm / 2))))

def gr_bin_weights(edges: np.ndarray, b: float) -> np.ndarray:
    """Probability of each magnitude bin under a truncated GR law on [edges[0], edges[-1]]."""
    cdf = 1 - 10 ** (-b * (edges - edges[0]))
    cdf = cdf / cdf[-1]
    return np.diff(cdf)

def bin_mean_moment(edges: np.ndarray, b: float) -> np.ndarray:
    """E[M0 | event in bin] under a truncated GR law, in N m (Hanks & Kanamori 1979)."""
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = np.linspace(edges[i], edges[i + 1], 4001)
        pdf = np.log(10) * b * 10 ** (-b * (m - edges[i]))
        m0 = 10 ** (1.5 * m + 9.05)
        out[i] = np.trapezoid(pdf * m0, m) / np.trapezoid(pdf, m)
    return out

def bin_mean_energy(edges: np.ndarray, b: float) -> np.ndarray:
    """E[E_s | event in bin], Choy & Boatwright (1995): log10 E_s = 1.5 m + 4.4, in J."""
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = np.linspace(edges[i], edges[i + 1], 4001)
        pdf = np.log(10) * b * 10 ** (-b * (m - edges[i]))
        es = 10 ** (1.5 * m + 4.4)
        out[i] = np.trapezoid(pdf * es, m) / np.trapezoid(pdf, m)
    return out

def sample_bin_magnitudes(n: int, lo: float, hi: float, b: float, rng) -> np.ndarray:
    u = rng.random(n)
    return lo - np.log10(1 - u * (1 - 10 ** (-b * (hi - lo)))) / b

def main():
    df = load_independent_catalog()
    b = aki_utsu_b(df.Magnitude.to_numpy(), mc=2.8)
    b_full = aki_utsu_b(pd.read_csv("repo/results/l0_l3/data/california_catalog_declustered_vetted.csv")
                        .Magnitude.to_numpy(), mc=2.8)
    json.dump(dict(b_declustered=b, b_full_catalog=b_full, mc=2.8), open("step4_b_value.json", "w"),
              indent=2)
    print(f"b (declustered) = {b:.3f}, b (full catalog) = {b_full:.3f}", flush=True)

    z1 = np.load("forecasts_step1.npz", allow_pickle=False)
    z3 = np.load("step3_arms.npz", allow_pickle=False)
    starts = pd.to_datetime([s for s in z1["starts"]], utc=True)
    ends = pd.to_datetime([s for s in z1["ends"]], utc=True)
    cal = np.where(starts < pd.Timestamp("2019-01-01", tz="UTC"))[0]
    tst = np.where(starts >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
    obs = z3["obs"]

    variants = {}
    for nm, key, src in (("EQJEPA_L0_original", "rates_l0", z1),
                         ("HistoricalRateGR", "rates_gr", z1),
                         ("ETASLite", "rates_etas", z1)):
        r = src[key]; variants[nm] = r * (obs[cal].sum() / r[cal].sum())
    for a in [k[len("rates_"):] for k in z3.files if k.startswith("rates_")]:
        r = z3[f"rates_{a}"]; variants[a] = r * (obs[cal].sum() / r[cal].sum())

    m0_bin = bin_mean_moment(MAG_EDGES, b)
    es_bin = bin_mean_energy(MAG_EDGES, b)
    m0_bin_b1 = bin_mean_moment(MAG_EDGES, 1.0)
    m0_bin_trunc7 = bin_mean_moment(np.array([2.8, 3.5, 4.5, 5.5, 7.5]), b)

    # ---------- (1) moment-rate budget ----------
    dur_years = float((ends[tst[-1]] - starts[tst[0]]).total_seconds() / SEC_PER_YEAR)
    rows = []
    obs_ev = df[(df.Time >= starts[tst[0]]) & (df.Time < ends[tst[-1]])]
    obs_m0 = (10 ** (1.5 * obs_ev.Magnitude.to_numpy() + 9.05)).sum() / dur_years
    full = pd.read_csv("repo/results/l0_l3/data/california_catalog_declustered_vetted.csv")
    full["Time"] = pd.to_datetime(full["Time"], format="mixed", utc=True)
    full_ev = full[(full.Time >= starts[tst[0]]) & (full.Time < ends[tst[-1]])]
    full_m0 = (10 ** (1.5 * full_ev.Magnitude.to_numpy() + 9.05)).sum() / dur_years
    rows.append(dict(source="observed (declustered catalog)", moment_rate_Nm_per_yr=obs_m0,
                     note=f"{len(obs_ev)} independent events, {dur_years:.2f} yr"))
    rows.append(dict(source="observed (full catalog, incl. aftershocks)",
                     moment_rate_Nm_per_yr=full_m0,
                     note=f"{len(full_ev)} events; aftershocks carry real moment"))
    for nm, r in variants.items():
        lam = r[tst].sum((0, 1))                     # expected counts per magnitude bin
        rows.append(dict(source=nm, moment_rate_Nm_per_yr=float((lam * m0_bin).sum() / dur_years),
                         note="Kostrov, b from catalog"))
        rows.append(dict(source=nm + " [b=1.0]",
                         moment_rate_Nm_per_yr=float((lam * m0_bin_b1).sum() / dur_years),
                         note="sensitivity to b"))
        rows.append(dict(source=nm + " [top bin truncated at 7.5]",
                         moment_rate_Nm_per_yr=float((lam * m0_bin_trunc7).sum() / dur_years),
                         note="sensitivity to upper truncation"))
    pd.DataFrame(rows).to_csv("step4_moment_rate.csv", index=False)

    # ---------- (2) E-test ----------
    rng = np.random.default_rng(7)
    erows = []
    for w in tst:
        ev = df[(df.Time >= starts[w]) & (df.Time < ends[w])]
        obs_e = float((10 ** (1.5 * ev.Magnitude.to_numpy() + 4.4)).sum())
        obs_n = len(ev)
        for nm, r in variants.items():
            lam = r[w].sum(0)                                   # per magnitude bin
            sims = np.zeros(NSIM)
            for j in range(len(lam)):
                n = rng.poisson(lam[j], NSIM)
                tot = n.sum()
                if tot:
                    mags = sample_bin_magnitudes(tot, MAG_EDGES[j], MAG_EDGES[j + 1], b, rng)
                    e = 10 ** (1.5 * mags + 4.4)
                    idx = np.repeat(np.arange(NSIM), n)
                    sims += np.bincount(idx, weights=e, minlength=NSIM)
            q = float((sims < obs_e).mean())
            erows.append(dict(window_start=starts[w].strftime("%Y-%m-%d"), model=nm,
                              observed_energy_J=obs_e, observed_n=obs_n,
                              expected_energy_J=float((lam * es_bin).sum()),
                              median_simulated_J=float(np.median(sims)), quantile=q,
                              passes=bool(0.025 <= q <= 0.975)))
    e = pd.DataFrame(erows); e.to_csv("step4_etest.csv", index=False)
    summ = e.groupby("model").agg(etest_pass_pct=("passes", lambda s: 100 * s.mean()),
                                  median_quantile=("quantile", "median"),
                                  frac_quantile_below_0p025=("quantile", lambda s: (s < 0.025).mean()),
                                  frac_quantile_above_0p975=("quantile", lambda s: (s > 0.975).mean()),
                                  mean_expected_energy_J=("expected_energy_J", "mean"),
                                  mean_observed_energy_J=("observed_energy_J", "mean")).round(4)
    summ.to_csv("step4_etest_summary.csv")
    print(summ.to_string(), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
