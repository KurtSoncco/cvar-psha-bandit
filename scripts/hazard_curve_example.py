"""Worked example: mean + fractile hazard curves for a continuous PSHA
logic tree, comparing G-PMC AIS ("continuous IS", given the closed-form
conditional hazard) against Hierarchical JEPA-CVaR v2 ("the best that we
have" -- scalar reward only) against the exact quadrature ground truth.

Both methods are adapted ONCE, targeting only the most extreme intensity
level in the curve (the hardest / rarest point), then their logged
(theta, y, iw) samples are reused to reconstruct the ENTIRE hazard curve
(mean and 5/50/95 fractiles) post-hoc -- the standard efficiency trick
behind "a single adaptive-IS run gives you the whole curve," and the
actual deliverable of Houng, Ceferino & Abrahamson (2025).

Usage:
    python scripts/hazard_curve_example.py
Outputs:
    results/hazard_curve_example/mean_hazard_curve.png
    results/hazard_curve_example/fractile_hazard_curves.png
    results/hazard_curve_example/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, ContinuousEpistemicSpec
from cvar_psha.hazard_curve import DEFAULT_FRACTILES, exact_hazard_curve, hazard_curve_from_samples
from cvar_psha.methods.gpmc_ais import run_gpmc_ais
from cvar_psha.methods.jepa_cvar import run_jepa_cvar_v2

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "hazard_curve_example"

PERCENTILES = [0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
BUDGET = 20_000
SEED = 42

PALETTE = {
    "Exact": "#1f1f1f",
    "G-PMC AIS": "#8e44ad",
    "Hierarchical JEPA-CVaR v2": "#c0392b",
}


def build_logic_tree() -> ContinuousEpistemicEnv:
    """Houng et al. (2025) four-parameter continuous epistemic tree."""
    return ContinuousEpistemicEnv(spec=ContinuousEpistemicSpec(), rng=np.random.default_rng(SEED))


def y_grid_from_percentiles(env: ContinuousEpistemicEnv, percentiles: list[float], deg: int = 8) -> np.ndarray:
    from cvar_psha.continuous_ground_truth import compute_continuous_ground_truth

    gt = compute_continuous_ground_truth(env, deg_trunc=deg, deg_normal=deg, target_rate=1e-4)
    return gt.pga_grid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_gt = build_logic_tree()

    print("Continuous logic tree: theta = (b, m_max, dmu, dsigma)")
    s = env_gt.spec
    print(
        f"  b={s.b_mean}±{s.b_std}  mmax={s.mmax_mean}±{s.mmax_std}  "
        f"dmu_std={s.dmu_std}  dsigma_std={s.dsigma_std}  R={s.distance_km} km"
    )

    y_grid = y_grid_from_percentiles(env_gt, PERCENTILES)
    v_extreme = float(y_grid[-1])
    print(f"\nPGA grid ({len(y_grid)} points from ground-truth hazard curve):")
    print(f"  [{y_grid[0]:.4f}, ..., {y_grid[-1]:.4f}] g")
    print(f"Adapting both IS methods toward the rarest grid level: v_extreme={v_extreme:.4f} g")

    print("\nComputing exact mean + fractile hazard curves (quadrature)...")
    exact = exact_hazard_curve(env_gt, y_grid)

    print(f"Running G-PMC AIS (continuous IS, given closed-form target), budget={BUDGET}...")
    env = ContinuousEpistemicEnv(spec=env_gt.spec, rng=np.random.default_rng(SEED + 1))
    res_gpmc = run_gpmc_ais(env, v_extreme, BUDGET, eval_every=2000, log_samples=True)
    gpmc_curve = hazard_curve_from_samples(
        env_gt, res_gpmc.extras["thetas"], res_gpmc.extras["ys"], res_gpmc.extras["iws"], y_grid
    )

    print(f"Running Hierarchical JEPA-CVaR v2 (scalar reward only), budget={BUDGET}...")
    env = ContinuousEpistemicEnv(spec=env_gt.spec, rng=np.random.default_rng(SEED + 2))
    res_v2 = run_jepa_cvar_v2(env, v_extreme, BUDGET, eval_every=2000, log_samples=True, seed=0)
    v2_curve = hazard_curve_from_samples(
        env_gt, res_v2.extras["thetas"], res_v2.extras["ys"], res_v2.extras["iws"], y_grid
    )

    # --- console summary table ---
    print("\n=== Mean hazard curve: H(y) = P(Y>y) ===")
    header = f"{'y':>10s}  {'exact':>12s}  {'G-PMC AIS':>12s}  {'v2':>12s}  {'G-PMC err%':>10s}  {'v2 err%':>10s}"
    print(header)
    for i, y in enumerate(y_grid):
        e, g, v = exact.mean[i], gpmc_curve.mean[i], v2_curve.mean[i]
        ge = 100 * abs(g - e) / e if e > 0 else float("nan")
        ve = 100 * abs(v - e) / e if e > 0 else float("nan")
        print(f"{y:10.4f}  {e:12.6f}  {g:12.6f}  {v:12.6f}  {ge:10.2f}  {ve:10.2f}")

    print("\n=== Fractile hazard curves (5% / 50% / 95%) at each y ===")
    for q in DEFAULT_FRACTILES:
        print(f"-- q={q} --")
        for i, y in enumerate(y_grid):
            e = exact.fractiles[q][i]
            g = gpmc_curve.fractiles[q][i]
            v = v2_curve.fractiles[q][i]
            print(f"  y={y:8.4f}  exact={e:.6f}  G-PMC={g:.6f}  v2={v:.6f}")

    # --- plots ---
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(y_grid, exact.mean, "o-", color=PALETTE["Exact"], linewidth=2.4, label="Exact (quadrature)")
    ax.plot(y_grid, gpmc_curve.mean, "s--", color=PALETTE["G-PMC AIS"], linewidth=1.8,
            label="G-PMC AIS (continuous IS, closed-form target)")
    ax.plot(y_grid, v2_curve.mean, "D--", color=PALETTE["Hierarchical JEPA-CVaR v2"], linewidth=1.8,
            label="Hierarchical JEPA-CVaR v2 (scalar reward only)")
    ax.set_yscale("log")
    ax.set_xlabel("Ground-motion intensity Y")
    ax.set_ylabel("Mean hazard H(y) = P(Y>y)")
    ax.set_title("Mean hazard curve: exact vs continuous IS vs best agent")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    mean_path = OUT_DIR / "mean_hazard_curve.png"
    fig.savefig(mean_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True, sharey=True)
    for ax, (name, curve) in zip(axes, [("G-PMC AIS", gpmc_curve), ("Hierarchical JEPA-CVaR v2", v2_curve)]):
        ax.fill_between(y_grid, exact.fractiles[0.05], exact.fractiles[0.95],
                         color=PALETTE["Exact"], alpha=0.15, label="Exact 5-95% band")
        ax.plot(y_grid, exact.fractiles[0.5], color=PALETTE["Exact"], linewidth=2.2, label="Exact median")
        ax.plot(y_grid, curve.fractiles[0.05], "--", color=PALETTE[name], linewidth=1.6, label=f"{name} 5%")
        ax.plot(y_grid, curve.fractiles[0.5], "-", color=PALETTE[name], linewidth=2.0, label=f"{name} median")
        ax.plot(y_grid, curve.fractiles[0.95], ":", color=PALETTE[name], linewidth=1.6, label=f"{name} 95%")
        ax.set_yscale("log")
        # The 5% fractile decays toward numerical zero at large y (most
        # theta realizations have ~0 hazard there) -- cropping the axis
        # keeps the informative median/95% comparison from being
        # compressed into a sliver by 10+ uninformative orders of magnitude.
        ax.set_ylim(1e-6, 1.0)
        ax.set_xlabel("Ground-motion intensity Y")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Fractile hazard P(Y>y | theta)")
    fig.suptitle("Fractile hazard curves: exact vs each method", fontsize=12)
    fractile_path = OUT_DIR / "fractile_hazard_curves.png"
    fig.savefig(fractile_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved {mean_path}")
    print(f"Saved {fractile_path}")

    summary = {
        "spec": {
            "mu0": env_gt.spec.mu0, "sigma0": env_gt.spec.sigma0,
            "tau_mu": env_gt.spec.tau_mu, "tau_sigma": env_gt.spec.tau_sigma, "s_max": env_gt.spec.s_max,
        },
        "percentiles": PERCENTILES,
        "y_grid": y_grid.tolist(),
        "v_extreme": v_extreme,
        "budget": BUDGET,
        "mean_hazard": {
            "exact": exact.mean.tolist(),
            "gpmc_ais": gpmc_curve.mean.tolist(),
            "jepa_cvar_v2": v2_curve.mean.tolist(),
        },
        "fractile_hazard": {
            str(q): {
                "exact": exact.fractiles[q].tolist(),
                "gpmc_ais": gpmc_curve.fractiles[q].tolist(),
                "jepa_cvar_v2": v2_curve.fractiles[q].tolist(),
            }
            for q in DEFAULT_FRACTILES
        },
        "k_hat_mean_v2": res_v2.extras["k_hat_mean"],
        "ess_gpmc": float(res_gpmc.metrics["ess"][-1]),
        "ess_v2": float(res_v2.metrics["ess"][-1]),
    }
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
