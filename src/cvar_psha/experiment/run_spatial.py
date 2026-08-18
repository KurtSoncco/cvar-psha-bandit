"""2D multi-site portfolio hazard experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvar_psha.core.metrics import kl_divergence, tv_distance
from cvar_psha.experiment.io import (
    last_metric_mean,
    mean_final_q,
    resolve_out_dir,
    spatial_env_from_config,
    write_summary,
)
from cvar_psha.methods.spatial_methods import (
    run_spatial_cvar_cpo,
    run_spatial_flat_reinforce,
    run_spatial_hierarchical,
    run_spatial_mc,
    run_spatial_qstar,
)
from cvar_psha.plot import plot_analysis_comparisons, plot_learning_curves, plot_mag_rupture_heatmap
from cvar_psha.spatial_ground_truth import compute_spatial_ground_truth


def run_spatial(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    env_gt = spatial_env_from_config(cfg, np.random.default_rng(seed))

    print("Computing spatial portfolio ground truth...")
    print(f"  paths = {env_gt.n_paths}  sites = {len(env_gt.sites)}")
    gt = compute_spatial_ground_truth(
        env_gt,
        n=int(cfg.get("ground_truth_n", 500_000)),
        percentile=float(cfg.get("percentile", 0.95)),
    )
    print(f"  v95  = {gt.v95:.6f}")
    print(f"  CVaR = {gt.cvar:.6f}")
    for lab, idx in zip(gt.top_path_labels[:3], gt.top_path_indices[:3]):
        print(f"  q*[{lab}] = {gt.q_star[int(idx)]:.4f}")

    budget = int(cfg.get("budget", 30_000))
    eval_every = int(cfg.get("eval_every", 500))
    n_reps = int(cfg.get("n_replications", 5))
    rf_cfg = cfg.get("reinforce", {})
    hier_cfg = cfg.get("hierarchical", {})
    cpo_cfg = cfg.get("cvar_cpo", {})

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "q* oracle": [],
        "Flat REINFORCE": [],
        "CVaR-CPO": [],
        "Hierarchical": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = spatial_env_from_config(cfg, np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(run_spatial_mc(env, gt.v95, budget, eval_every=eval_every))

        env = spatial_env_from_config(cfg, np.random.default_rng(rep_seed + 10))
        method_runs["q* oracle"].append(
            run_spatial_qstar(env, gt.v95, budget, gt.q_star, eval_every=eval_every)
        )

        env = spatial_env_from_config(cfg, np.random.default_rng(rep_seed + 1))
        method_runs["Flat REINFORCE"].append(
            run_spatial_flat_reinforce(
                env,
                gt.v95,
                budget,
                learning_rate=float(rf_cfg.get("learning_rate", 0.03)),
                baseline_alpha=float(rf_cfg.get("baseline_alpha", 0.1)),
                eval_every=eval_every,
            )
        )

        env = spatial_env_from_config(cfg, np.random.default_rng(rep_seed + 2))
        method_runs["CVaR-CPO"].append(
            run_spatial_cvar_cpo(
                env,
                gt.v95,
                budget,
                learning_rate=float(cpo_cfg.get("learning_rate", 0.03)),
                dual_lr=float(cpo_cfg.get("dual_lr", 0.05)),
                kl_coef=float(cpo_cfg.get("kl_coef", 0.15)),
                prior_kl_coef=float(cpo_cfg.get("prior_kl_coef", 0.25)),
                cvar_tol=float(cpo_cfg.get("cvar_tol", 0.15)),
                target_ema=float(cpo_cfg.get("target_ema", 0.05)),
                eval_every=eval_every,
                true_cvar=gt.cvar,
            )
        )

        env = spatial_env_from_config(cfg, np.random.default_rng(rep_seed + 3))
        method_runs["Hierarchical"].append(
            run_spatial_hierarchical(
                env,
                gt.v95,
                budget,
                learning_rate=float(hier_cfg.get("learning_rate", 0.04)),
                baseline_alpha=float(hier_cfg.get("baseline_alpha", 0.1)),
                eval_every=eval_every,
                manager_lr_scale=float(hier_cfg.get("manager_lr_scale", 0.4)),
                worker_lr_scale=float(hier_cfg.get("worker_lr_scale", 1.0)),
                temp_start=float(hier_cfg.get("temp_start", 1.4)),
                temp_end=float(hier_cfg.get("temp_end", 0.95)),
                entropy_start=float(hier_cfg.get("entropy_start", 0.12)),
                entropy_end=float(hier_cfg.get("entropy_end", 0.02)),
                gae_lambda=float(hier_cfg.get("gae_lambda", 0.9)),
                soft_tau_start=float(hier_cfg.get("soft_tau_start", 0.01)),
                soft_tau_end=float(hier_cfg.get("soft_tau_end", 0.002)),
                soft_mix_start=float(hier_cfg.get("soft_mix_start", 0.2)),
                soft_mix_end=float(hier_cfg.get("soft_mix_end", 0.0)),
                prior_mix=float(hier_cfg.get("prior_mix", 0.02)),
                q_star=gt.q_star,
            )
        )

    out_dir = resolve_out_dir(cfg, config_path)

    def _label(path: tuple[int, ...]) -> str:
        g, gmm, m, r = path
        return (
            f"{env_gt.geometries[g].name}/"
            f"{env_gt.gmms[gmm].name}/"
            f"M{env_gt.mags[m]:.1f}/rup{r}"
        )

    path_labels = [_label(p) for p in env_gt.paths]
    fig_path = plot_learning_curves(
        method_runs, gt.cvar, out_dir, q_star=gt.q_star, path_labels=path_labels
    )
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)

    mean_qs = {}
    for name, runs in method_runs.items():
        q = mean_final_q(runs)
        if q is not None:
            mean_qs[name] = q
    heat_path = plot_mag_rupture_heatmap(
        env_gt,
        {k: mean_qs[k] for k in ("Flat REINFORCE", "CVaR-CPO", "Hierarchical") if k in mean_qs},
        gt.q_star,
        out_dir,
    )

    print(f"\nSaved learning curves to {fig_path}")
    print(f"Saved analysis plot: {cvar_path}")
    print(f"Saved analysis plot: {ess_path}")
    print(f"Saved heatmap: {heat_path}")

    summary = {
        "mode": "spatial",
        "n_paths": env_gt.n_paths,
        "n_sites": len(env_gt.sites),
        "v95": gt.v95,
        "true_cvar": gt.cvar,
        "top_path_labels": gt.top_path_labels,
        "top_q_star": [float(gt.q_star[i]) for i in gt.top_path_indices],
        "final_policy_top": {},
        "distance_to_qstar": {},
    }
    top20 = [int(i) for i in np.argsort(gt.q_star)[::-1][:20]]
    print("\nFinal results (mean over replications):")
    for name, runs in method_runs.items():
        last_cvar = last_metric_mean(runs, "cvar")
        last_ess = last_metric_mean(runs, "ess")
        q = mean_qs.get(name)
        if q is None:
            print(f"  {name:14s}  CVaR~{last_cvar:.4f}")
            continue
        top_i = int(np.argmax(q))
        top_lab = path_labels[top_i]
        top_p = float(q[top_i])
        summary["final_policy_top"][name] = {"label": top_lab, "prob": top_p}
        if name != "q* oracle":
            kl = kl_divergence(q, gt.q_star)
            tv = tv_distance(q, gt.q_star)
            top_mass = float(np.sum(q[top20]))
            summary["distance_to_qstar"][name] = {
                "kl": kl,
                "tv": tv,
                "mass_on_top20_qstar": top_mass,
            }
            print(
                f"  {name:14s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  "
                f"KL={kl:.3f}  TV={tv:.3f}  top20mass={top_mass:.3f}  "
                f"top={top_lab} ({top_p:.3f})"
            )
        else:
            print(
                f"  {name:14s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  "
                f"top={top_lab} ({top_p:.3f})"
            )

    write_summary(summary, out_dir)
    return summary
