"""3-node delayed-reward logic-tree experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvar_psha.experiment.io import (
    last_metric_mean,
    mean_final_q,
    report_distance_to_targets,
    resolve_out_dir,
    tree_env_from_config,
    write_summary,
)
from cvar_psha.methods.cvar_cpo import run_cvar_cpo
from cvar_psha.methods.hierarchical import run_hierarchical
from cvar_psha.methods.sto_assign import run_tree_sto_assign
from cvar_psha.methods.tree_cem import run_tree_cem
from cvar_psha.methods.tree_exp3 import run_tree_exp3
from cvar_psha.methods.tree_mc import run_tree_mc, run_tree_oracle
from cvar_psha.methods.tree_reinforce import run_tree_reinforce
from cvar_psha.plot import plot_analysis_comparisons, plot_learning_curves
from cvar_psha.tree_ground_truth import compute_tree_ground_truth


def run_3node(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    env_gt = tree_env_from_config(cfg, np.random.default_rng(seed))

    print("Computing 3-node ground truth (exact closed-form disaggregation)...")
    gt = compute_tree_ground_truth(
        env_gt,
        n=int(cfg.get("ground_truth_n", 1_000_000)),
        percentile=float(cfg.get("percentile", 0.95)),
    )
    print(f"  v95  = {gt.v95:.6f}")
    print(f"  CVaR = {gt.cvar:.6f}")
    print(f"  paths = {env_gt.n_paths}")
    top = np.argsort(gt.q_star)[::-1][:3]
    for i in top:
        print(f"  q_star[{gt.path_labels[i]}] = {gt.q_star[i]:.4f}  q_disagg = {gt.q_disagg[i]:.4f}")

    budget = int(cfg.get("budget", 20_000))
    eval_every = int(cfg.get("eval_every", 200))
    n_reps = int(cfg.get("n_replications", 5))
    cem_cfg = cfg.get("cem", {})
    exp3_cfg = cfg.get("exp3", {})
    rf_cfg = cfg.get("reinforce", {})
    hier_cfg = cfg.get("hierarchical", {})
    cpo_cfg = cfg.get("cvar_cpo", {})
    sto_cfg = cfg.get("sto_assign", {})

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "q* oracle": [],
        "Disagg-IS oracle": [],
        "CEM-IS": [],
        "Flat Exp3": [],
        "Flat REINFORCE": [],
        "Hierarchical": [],
        "CVaR-CPO": [],
        "CO-STC": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(run_tree_mc(env, gt.v95, budget, eval_every=eval_every))

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 10))
        method_runs["q* oracle"].append(
            run_tree_oracle(env, gt.v95, budget, gt.q_star, name="q* oracle", eval_every=eval_every)
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 11))
        method_runs["Disagg-IS oracle"].append(
            run_tree_oracle(
                env, gt.v95, budget, gt.q_disagg, name="Disagg-IS oracle", eval_every=eval_every
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 1))
        method_runs["CEM-IS"].append(
            run_tree_cem(
                env,
                gt.v95,
                budget,
                batch_size=int(cem_cfg.get("batch_size", 500)),
                elite_frac=float(cem_cfg.get("elite_frac", 0.05)),
                smoothing=float(cem_cfg.get("smoothing", 0.7)),
                eval_every=eval_every,
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 2))
        method_runs["Flat Exp3"].append(
            run_tree_exp3(
                env, gt.v95, budget, gamma=float(exp3_cfg.get("gamma", 0.05)), eval_every=eval_every
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 3))
        method_runs["Flat REINFORCE"].append(
            run_tree_reinforce(
                env,
                gt.v95,
                budget,
                learning_rate=float(rf_cfg.get("learning_rate", 0.05)),
                baseline_alpha=float(rf_cfg.get("baseline_alpha", 0.1)),
                eval_every=eval_every,
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 4))
        method_runs["Hierarchical"].append(
            run_hierarchical(
                env,
                gt.v95,
                budget,
                learning_rate=float(hier_cfg.get("learning_rate", 0.05)),
                baseline_alpha=float(hier_cfg.get("baseline_alpha", 0.1)),
                eval_every=eval_every,
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 5))
        method_runs["CVaR-CPO"].append(
            run_cvar_cpo(
                env,
                gt.v95,
                budget,
                learning_rate=float(cpo_cfg.get("learning_rate", 0.05)),
                dual_lr=float(cpo_cfg.get("dual_lr", 0.05)),
                kl_coef=float(cpo_cfg.get("kl_coef", 0.1)),
                cvar_tol=float(cpo_cfg.get("cvar_tol", 0.15)),
                target_ema=float(cpo_cfg.get("target_ema", 0.05)),
                eval_every=eval_every,
                true_cvar=gt.cvar,
            )
        )

        env = tree_env_from_config(cfg, np.random.default_rng(rep_seed + 6))
        method_runs["CO-STC"].append(
            run_tree_sto_assign(
                env,
                gt.v95,
                budget,
                fan_size=int(sto_cfg.get("fan_size", 128)),
                group_size=int(sto_cfg.get("group_size", 8)),
                smoothing=float(sto_cfg.get("smoothing", 0.35)),
                defensive_eps=float(sto_cfg.get("defensive_eps", 0.1)),
                identity_bias=float(sto_cfg.get("identity_bias", 0.0)),
                hidden=int(sto_cfg.get("hidden", 32)),
                ppo_lr=float(sto_cfg.get("ppo_lr", 0.03)),
                ppo_clip=float(sto_cfg.get("ppo_clip", 0.2)),
                ppo_epochs=int(sto_cfg.get("ppo_epochs", 4)),
                entropy_coef=float(sto_cfg.get("entropy_coef", 0.02)),
                kl_stop=float(sto_cfg.get("kl_stop", 0.05)),
                eval_every=eval_every,
            )
        )

    out_dir = resolve_out_dir(cfg, config_path)
    fig_path = plot_learning_curves(
        method_runs, gt.cvar, out_dir, q_star=gt.q_star, path_labels=gt.path_labels
    )
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)
    print(f"\nSaved learning curves to {fig_path}")
    print(f"Saved analysis plot: {cvar_path}")
    print(f"Saved analysis plot: {ess_path}")

    summary = {
        "mode": "3node",
        "v95": gt.v95,
        "true_cvar": gt.cvar,
        "path_labels": gt.path_labels,
        "q_star": gt.q_star.tolist(),
        "q_disagg": gt.q_disagg.tolist(),
        "final_policies": {},
    }
    print("\nFinal path policies (mean over replications):")
    for name, runs in method_runs.items():
        mean_q = mean_final_q(runs)
        if mean_q is None:
            continue
        summary["final_policies"][name] = mean_q.tolist()
        last_cvar = last_metric_mean(runs, "cvar")
        last_ess = last_metric_mean(runs, "ess")
        top_i = int(np.argmax(mean_q))
        dist_str = report_distance_to_targets(name, mean_q, gt, summary)
        print(
            f"  {name:16s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  "
            f"top={gt.path_labels[top_i]} ({mean_q[top_i]:.3f})"
        )
        print(f"  {'':16s}  {dist_str}")
    top_star = int(np.argmax(gt.q_star))
    top_disagg = int(np.argmax(gt.q_disagg))
    print(f"  {'q_star':16s}  top={gt.path_labels[top_star]} ({gt.q_star[top_star]:.3f})")
    print(f"  {'q_disagg':16s}  top={gt.path_labels[top_disagg]} ({gt.q_disagg[top_disagg]:.3f})")

    write_summary(summary, out_dir)
    return summary
