"""1-node GMM bandit experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvar_psha.experiment.io import (
    bandit_env_from_config,
    last_metric_mean,
    mean_final_q,
    report_distance_to_targets,
    resolve_out_dir,
    write_summary,
)
from cvar_psha.ground_truth import compute_ground_truth
from cvar_psha.methods.cem import run_cem
from cvar_psha.methods.exp3 import run_exp3
from cvar_psha.methods.mc import run_mc, run_oracle
from cvar_psha.methods.reinforce import run_reinforce
from cvar_psha.methods.sto_assign import run_sto_assign
from cvar_psha.plot import plot_analysis_comparisons, plot_learning_curves


def run_1node(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    env_gt = bandit_env_from_config(cfg, np.random.default_rng(seed))

    print("Computing ground truth (large prior MC)...")
    gt = compute_ground_truth(
        env_gt,
        n=int(cfg.get("ground_truth_n", 1_000_000)),
        percentile=float(cfg.get("percentile", 0.95)),
    )
    print(f"  v95      = {gt.v95:.6f}")
    print(f"  CVaR     = {gt.cvar:.6f}")
    print(f"  q_star   = {np.array2string(gt.q_star, precision=4)}  (ours, CVaR-optimal)")
    print(f"  q_disagg = {np.array2string(gt.q_disagg, precision=4)}  (Houng & Ceferino 2025, proven-optimal)")

    budget = int(cfg.get("budget", 20_000))
    eval_every = int(cfg.get("eval_every", 200))
    n_reps = int(cfg.get("n_replications", 5))
    cem_cfg = cfg.get("cem", {})
    exp3_cfg = cfg.get("exp3", {})
    rf_cfg = cfg.get("reinforce", {})
    sto_cfg = cfg.get("sto_assign", {})

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "q* oracle": [],
        "Disagg-IS oracle": [],
        "CEM-IS": [],
        "Exp3": [],
        "REINFORCE": [],
        "CO-STC": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(run_mc(env, gt.v95, budget, eval_every=eval_every))

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 10))
        method_runs["q* oracle"].append(
            run_oracle(env, gt.v95, budget, gt.q_star, name="q* oracle", eval_every=eval_every)
        )

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 11))
        method_runs["Disagg-IS oracle"].append(
            run_oracle(env, gt.v95, budget, gt.q_disagg, name="Disagg-IS oracle", eval_every=eval_every)
        )

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 1))
        method_runs["CEM-IS"].append(
            run_cem(
                env,
                gt.v95,
                budget,
                batch_size=int(cem_cfg.get("batch_size", 500)),
                elite_frac=float(cem_cfg.get("elite_frac", 0.05)),
                smoothing=float(cem_cfg.get("smoothing", 0.7)),
                eval_every=eval_every,
            )
        )

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 2))
        method_runs["Exp3"].append(
            run_exp3(env, gt.v95, budget, gamma=float(exp3_cfg.get("gamma", 0.05)), eval_every=eval_every)
        )

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 3))
        method_runs["REINFORCE"].append(
            run_reinforce(
                env,
                gt.v95,
                budget,
                learning_rate=float(rf_cfg.get("learning_rate", 0.05)),
                baseline_alpha=float(rf_cfg.get("baseline_alpha", 0.1)),
                eval_every=eval_every,
            )
        )

        env = bandit_env_from_config(cfg, np.random.default_rng(rep_seed + 4))
        method_runs["CO-STC"].append(
            run_sto_assign(
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
    fig_path = plot_learning_curves(method_runs, gt.cvar, out_dir, q_star=gt.q_star)
    print(f"\nSaved learning curves to {fig_path}")
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)
    print(f"Saved analysis plot: {cvar_path}")
    print(f"Saved analysis plot: {ess_path}")

    summary = {
        "mode": "1node",
        "v95": gt.v95,
        "true_cvar": gt.cvar,
        "q_star": gt.q_star.tolist(),
        "q_disagg": gt.q_disagg.tolist(),
        "final_policies": {},
    }
    print("\nFinal policies (mean over replications):")
    for name, runs in method_runs.items():
        mean_q = mean_final_q(runs)
        if mean_q is None:
            continue
        summary["final_policies"][name] = mean_q.tolist()
        last_cvar = last_metric_mean(runs, "cvar")
        dist_str = report_distance_to_targets(name, mean_q, gt, summary)
        print(f"  {name:18s}  q={np.array2string(mean_q, precision=4)}  CVaR~{last_cvar:.4f}")
        print(f"  {'':18s}  {dist_str}")
    print(f"  {'q_star':18s}  q={np.array2string(gt.q_star, precision=4)}")
    print(f"  {'q_disagg':18s}  q={np.array2string(gt.q_disagg, precision=4)}")

    write_summary(summary, out_dir)
    return summary
