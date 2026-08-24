"""Continuous-epistemic-space experiment (paper 2 style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvar_psha.continuous_env import GaussianProposal
from cvar_psha.continuous_ground_truth import compute_continuous_ground_truth, policy_grid_mass
from cvar_psha.core.metrics import kl_divergence, ks_statistic, tv_distance
from cvar_psha.experiment.io import (
    continuous_env_from_config,
    last_metric_mean,
    resolve_out_dir,
    write_summary,
)
from cvar_psha.methods.continuous_methods import (
    grid_moment_match,
    run_continuous_mc,
    run_continuous_oracle,
)
from cvar_psha.methods.cvar_bf import run_cvar_bf_ais
from cvar_psha.methods.gpmc_ais import run_gpmc_ais
from cvar_psha.methods.jepa_cvar import run_jepa_cvar, run_jepa_cvar_v2
from cvar_psha.methods.qr_srm import run_qr_srm_ais
from cvar_psha.plot import (
    plot_analysis_comparisons,
    plot_continuous_theta_comparison,
    plot_hazard_curves,
)


def run_continuous(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    env_gt = continuous_env_from_config(cfg, np.random.default_rng(seed))

    print("Computing Houng PSHA ground truth (product quadrature)...")
    c = cfg.get("continuous", {})
    gt = compute_continuous_ground_truth(
        env_gt,
        deg_trunc=int(c.get("quadrature_deg_trunc", 8)),
        deg_normal=int(c.get("quadrature_deg_normal", 8)),
        percentile=float(cfg.get("percentile", 0.95)),
        target_rate=float(cfg.get("target_rate", 1e-4)),
    )
    mean_disagg = (gt.nodes * gt.q_disagg[:, None]).sum(axis=0)
    mean_star = (gt.nodes * gt.q_star[:, None]).sum(axis=0)
    print(f"  target_rate = {gt.target_rate:.2e} /yr")
    print(f"  v (PGA at target rate) = {gt.v95:.6f} g")
    print(f"  CVaR (E[PGA | PGA > v]) = {gt.cvar:.6f} g")
    print(f"  E[theta | q_disagg] = {mean_disagg}  (Houng, Ceferino & Abrahamson 2025 target)")
    print(f"  E[theta | q_star]   = {mean_star}  (ours, CVaR-optimal)")

    out_dir = resolve_out_dir(cfg, config_path)
    hazard_path = plot_hazard_curves(gt.pga_grid, gt.hazard_curves, out_dir, v_threshold=gt.v95)
    print(f"Saved hazard curves: {hazard_path}")

    oracle_proposal = grid_moment_match(gt.nodes, gt.q_disagg)

    budget = int(cfg.get("budget", 20_000))
    eval_every = int(cfg.get("eval_every", 500))
    n_reps = int(cfg.get("n_replications", 5))
    gpmc_cfg = cfg.get("gpmc_ais", {})
    jepa_cfg = cfg.get("jepa_cvar", {})
    cvar_bf_cfg = cfg.get("cvar_bf", {})
    qr_srm_cfg = cfg.get("qr_srm", {})

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "Disagg-IS oracle": [],
        "G-PMC AIS": [],
        "Hierarchical JEPA-CVaR": [],
        "Hierarchical JEPA-CVaR v2": [],
        "CVaR-BF AIS": [],
        "QR-SRM AIS": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(run_continuous_mc(env, gt.v95, budget, eval_every=eval_every))

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 10))
        method_runs["Disagg-IS oracle"].append(
            run_continuous_oracle(env, gt.v95, budget, oracle_proposal, eval_every=eval_every)
        )

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 1))
        method_runs["G-PMC AIS"].append(
            run_gpmc_ais(
                env,
                gt.v95,
                budget,
                batch_size=int(gpmc_cfg.get("batch_size", 300)),
                smoothing=float(gpmc_cfg.get("smoothing", 0.5)),
                cov_inflation=float(gpmc_cfg.get("cov_inflation", 1.15)),
                defensive_eps=float(gpmc_cfg.get("defensive_eps", 0.1)),
                target=str(gpmc_cfg.get("target", "disagg")),
                eval_every=eval_every,
            )
        )

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 2))
        method_runs["Hierarchical JEPA-CVaR"].append(
            run_jepa_cvar(
                env,
                gt.v95,
                budget,
                manager_lr=float(jepa_cfg.get("manager_lr", 0.02)),
                worker_lr=float(jepa_cfg.get("worker_lr", 0.08)),
                std_start=float(jepa_cfg.get("std_start", 0.9)),
                std_end=float(jepa_cfg.get("std_end", 0.35)),
                jepa_train_every=int(jepa_cfg.get("jepa_train_every", 16)),
                true_cvar=gt.cvar,
                eval_every=eval_every,
                seed=rep,
            )
        )

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 3))
        method_runs["Hierarchical JEPA-CVaR v2"].append(
            run_jepa_cvar_v2(
                env,
                gt.v95,
                budget,
                batch_size=int(jepa_cfg.get("v2_batch_size", 300)),
                smoothing=float(jepa_cfg.get("v2_smoothing", 0.5)),
                cov_inflation=float(jepa_cfg.get("v2_cov_inflation", 1.15)),
                defensive_eps=float(jepa_cfg.get("v2_defensive_eps", 0.1)),
                jepa_train_every=int(jepa_cfg.get("jepa_train_every", 16)),
                enable_joint_tilt=bool(jepa_cfg.get("v2_joint_tilt", True)),
                enable_hierarchical=bool(jepa_cfg.get("v2_hierarchical", True)),
                use_psis=bool(jepa_cfg.get("v2_use_psis", True)),
                mixture_window=int(jepa_cfg.get("v2_mixture_window", 6)),
                tilt_cap=jepa_cfg.get("v2_tilt_cap", 1.0),
                eval_every=eval_every,
                seed=rep,
            )
        )

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 4))
        method_runs["CVaR-BF AIS"].append(
            run_cvar_bf_ais(
                env,
                gt.v95,
                budget,
                learning_rate=float(cvar_bf_cfg.get("learning_rate", 0.05)),
                beta_lr=float(cvar_bf_cfg.get("beta_lr", 0.03)),
                delta_lr=float(cvar_bf_cfg.get("delta_lr", 0.03)),
                std_start=float(cvar_bf_cfg.get("std_start", 0.9)),
                std_end=float(cvar_bf_cfg.get("std_end", 0.35)),
                beta_init=(
                    float(cvar_bf_cfg["beta_init"]) if cvar_bf_cfg.get("beta_init") is not None else None
                ),
                beta_min=float(cvar_bf_cfg.get("beta_min", 0.50)),
                beta_max=float(cvar_bf_cfg.get("beta_max", 0.99)),
                delta_max=float(cvar_bf_cfg.get("delta_max", 0.5)),
                defensive_eps=float(cvar_bf_cfg.get("defensive_eps", 0.1)),
                eval_every=eval_every,
            )
        )

        env = continuous_env_from_config(cfg, np.random.default_rng(rep_seed + 5))
        tail_alpha = 1.0 - float(cfg.get("percentile", 0.95))
        method_runs["QR-SRM AIS"].append(
            run_qr_srm_ais(
                env,
                gt.v95,
                budget,
                batch_size=int(qr_srm_cfg.get("batch_size", 300)),
                smoothing=float(qr_srm_cfg.get("smoothing", 0.5)),
                cov_inflation=float(qr_srm_cfg.get("cov_inflation", 1.15)),
                defensive_eps=float(qr_srm_cfg.get("defensive_eps", 0.1)),
                n_quantiles=int(qr_srm_cfg.get("n_quantiles", 32)),
                spectrum=str(qr_srm_cfg.get("spectrum", "srm_mix")),
                lambda_mean=float(qr_srm_cfg.get("lambda_mean", 0.10)),
                alpha=float(qr_srm_cfg.get("alpha", tail_alpha)),
                qr_lr=float(qr_srm_cfg.get("qr_lr", 0.08)),
                eval_every=eval_every,
            )
        )

    out_dir = resolve_out_dir(cfg, config_path)
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)
    print(f"\nSaved analysis plot: {cvar_path}")
    print(f"Saved analysis plot: {ess_path}")

    summary = {
        "mode": "continuous",
        "target_rate": gt.target_rate,
        "v95": gt.v95,
        "true_cvar": gt.cvar,
        "mean_theta_q_disagg": mean_disagg.tolist(),
        "mean_theta_q_star": mean_star.tolist(),
        "distance_to_targets": {},
        "final_theta_mean": {},
    }

    method_means: dict[str, np.ndarray] = {}
    print("\nFinal results (mean over replications):")
    for name, runs in method_runs.items():
        last_cvar = last_metric_mean(runs, "cvar")
        last_ess = last_metric_mean(runs, "ess")

        means, covs = [], []
        for r in runs:
            if name == "Hierarchical JEPA-CVaR":
                m = r.extras["tail_avg_mean"]
                s = r.extras["final_std"]
                c = np.diag(np.asarray(s, dtype=float) ** 2)
            else:
                m = r.extras["proposal_mean"] if "proposal_mean" in r.extras else r.extras["final_mean"]
                c = r.extras["proposal_cov"] if "proposal_cov" in r.extras else r.extras["final_cov"]
            means.append(m)
            covs.append(c)
        mean = np.mean(np.vstack(means), axis=0)
        cov = np.mean(np.stack(covs), axis=0)
        method_means[name] = mean
        summary["final_theta_mean"][name] = mean.tolist()

        prop = GaussianProposal(mean, cov)
        mass = policy_grid_mass(gt, env_gt.prior, prop.pdf)
        d = {}
        for label, target in (("q_star", gt.q_star), ("q_disagg", gt.q_disagg)):
            d[label] = {
                "kl": kl_divergence(mass, target),
                "tv": tv_distance(mass, target),
                "ks": ks_statistic(mass, target),
            }
        summary["distance_to_targets"][name] = d
        extra_note = ""
        if name == "CVaR-BF AIS":
            beta_m = float(np.mean([r.extras["final_beta"] for r in runs]))
            delta_m = float(np.mean([r.extras["final_delta"] for r in runs]))
            act_m = float(np.mean([r.extras["frac_qp_active"] for r in runs]))
            summary["cvar_bf"] = {
                "final_beta": beta_m,
                "final_delta": delta_m,
                "frac_qp_active": act_m,
            }
            extra_note = f"  beta={beta_m:.3f}  delta={delta_m:.3f}  qp_active={act_m:.2f}"
        if name == "QR-SRM AIS":
            srm_m = float(np.nanmean([r.extras.get("final_srm", np.nan) for r in runs]))
            lam_m = float(np.mean([r.extras.get("lambda_mean", np.nan) for r in runs]))
            spec_m = str(runs[0].extras.get("spectrum", "srm_mix"))
            summary["qr_srm"] = {"final_srm": srm_m, "lambda_mean": lam_m, "spectrum": spec_m}
            extra_note = f"  srm={srm_m:.3f}  spectrum={spec_m}"
        print(
            f"  {name:22s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  theta_mean={mean}{extra_note}\n"
            f"  {'':22s}  vs q_star: KL={d['q_star']['kl']:.3f} TV={d['q_star']['tv']:.3f} KS={d['q_star']['ks']:.3f}"
            f"  |  vs q_disagg: KL={d['q_disagg']['kl']:.3f} TV={d['q_disagg']['tv']:.3f} KS={d['q_disagg']['ks']:.3f}"
        )
    print(f"  {'q_disagg':22s}  theta_mean={mean_disagg}")

    heat_path = plot_continuous_theta_comparison(
        gt.nodes, gt.q_disagg, method_means, out_dir, theta_names=env_gt.theta_names
    )
    print(f"Saved theta-space plot: {heat_path}")

    write_summary(summary, out_dir)
    return summary
