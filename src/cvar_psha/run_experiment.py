"""Run 1-node or 3-node CVaR PSHA IS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cvar_psha.continuous_env import ContinuousEpistemicEnv, ContinuousEpistemicSpec, GaussianProposal
from cvar_psha.continuous_ground_truth import compute_continuous_ground_truth, policy_grid_mass
from cvar_psha.env import Arm, LogicTreeEnv
from cvar_psha.ground_truth import compute_ground_truth
from cvar_psha.methods.cem import run_cem
from cvar_psha.methods.gpmc_ais import run_gpmc_ais
from cvar_psha.methods.jepa_cvar import run_jepa_cvar
from cvar_psha.methods.cvar_cpo import run_cvar_cpo
from cvar_psha.methods.exp3 import run_exp3
from cvar_psha.methods.hierarchical import run_hierarchical
from cvar_psha.methods.mc import run_mc, run_oracle
from cvar_psha.methods.reinforce import run_reinforce
from cvar_psha.methods.tree_cem import run_tree_cem
from cvar_psha.methods.tree_exp3 import run_tree_exp3
from cvar_psha.methods.tree_mc import run_tree_mc, run_tree_oracle
from cvar_psha.methods.tree_reinforce import run_tree_reinforce
from cvar_psha.metrics import kl_divergence, ks_statistic, tv_distance
from cvar_psha.plot import (
    plot_analysis_comparisons,
    plot_continuous_theta_comparison,
    plot_learning_curves,
    plot_mag_rupture_heatmap,
)
from cvar_psha.spatial_env import GeometryBranch, GMMBranch, SpatialPortfolioEnv, default_sites
from cvar_psha.spatial_ground_truth import compute_spatial_ground_truth
from cvar_psha.methods.spatial_methods import (
    run_spatial_cvar_cpo,
    run_spatial_flat_reinforce,
    run_spatial_hierarchical,
    run_spatial_mc,
    run_spatial_qstar,
)
from cvar_psha.tree_env import Branch, NodeSpec, TreeLogicEnv
from cvar_psha.tree_ground_truth import compute_tree_ground_truth


def _report_distance_to_targets(
    name: str, q: np.ndarray, gt, summary: dict
) -> str:
    """KL/TV/KS of a method's final path distribution against both the
    CVaR-optimal q_star (ours) and the paper-1 proven-optimal q_disagg."""
    d = {}
    for label, target in (("q_star", gt.q_star), ("q_disagg", gt.q_disagg)):
        d[label] = {
            "kl": kl_divergence(q, target),
            "tv": tv_distance(q, target),
            "ks": ks_statistic(q, target),
        }
    summary.setdefault("distance_to_targets", {})[name] = d
    return (
        f"vs q_star: KL={d['q_star']['kl']:.3f} TV={d['q_star']['tv']:.3f} KS={d['q_star']['ks']:.3f}  |  "
        f"vs q_disagg: KL={d['q_disagg']['kl']:.3f} TV={d['q_disagg']['tv']:.3f} KS={d['q_disagg']['ks']:.3f}"
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def arms_from_config(cfg: dict) -> tuple[Arm, ...]:
    return tuple(
        Arm(
            name=a["name"],
            mu=float(a["mu"]),
            sigma=float(a["sigma"]),
            weight=float(a["weight"]),
        )
        for a in cfg["arms"]
    )


def nodes_from_config(cfg: dict) -> tuple[NodeSpec, ...]:
    nodes = []
    for n in cfg["nodes"]:
        branches = []
        for b in n["branches"]:
            branches.append(
                Branch(
                    name=b["name"],
                    weight=float(b["weight"]),
                    delta_mu=float(b.get("delta_mu", 0.0)),
                    mu=float(b["mu"]) if "mu" in b else None,
                    sigma=float(b["sigma"]) if "sigma" in b else None,
                )
            )
        nodes.append(NodeSpec(name=n["name"], branches=tuple(branches)))
    return tuple(nodes)


def _resolve_out_dir(cfg: dict, config_path: Path | None) -> Path:
    out_dir = Path(cfg.get("output_dir", "results"))
    if not out_dir.is_absolute() and config_path is not None:
        repo_root = config_path.resolve().parent.parent
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_1node(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    arms = arms_from_config(cfg)
    env_gt = LogicTreeEnv(arms=arms, rng=np.random.default_rng(seed))

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

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "q* oracle": [],
        "Disagg-IS oracle": [],
        "CEM-IS": [],
        "Exp3": [],
        "REINFORCE": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(run_mc(env, gt.v95, budget, eval_every=eval_every))

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed + 10))
        method_runs["q* oracle"].append(
            run_oracle(env, gt.v95, budget, gt.q_star, name="q* oracle", eval_every=eval_every)
        )

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed + 11))
        method_runs["Disagg-IS oracle"].append(
            run_oracle(
                env, gt.v95, budget, gt.q_disagg, name="Disagg-IS oracle", eval_every=eval_every
            )
        )

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed + 1))
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

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed + 2))
        method_runs["Exp3"].append(
            run_exp3(
                env,
                gt.v95,
                budget,
                gamma=float(exp3_cfg.get("gamma", 0.05)),
                eval_every=eval_every,
            )
        )

        env = LogicTreeEnv(arms=arms, rng=np.random.default_rng(rep_seed + 3))
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

    out_dir = _resolve_out_dir(cfg, config_path)
    fig_path = plot_learning_curves(method_runs, gt.cvar, out_dir, q_star=gt.q_star)
    print(f"\nSaved learning curves to {fig_path}")

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
        qs = [r.final_q for r in runs if r.final_q is not None]
        if not qs:
            continue
        mean_q = np.mean(np.vstack(qs), axis=0)
        summary["final_policies"][name] = mean_q.tolist()
        last_cvar = np.nanmean([r.metrics["cvar"][-1] for r in runs])
        dist_str = _report_distance_to_targets(name, mean_q, gt, summary)
        print(f"  {name:18s}  q={np.array2string(mean_q, precision=4)}  CVaR~{last_cvar:.4f}")
        print(f"  {'':18s}  {dist_str}")
    print(f"  {'q_star':18s}  q={np.array2string(gt.q_star, precision=4)}")
    print(f"  {'q_disagg':18s}  q={np.array2string(gt.q_disagg, precision=4)}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return summary


def run_3node(cfg: dict, config_path: Path | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    nodes = nodes_from_config(cfg)
    env_gt = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(seed))

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

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "q* oracle": [],
        "Disagg-IS oracle": [],
        "CEM-IS": [],
        "Flat Exp3": [],
        "Flat REINFORCE": [],
        "Hierarchical": [],
        "CVaR-CPO": [],
    }

    for rep in range(n_reps):
        rep_seed = seed + 1000 * (rep + 1)
        print(f"\nReplication {rep + 1}/{n_reps} (seed={rep_seed})")

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed))
        method_runs["Naive MC"].append(
            run_tree_mc(env, gt.v95, budget, eval_every=eval_every)
        )

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 10))
        method_runs["q* oracle"].append(
            run_tree_oracle(env, gt.v95, budget, gt.q_star, name="q* oracle", eval_every=eval_every)
        )

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 11))
        method_runs["Disagg-IS oracle"].append(
            run_tree_oracle(
                env, gt.v95, budget, gt.q_disagg, name="Disagg-IS oracle", eval_every=eval_every
            )
        )

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 1))
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

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 2))
        method_runs["Flat Exp3"].append(
            run_tree_exp3(
                env,
                gt.v95,
                budget,
                gamma=float(exp3_cfg.get("gamma", 0.05)),
                eval_every=eval_every,
            )
        )

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 3))
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

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 4))
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

        env = TreeLogicEnv(nodes=nodes, rng=np.random.default_rng(rep_seed + 5))
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

    out_dir = _resolve_out_dir(cfg, config_path)
    fig_path = plot_learning_curves(
        method_runs,
        gt.cvar,
        out_dir,
        q_star=gt.q_star,
        path_labels=gt.path_labels,
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
        qs = [r.final_q for r in runs if r.final_q is not None]
        if not qs:
            continue
        mean_q = np.mean(np.vstack(qs), axis=0)
        summary["final_policies"][name] = mean_q.tolist()
        last_cvar = np.nanmean([r.metrics["cvar"][-1] for r in runs])
        last_ess = np.nanmean([r.metrics["ess"][-1] for r in runs])
        top_i = int(np.argmax(mean_q))
        dist_str = _report_distance_to_targets(name, mean_q, gt, summary)
        print(
            f"  {name:16s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  "
            f"top={gt.path_labels[top_i]} ({mean_q[top_i]:.3f})"
        )
        print(f"  {'':16s}  {dist_str}")
    top_star = int(np.argmax(gt.q_star))
    top_disagg = int(np.argmax(gt.q_disagg))
    print(f"  {'q_star':16s}  top={gt.path_labels[top_star]} ({gt.q_star[top_star]:.3f})")
    print(f"  {'q_disagg':16s}  top={gt.path_labels[top_disagg]} ({gt.q_disagg[top_disagg]:.3f})")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return summary


def spatial_env_from_config(cfg: dict, rng: np.random.Generator) -> SpatialPortfolioEnv:
    sp = cfg["spatial"]
    geometries = tuple(
        GeometryBranch(name=g["name"], weight=float(g["weight"]), y_offset=float(g["y_offset"]))
        for g in sp["geometries"]
    )
    gmms = tuple(
        GMMBranch(
            name=g["name"],
            weight=float(g["weight"]),
            mu_a=float(g["mu_a"]),
            mu_b=float(g["mu_b"]),
            sigma=float(g["sigma"]),
        )
        for g in sp["gmms"]
    )
    return SpatialPortfolioEnv(
        sites=default_sites(int(sp.get("n_sites", 10))),
        geometries=geometries,
        gmms=gmms,
        m_min=float(sp.get("m_min", 5.0)),
        m_max=float(sp.get("m_max", 8.0)),
        m_step=float(sp.get("m_step", 0.2)),
        gr_b=float(sp.get("gr_b", 1.0)),
        n_rupture_bins=int(sp.get("n_rupture_bins", 20)),
        fault_start=tuple(sp.get("fault_start", [0.0, 0.0])),
        fault_end=tuple(sp.get("fault_end", [100.0, 0.0])),
        gamma_atten=float(sp.get("gamma_atten", 1.0)),
        r0=float(sp.get("r0", 5.0)),
        m_ref=float(sp.get("m_ref", 6.0)),
        rng=rng,
    )


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
        method_runs["Naive MC"].append(
            run_spatial_mc(env, gt.v95, budget, eval_every=eval_every)
        )

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

    out_dir = _resolve_out_dir(cfg, config_path)

    def _label(path: tuple[int, ...]) -> str:
        g, gmm, m, r = path
        return (
            f"{env_gt.geometries[g].name}/"
            f"{env_gt.gmms[gmm].name}/"
            f"M{env_gt.mags[m]:.1f}/rup{r}"
        )

    path_labels = [_label(p) for p in env_gt.paths]
    fig_path = plot_learning_curves(
        method_runs,
        gt.cvar,
        out_dir,
        q_star=gt.q_star,
        path_labels=path_labels,
    )
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)

    mean_qs = {}
    for name, runs in method_runs.items():
        qs = [r.final_q for r in runs if r.final_q is not None]
        if qs:
            mean_qs[name] = np.mean(np.vstack(qs), axis=0)
    heat_path = plot_mag_rupture_heatmap(
        env_gt,
        {
            k: mean_qs[k]
            for k in ("Flat REINFORCE", "CVaR-CPO", "Hierarchical")
            if k in mean_qs
        },
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
        last_cvar = np.nanmean([r.metrics["cvar"][-1] for r in runs])
        last_ess = np.nanmean([r.metrics["ess"][-1] for r in runs])
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

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return summary


def continuous_env_from_config(cfg: dict, rng: np.random.Generator) -> ContinuousEpistemicEnv:
    c = cfg.get("continuous", {})
    spec = ContinuousEpistemicSpec(
        mu0=float(c.get("mu0", -1.0)),
        sigma0=float(c.get("sigma0", 0.6)),
        tau_mu=float(c.get("tau_mu", 0.5)),
        tau_sigma=float(c.get("tau_sigma", 0.35)),
        s_max=float(c.get("s_max", 0.5)),
    )
    return ContinuousEpistemicEnv(spec=spec, rng=rng)


def run_continuous(cfg: dict, config_path: Path | None = None) -> dict:
    """Continuous-epistemic-space experiment (paper 2 style): compare
    Naive MC, the closed-form Disagg-IS oracle, G-PMC AIS (given the
    closed-form conditional hazard), and Hierarchical JEPA-CVaR (scalar
    reward only) against the exact quadrature ground truth."""
    from cvar_psha.methods.continuous_methods import (
        grid_moment_match,
        run_continuous_mc,
        run_continuous_oracle,
    )

    seed = int(cfg.get("seed", 42))
    env_gt = continuous_env_from_config(cfg, np.random.default_rng(seed))

    print("Computing continuous-epistemic ground truth (Gauss-Hermite quadrature)...")
    gt = compute_continuous_ground_truth(
        env_gt,
        deg=int(cfg.get("quadrature_deg", 40)),
        percentile=float(cfg.get("percentile", 0.95)),
    )
    mean_disagg = (gt.nodes * gt.q_disagg[:, None]).sum(axis=0)
    mean_star = (gt.nodes * gt.q_star[:, None]).sum(axis=0)
    print(f"  v95      = {gt.v95:.6f}")
    print(f"  CVaR     = {gt.cvar:.6f}")
    print(f"  E[theta | q_disagg] = {mean_disagg}  (Houng, Ceferino & Abrahamson 2025 target)")
    print(f"  E[theta | q_star]   = {mean_star}  (ours, CVaR-optimal)")

    oracle_proposal = grid_moment_match(gt.nodes, gt.q_disagg)

    budget = int(cfg.get("budget", 20_000))
    eval_every = int(cfg.get("eval_every", 500))
    n_reps = int(cfg.get("n_replications", 5))
    gpmc_cfg = cfg.get("gpmc_ais", {})
    jepa_cfg = cfg.get("jepa_cvar", {})

    method_runs: dict[str, list] = {
        "Naive MC": [],
        "Disagg-IS oracle": [],
        "G-PMC AIS": [],
        "Hierarchical JEPA-CVaR": [],
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
                eval_every=eval_every,
                seed=rep,
            )
        )

    out_dir = _resolve_out_dir(cfg, config_path)
    cvar_path, ess_path = plot_analysis_comparisons(method_runs, gt.cvar, out_dir)
    print(f"\nSaved analysis plot: {cvar_path}")
    print(f"Saved analysis plot: {ess_path}")

    summary = {
        "mode": "continuous",
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
        last_cvar = np.nanmean([r.metrics["cvar"][-1] for r in runs])
        last_ess = np.nanmean([r.metrics["ess"][-1] for r in runs])

        means, covs = [], []
        for r in runs:
            if name == "Hierarchical JEPA-CVaR":
                m = r.extras["tail_avg_mean"]
                s = r.extras["final_std"]
                c = np.diag([s * s, s * s])
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
        print(
            f"  {name:22s}  CVaR~{last_cvar:.4f}  ESS~{last_ess:.1f}  theta_mean={mean}\n"
            f"  {'':22s}  vs q_star: KL={d['q_star']['kl']:.3f} TV={d['q_star']['tv']:.3f} KS={d['q_star']['ks']:.3f}"
            f"  |  vs q_disagg: KL={d['q_disagg']['kl']:.3f} TV={d['q_disagg']['tv']:.3f} KS={d['q_disagg']['ks']:.3f}"
        )
    print(f"  {'q_disagg':22s}  theta_mean={mean_disagg}")

    heat_path = plot_continuous_theta_comparison(gt.nodes, gt.q_disagg, method_means, out_dir)
    print(f"Saved theta-space plot: {heat_path}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return summary


def run_all(cfg: dict, config_path: Path | None = None) -> dict:
    if "continuous" in cfg:
        return run_continuous(cfg, config_path=config_path)
    if "spatial" in cfg:
        return run_spatial(cfg, config_path=config_path)
    if "nodes" in cfg:
        return run_3node(cfg, config_path=config_path)
    return run_1node(cfg, config_path=config_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CVaR PSHA bandit IS experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "config.yaml",
        help="Path to experiment YAML config (1-node, 3-node, or spatial)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_all(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
