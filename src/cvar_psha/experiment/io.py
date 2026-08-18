"""Config loading, output paths, and shared experiment reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from cvar_psha.core.metrics import kl_divergence, ks_statistic, tv_distance
from cvar_psha.core.result import MethodResult
from cvar_psha.env import Arm, LogicTreeEnv
from cvar_psha.tree_env import Branch, NodeSpec, TreeLogicEnv
from cvar_psha.continuous_env import ContinuousEpistemicEnv, ContinuousEpistemicSpec
from cvar_psha.spatial_env import GeometryBranch, GMMBranch, SpatialPortfolioEnv, default_sites


def repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or this file) until ``pyproject.toml`` is found."""
    here = start.resolve() if start is not None else Path(__file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    return Path(__file__).resolve().parents[3]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_out_dir(cfg: dict, config_path: Path | None) -> Path:
    out_dir = Path(cfg.get("output_dir", "results"))
    if not out_dir.is_absolute():
        root = repo_root(config_path) if config_path is not None else repo_root()
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_summary(summary: dict, out_dir: Path) -> Path:
    path = out_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {path}")
    return path


def mean_final_q(runs: list[MethodResult]) -> np.ndarray | None:
    qs = [r.final_q for r in runs if r.final_q is not None]
    if not qs:
        return None
    return np.mean(np.vstack(qs), axis=0)


def last_metric_mean(runs: list[MethodResult], key: str) -> float:
    return float(np.nanmean([r.metrics[key][-1] for r in runs]))


def report_distance_to_targets(name: str, q: np.ndarray, gt, summary: dict) -> str:
    """KL/TV/KS of a method's final path distribution against q_star and q_disagg."""
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


def bandit_env_from_config(cfg: dict, rng: np.random.Generator) -> LogicTreeEnv:
    return LogicTreeEnv(arms=arms_from_config(cfg), rng=rng)


def tree_env_from_config(cfg: dict, rng: np.random.Generator) -> TreeLogicEnv:
    return TreeLogicEnv(nodes=nodes_from_config(cfg), rng=rng)
