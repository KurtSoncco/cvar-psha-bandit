#!/usr/bin/env python3
"""Build Houng-style mean + fractile hazard curves without running IS methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from cvar_psha.continuous_ground_truth import compute_continuous_ground_truth
from cvar_psha.plot import plot_hazard_curves
from cvar_psha.run_experiment import continuous_env_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PSHA hazard curves from ground truth only")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "config_continuous.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    env = continuous_env_from_config(cfg, np.random.default_rng(seed))
    c = cfg.get("continuous", {})
    gt = compute_continuous_ground_truth(
        env,
        deg_trunc=int(c.get("quadrature_deg_trunc", 8)),
        deg_normal=int(c.get("quadrature_deg_normal", 8)),
        target_rate=float(cfg.get("target_rate", 1e-4)),
    )

    out_dir = Path(args.output_dir or cfg.get("output_dir", "results/continuous"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = plot_hazard_curves(gt.pga_grid, gt.hazard_curves, out_dir, v_threshold=gt.v95)

    print(f"Target rate: {gt.target_rate:.2e} /yr")
    print(f"PGA at target rate: {gt.v95:.4f} g")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
