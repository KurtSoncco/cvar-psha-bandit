"""CLI: ``python scripts/run.py --config 1node`` or a YAML path."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvar_psha.experiment.io import load_config, repo_root
from cvar_psha.experiment.run_1node import run_1node
from cvar_psha.experiment.run_3node import run_3node
from cvar_psha.experiment.run_continuous import run_continuous
from cvar_psha.experiment.run_spatial import run_spatial

CONFIG_ALIASES = {
    "1node": Path("experiments") / "1node" / "config.yaml",
    "3node": Path("experiments") / "3node" / "config.yaml",
    "continuous": Path("experiments") / "continuous" / "config.yaml",
    "spatial": Path("experiments") / "spatial" / "config.yaml",
}


def resolve_config_path(raw: str | Path) -> Path:
    text = str(raw)
    root = repo_root()
    if text in CONFIG_ALIASES:
        return root / CONFIG_ALIASES[text]
    path = Path(text)
    if not path.is_absolute():
        path = (root / path).resolve() if not path.exists() else path.resolve()
    return path


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
        type=str,
        default="1node",
        help="YAML path, or alias: 1node | 3node | continuous | spatial",
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    run_all(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
