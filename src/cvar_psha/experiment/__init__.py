"""Experiment runners, split by environment."""

from __future__ import annotations

__all__ = [
    "main",
    "run_all",
    "run_1node",
    "run_3node",
    "run_continuous",
    "run_spatial",
    "resolve_config_path",
]


def __getattr__(name: str):
    if name in {"main", "run_all", "resolve_config_path"}:
        from cvar_psha.experiment import cli

        return getattr(cli, name)
    if name == "run_1node":
        from cvar_psha.experiment.run_1node import run_1node

        return run_1node
    if name == "run_3node":
        from cvar_psha.experiment.run_3node import run_3node

        return run_3node
    if name == "run_continuous":
        from cvar_psha.experiment.run_continuous import run_continuous

        return run_continuous
    if name == "run_spatial":
        from cvar_psha.experiment.run_spatial import run_spatial

        return run_spatial
    raise AttributeError(name)
