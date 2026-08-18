"""Compatibility shim. Prefer ``cvar_psha.experiment``."""

from cvar_psha.experiment.cli import main, run_all
from cvar_psha.experiment.io import (
    arms_from_config,
    continuous_env_from_config,
    load_config,
    nodes_from_config,
    spatial_env_from_config,
)
from cvar_psha.experiment.run_1node import run_1node
from cvar_psha.experiment.run_3node import run_3node
from cvar_psha.experiment.run_continuous import run_continuous
from cvar_psha.experiment.run_spatial import run_spatial

__all__ = [
    "main",
    "run_all",
    "run_1node",
    "run_3node",
    "run_continuous",
    "run_spatial",
    "load_config",
    "arms_from_config",
    "nodes_from_config",
    "spatial_env_from_config",
    "continuous_env_from_config",
]


if __name__ == "__main__":
    main()
