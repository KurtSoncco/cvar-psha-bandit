"""Shared run-result container for sampling methods."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MethodResult:
    name: str
    metrics: dict[str, np.ndarray]
    final_q: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
