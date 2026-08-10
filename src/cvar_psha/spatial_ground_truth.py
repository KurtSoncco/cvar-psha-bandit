"""Ground truth for multi-site portfolio loss L = sum PGA_k."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.spatial_env import SpatialPortfolioEnv


@dataclass(frozen=True)
class SpatialGroundTruth:
    v95: float
    cvar: float
    n: int
    percentile: float
    q_star: np.ndarray
    top_path_indices: np.ndarray
    top_path_labels: list[str]


def _path_label(env: SpatialPortfolioEnv, path: tuple[int, ...]) -> str:
    g, gmm, m, r = path
    return (
        f"{env.geometries[g].name}/"
        f"{env.gmms[gmm].name}/"
        f"M{env.mags[m]:.1f}/"
        f"rup{r}"
    )


def compute_spatial_ground_truth(
    env: SpatialPortfolioEnv,
    n: int = 500_000,
    percentile: float = 0.95,
    top_k: int = 12,
    contrib_samples: int = 128,
) -> SpatialGroundTruth:
    path_ids = env.rng.choice(env.n_paths, size=n, p=env.path_priors)
    path_arr = np.asarray([env.paths[i] for i in path_ids], dtype=int)
    # Chunked batch PGA for memory safety
    chunk = 50_000
    losses = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        pgas = env.sample_pga_field_batch(path_arr[start:end])
        losses[start:end] = pgas.sum(axis=1)

    v = float(np.quantile(losses, percentile))
    tail = losses[losses > v]
    if tail.size == 0:
        raise RuntimeError("No exceedances in spatial ground-truth; increase n.")
    cvar = float(tail.mean())

    contributions = np.zeros(env.n_paths, dtype=float)
    # Evaluate contributions path-by-path in batches of repeated rows
    for pid in range(env.n_paths):
        path = env.paths[pid]
        reps = np.repeat(np.asarray([path], dtype=int), contrib_samples, axis=0)
        vals = env.sample_pga_field_batch(reps).sum(axis=1)
        exceed = vals > v
        if np.any(exceed):
            contributions[pid] = (
                env.path_priors[pid] * float(exceed.mean()) * float(vals[exceed].mean())
            )

    if contributions.sum() <= 0:
        q_star = env.path_priors.copy()
    else:
        q_star = contributions / contributions.sum()

    top = np.argsort(q_star)[::-1][:top_k]
    labels = [_path_label(env, env.paths[int(i)]) for i in top]
    return SpatialGroundTruth(
        v95=v,
        cvar=cvar,
        n=n,
        percentile=percentile,
        q_star=q_star,
        top_path_indices=top,
        top_path_labels=labels,
    )
