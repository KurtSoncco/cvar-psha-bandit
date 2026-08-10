"""2D multi-site portfolio hazard MDP along a fault line.

Decision order (Manager then Worker):
  1. Source geometry (epistemic fault offset)
  2. GMM (epistemic)
  3. Magnitude bin (Gutenberg-Richter)
  4. Rupture location bin along the fault

Leaf metric: L = sum_k PGA_k under a simple distance-attenuated GMPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


NODE_NAMES = ("geometry", "gmm", "magnitude", "rupture")
MANAGER_DEPTHS = (0, 1)
WORKER_DEPTHS = (2, 3)


@dataclass(frozen=True)
class Site:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class GMMBranch:
    name: str
    weight: float
    mu_a: float  # intercept in mu(M) = mu_a + mu_b * (M - Mref)
    mu_b: float
    sigma: float


@dataclass(frozen=True)
class GeometryBranch:
    name: str
    weight: float
    y_offset: float


def gutenberg_richter_weights(
    mags: np.ndarray,
    b: float = 1.0,
    m_min: float | None = None,
) -> np.ndarray:
    m_min = float(mags.min() if m_min is None else m_min)
    w = np.power(10.0, -b * (mags - m_min))
    return w / w.sum()


def default_sites(k: int = 10) -> tuple[Site, ...]:
    """Strip of sites parallel to the fault (y≈15), spanning x in [5, 95]."""
    xs = np.linspace(5.0, 95.0, k)
    return tuple(Site(name=f"site_{i+1}", x=float(x), y=15.0) for i, x in enumerate(xs))


def default_gmms() -> tuple[GMMBranch, ...]:
    return (
        GMMBranch("optimistic", weight=0.4, mu_a=-3.2, mu_b=0.9, sigma=0.55),
        GMMBranch("average", weight=0.4, mu_a=-2.6, mu_b=0.9, sigma=0.55),
        GMMBranch("pessimistic", weight=0.2, mu_a=-2.0, mu_b=0.9, sigma=0.55),
    )


def default_geometries() -> tuple[GeometryBranch, ...]:
    return (
        GeometryBranch("nominal", weight=0.5, y_offset=0.0),
        GeometryBranch("offset", weight=0.5, y_offset=8.0),
    )


class SpatialPortfolioEnv:
    """Fault-line multi-site delayed-reward MDP."""

    def __init__(
        self,
        sites: tuple[Site, ...] | None = None,
        geometries: tuple[GeometryBranch, ...] | None = None,
        gmms: tuple[GMMBranch, ...] | None = None,
        m_min: float = 5.0,
        m_max: float = 8.0,
        m_step: float = 0.2,
        gr_b: float = 1.0,
        n_rupture_bins: int = 20,
        fault_start: tuple[float, float] = (0.0, 0.0),
        fault_end: tuple[float, float] = (100.0, 0.0),
        gamma_atten: float = 1.0,
        r0: float = 5.0,
        m_ref: float = 6.0,
        rng: np.random.Generator | None = None,
    ):
        self.sites = sites or default_sites(10)
        self.geometries = geometries or default_geometries()
        self.gmms = gmms or default_gmms()
        self.mags = np.arange(m_min, m_max + 1e-9, m_step, dtype=float)
        self.mag_priors = gutenberg_richter_weights(self.mags, b=gr_b, m_min=m_min)
        self.n_rupture_bins = int(n_rupture_bins)
        self.rupture_priors = np.full(self.n_rupture_bins, 1.0 / self.n_rupture_bins)
        self.fault_start = np.asarray(fault_start, dtype=float)
        self.fault_end = np.asarray(fault_end, dtype=float)
        self.gamma_atten = float(gamma_atten)
        self.r0 = float(r0)
        self.m_ref = float(m_ref)
        self.rng = rng or np.random.default_rng()

        self.geometry_priors = np.asarray([g.weight for g in self.geometries], dtype=float)
        self.geometry_priors /= self.geometry_priors.sum()
        self.gmm_priors = np.asarray([g.weight for g in self.gmms], dtype=float)
        self.gmm_priors /= self.gmm_priors.sum()

        self.n_actions = [
            len(self.geometries),
            len(self.gmms),
            len(self.mags),
            self.n_rupture_bins,
        ]
        self.depth = 4
        self.node_names = NODE_NAMES

        # Indexed joint paths: (geom, gmm, mag, rup)
        self.paths = [
            tuple(p) for p in product(*[range(n) for n in self.n_actions])
        ]
        self.n_paths = len(self.paths)
        self.path_index = {p: i for i, p in enumerate(self.paths)}
        self.path_priors = np.asarray(
            [self.path_prior_prob(p) for p in self.paths], dtype=float
        )

        self.site_xy = np.asarray([[s.x, s.y] for s in self.sites], dtype=float)

    def reset(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (), {"depth": 0}

    def depth_of(self, state: tuple[int, ...]) -> int:
        return len(state)

    def prior(self, state: tuple[int, ...]) -> np.ndarray:
        d = len(state)
        if d == 0:
            return self.geometry_priors.copy()
        if d == 1:
            return self.gmm_priors.copy()
        if d == 2:
            return self.mag_priors.copy()
        if d == 3:
            return self.rupture_priors.copy()
        raise ValueError(f"No prior at terminal depth {d}")

    def prior_action(self, state: tuple[int, ...], action: int) -> float:
        return float(self.prior(state)[action])

    def path_prior_prob(self, path: tuple[int, ...]) -> float:
        p = 1.0
        state: tuple[int, ...] = ()
        for a in path:
            p *= self.prior_action(state, a)
            state = state + (a,)
        return p

    def rupture_xy(self, geometry_idx: int, rupture_idx: int) -> np.ndarray:
        g = self.geometries[geometry_idx]
        t = (rupture_idx + 0.5) / self.n_rupture_bins
        base = self.fault_start + t * (self.fault_end - self.fault_start)
        return base + np.array([0.0, g.y_offset], dtype=float)

    def sample_pga_field(
        self,
        path: tuple[int, ...],
    ) -> tuple[np.ndarray, float]:
        """Return (PGA vector length K, portfolio L = sum PGA)."""
        pgas = self.sample_pga_field_batch(np.asarray([path], dtype=int))
        return pgas[0], float(pgas[0].sum())

    def sample_pga_field_batch(self, paths: np.ndarray) -> np.ndarray:
        """paths: (n, 4) int array -> PGA matrix (n, K)."""
        paths = np.asarray(paths, dtype=int)
        if paths.ndim == 1:
            paths = paths[None, :]
        n = paths.shape[0]
        geom_i = paths[:, 0]
        gmm_i = paths[:, 1]
        mag_i = paths[:, 2]
        rup_i = paths[:, 3]

        # Rupture positions
        t = (rup_i.astype(float) + 0.5) / self.n_rupture_bins
        base = self.fault_start[None, :] + t[:, None] * (self.fault_end - self.fault_start)[None, :]
        y_off = np.asarray([self.geometries[i].y_offset for i in geom_i], dtype=float)
        rup = base.copy()
        rup[:, 1] += y_off

        m = self.mags[mag_i]
        mu_a = np.asarray([self.gmms[i].mu_a for i in gmm_i], dtype=float)
        mu_b = np.asarray([self.gmms[i].mu_b for i in gmm_i], dtype=float)
        sigma = np.asarray([self.gmms[i].sigma for i in gmm_i], dtype=float)

        # distances (n, K)
        dist = np.linalg.norm(self.site_xy[None, :, :] - rup[:, None, :], axis=2)
        mu = (
            mu_a[:, None]
            + mu_b[:, None] * (m[:, None] - self.m_ref)
            - self.gamma_atten * np.log(dist + self.r0)
        )
        eps = self.rng.normal(0.0, 1.0, size=mu.shape) * sigma[:, None]
        return np.exp(mu + eps)

    def step(
        self,
        state: tuple[int, ...],
        action: int,
    ) -> tuple[tuple[int, ...], float, bool, dict[str, Any]]:
        d = len(state)
        if d >= self.depth:
            raise RuntimeError("Cannot step from a terminal state")
        if action < 0 or action >= self.n_actions[d]:
            raise ValueError(f"Invalid action {action} at depth {d}")

        nxt = state + (action,)
        done = len(nxt) == self.depth
        info: dict[str, Any] = {
            "depth": len(nxt),
            "prior_w": self.prior_action(state, action),
            "action": action,
            "state": state,
            "node": self.node_names[d],
            "is_manager": d in MANAGER_DEPTHS,
            "is_worker": d in WORKER_DEPTHS,
        }
        if done:
            pga, L = self.sample_pga_field(nxt)
            info["pga"] = pga
            info["L"] = L
            info["path"] = nxt
            info["path_idx"] = self.path_index[nxt]
            return nxt, 0.0, True, info
        return nxt, 0.0, False, info

    def rollout_with_policy(
        self,
        action_fn,
    ) -> tuple[tuple[int, ...], float, float, list[dict[str, Any]], np.ndarray]:
        """Roll one episode. action_fn(state, prior) -> (action, q_prob)."""
        state, _ = self.reset()
        iw = 1.0
        traj: list[dict[str, Any]] = []
        done = False
        info: dict[str, Any] = {}
        while not done:
            prior = self.prior(state)
            action, q_prob = action_fn(state, prior)
            q_prob = max(float(q_prob), 1e-12)
            nxt, _r, done, info = self.step(state, action)
            step_w = info["prior_w"] / q_prob
            iw *= step_w
            traj.append(
                {
                    "state": state,
                    "action": action,
                    "prior_w": info["prior_w"],
                    "q_prob": q_prob,
                    "step_iw": step_w,
                    "depth": len(state),
                    "is_manager": info["is_manager"],
                    "is_worker": info["is_worker"],
                }
            )
            state = nxt
        L = float(info["L"])
        pga = info["pga"]
        return state, L, iw, traj, pga

    def sample_path_from_flat_q(
        self,
        q: np.ndarray,
    ) -> tuple[tuple[int, ...], float, float, np.ndarray]:
        q = np.asarray(q, dtype=float)
        q = q / q.sum()
        idx = int(self.rng.choice(self.n_paths, p=q))
        path = self.paths[idx]
        pga, L = self.sample_pga_field(path)
        iw = self.path_priors[idx] / max(q[idx], 1e-12)
        return path, L, iw, pga
