"""L3: nonlinear single-degree-of-freedom (SDOF) structural-response module.

Packages the bilinear kinematic-hardening / cyclic-degrading SDOF integrator
used in the tail-risk pilot (unit mass, explicit central-difference time
stepping) into an installable, tested component of the L0-L3 pipeline:

    L0 (JEPA seismicity forecast) -> L1 (floatCSEP scoreboard)
    -> L2 (scenario ground-motion, Sa(T) + epsilon) -> L3 (this module:
    Sa -> peak ductility -> damage/loss) -> CVaR risk metric.

Physics
-------
Single-degree-of-freedom oscillator, unit mass, natural frequency
``w0 = 2*pi/T1``, viscous damping ratio ``zeta``, integrated with the
*explicit central-difference* method (unconditionally simple, conditionally
stable -- dt must resolve T1; standard practice is dt <= T1/20).

Restoring force is bilinear kinematic hardening about a strength/yield level
``fy = eta * g`` (``eta`` = yield-strength ratio, g = gravitational
acceleration): elastic branch has stiffness ``k0``; post-yield branch has
stiffness ``alpha * k0``. ``alpha < 0`` produces in-cycle softening
("cyclic degradation" / negative post-yield stiffness), the mechanism that
drives dynamic instability (collapse) at large shaking intensity -- this is
what produces IDA-curve flatlining (Vamvatsikos & Cornell, 2002).

An explicit collapse criterion flags a record "dead" once its displacement
exceeds ``mu_coll`` times the yield displacement (or goes non-finite); dead
records are frozen and reported at the collapse ductility rather than
integrated further (this matches the pilot's convention and keeps the
loss-mapping well-defined for excursions that would otherwise diverge).
"""

from __future__ import annotations

import numpy as np

G = 9.81  # m/s^2


def sdof_max_ductility(
    accel: np.ndarray,
    alpha: float,
    T1: float,
    zeta: float,
    eta: float,
    dt: float,
    elastic: bool = False,
    mu_coll: float = 100.0,
) -> np.ndarray:
    """Peak ductility demand for a bank of ground-motion records.

    Parameters
    ----------
    accel : (n_records, n_steps) array, m/s^2 -- ground acceleration histories.
    alpha : post-yield to elastic stiffness ratio (alpha=1 <=> linear elastic;
        0 < alpha < 1 <=> hardening; alpha < 0 <=> softening/degrading).
    T1, zeta : elastic period (s) and damping ratio of the oscillator.
    eta : yield-strength ratio, fy = eta * g (unit mass).
    dt : time step (s) of `accel`; must match the sampling of the records.
    elastic : if True, ignore `alpha` and integrate the linear (never-yielding)
        oscillator -- used for the closed-form validation check.
    mu_coll : ductility threshold beyond which a record is declared collapsed
        and frozen (dynamic-instability criterion).

    Returns
    -------
    (n_records,) array of peak ductility mu = max|u| / uy.
    """
    w0 = 2 * np.pi / T1
    k0 = w0**2
    c = 2 * zeta * w0
    fy = eta * G
    uy = fy / k0

    N = accel.shape[0]
    u_prev = np.zeros(N)
    u = np.zeros(N)
    fh = np.zeros(N)
    fh_cap = (1 - alpha) * fy
    a1 = 1 / dt**2 + c / (2 * dt)
    a2 = 2 / dt**2
    a3 = 1 / dt**2 - c / (2 * dt)
    umax = np.zeros(N)
    dead = np.zeros(N, dtype=bool)

    for i in range(accel.shape[1]):
        fs = k0 * u if elastic else alpha * k0 * u + fh
        un = (-accel[:, i] - fs + a2 * u - a3 * u_prev) / a1
        if not elastic:
            fh = np.clip(fh + (1 - alpha) * k0 * (un - u), -fh_cap, fh_cap)
        u_prev, u = u, np.where(dead, u, un)
        umax = np.maximum(umax, np.abs(u))
        dead |= ~np.isfinite(umax) | (umax > mu_coll * uy)
        umax = np.where(dead, mu_coll * uy * 1.001, umax)

    return umax / uy


def response_surface(
    accel_unit: np.ndarray,
    Sa_grid: np.ndarray,
    Sa0: np.ndarray,
    alpha: float | None,
    T1: float,
    zeta: float,
    eta: float,
    dt: float,
    mu_coll: float = 100.0,
) -> np.ndarray:
    """Ductility response surface mu[Sa_level, record] via intensity scaling.

    `accel_unit` is a bank of unit-PGA-normalised records (m/s^2); each is
    rescaled by (Sa/Sa0) before integration, sweeping an IDA-style intensity
    ladder `Sa_grid` (g) for every record.
    """
    elastic = alpha is None
    out = np.zeros((len(Sa_grid), accel_unit.shape[0]))
    for j, Sa in enumerate(Sa_grid):
        out[j] = sdof_max_ductility(
            accel_unit * (Sa / Sa0)[:, None],
            1.0 if elastic else alpha,
            T1=T1, zeta=zeta, eta=eta, dt=dt,
            elastic=elastic, mu_coll=mu_coll,
        )
    return out


def loss_from_ductility(mu: np.ndarray, mu_coll: float = 100.0) -> np.ndarray:
    """FEMA P-58-flavoured damage-to-loss map: saturating repair-cost ratio
    below collapse, unit loss (total loss) at/above the collapse ductility."""
    collapsed = mu >= mu_coll
    return np.where(collapsed, 1.0, np.minimum(0.90, 0.06 * np.clip(mu, 0, None) ** 1.4))
