"""Unit tests for the L3 nonlinear-SDOF module.

Core check is the closed-form linear limit: for a *linear* (never-yielding)
oscillator, ductility demand is an exact ratio of elastic spectral
acceleration to the notional yield-strength level, mu_d = Sa / (eta * g),
for ANY period and ANY input record -- this is not the "equal displacement"
*approximation* used for nonlinear systems, it is exact algebra for a
linear system. It is the sharpest available correctness check on the
integrator's elastic branch.
"""
import numpy as np
import pytest

from cvar_psha.eq_jepa.sdof import G, sdof_max_ductility, response_surface, loss_from_ductility


def _sine_bank(n_records=5, dt=0.005, duration=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    freqs = rng.uniform(0.5, 5.0, n_records)
    amps = rng.uniform(0.5, 2.0, n_records)  # m/s^2
    accel = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :])
    return accel, dt


def _nigam_jennings_Sa(accel, dt, T, zeta=0.05):
    """Independent reference elastic-response solver (exact piecewise-linear
    recursion, Nigam & Jennings 1969) -- a different numerical algorithm from
    the central-difference integrator under test, used ONLY as ground truth
    for the closed-form-linear cross-check below."""
    if T < 1e-6:
        return np.max(np.abs(accel))
    wn = 2 * np.pi / T
    wd = wn * np.sqrt(1 - zeta**2)
    w2 = wn**2
    ex = np.exp(-zeta * wn * dt)
    cwd, swd = np.cos(wd * dt), np.sin(wd * dt)
    A = ex * (zeta / np.sqrt(1 - zeta**2) * swd + cwd)
    B = ex * (swd / wd)
    C_ = (1 / w2) * (2 * zeta / (wn * dt) + ex * (((1 - 2 * zeta**2) / (wd * dt)
         - zeta / np.sqrt(1 - zeta**2)) * swd - (1 + 2 * zeta / (wn * dt)) * cwd))
    D_ = (1 / w2) * (1 - 2 * zeta / (wn * dt) + ex * ((2 * zeta**2 - 1) / (wd * dt)
         * swd + 2 * zeta / (wn * dt) * cwd))
    Ap = -ex * (wn / np.sqrt(1 - zeta**2) * swd)
    Bp = ex * (cwd - zeta / np.sqrt(1 - zeta**2) * swd)
    Cp = (1 / w2) * (-1 / dt + ex * ((wn / np.sqrt(1 - zeta**2)
         + zeta / (dt * np.sqrt(1 - zeta**2))) * swd + cwd / dt))
    Dp = (1 / w2) * (1 / dt - ex * (zeta / (dt * np.sqrt(1 - zeta**2)) * swd + cwd / dt))
    n = len(accel)
    u, v = np.zeros(n), np.zeros(n)
    for i in range(n - 1):
        u[i + 1] = A * u[i] + B * v[i] + C_ * accel[i] + D_ * accel[i + 1]
        v[i + 1] = Ap * u[i] + Bp * v[i] + Cp * accel[i] + Dp * accel[i + 1]
    abs_accel = -2 * zeta * wn * v - w2 * u
    return np.max(np.abs(abs_accel))


def test_linear_closed_form_ductility():
    """mu_d = Sa / (eta * g) exactly, for the elastic (never-yielding) branch.

    This is not the "equal displacement" *approximation* used to relate
    inelastic and elastic demand for nonlinear systems -- for a genuinely
    linear system it is exact algebra: mu = umax/uy = (Sa/k0)/(fy/k0) =
    Sa/fy = Sa/(eta*g). We cross-check the module's central-difference
    integrator against an independently-implemented exact recursive solver
    (Nigam & Jennings, 1969) computing Sa for the same records/period.
    """
    accel, dt = _sine_bank(n_records=8, seed=1)
    T1, zeta, eta = 0.5, 0.05, 0.20

    mu = sdof_max_ductility(accel, alpha=1.0, T1=T1, zeta=zeta, eta=eta, dt=dt,
                             elastic=True, mu_coll=1e9)

    Sa_ref_g = np.array([_nigam_jennings_Sa(accel[i], dt, T1, zeta) / G
                          for i in range(accel.shape[0])])
    mu_expected = Sa_ref_g / eta

    # central-difference (this module) vs exact recursion (reference): should
    # agree to within a few % (both are numerically exact for a linear ODE;
    # residual gap is discretization-scheme, not physics).
    np.testing.assert_allclose(mu, mu_expected, rtol=0.05)


def test_zero_input_gives_zero_ductility():
    accel = np.zeros((3, 200))
    mu = sdof_max_ductility(accel, alpha=0.5, T1=0.5, zeta=0.05, eta=0.2,
                             dt=0.005, mu_coll=1e9)
    assert np.allclose(mu, 0.0)


def test_softening_collapses_before_hardening():
    """At high intensity, a degrading (alpha<0) system should reach a higher
    (or equal) ductility than a hardening (alpha>0) system under the same
    input -- softening cannot be structurally *more* stable than hardening."""
    accel, dt = _sine_bank(n_records=6, seed=2)
    kwargs = dict(T1=0.5, zeta=0.05, eta=0.15, dt=dt, mu_coll=50.0)
    scale = 6.0  # push well past yield
    mu_hard = sdof_max_ductility(accel * scale, alpha=0.05, **kwargs)
    mu_soft = sdof_max_ductility(accel * scale, alpha=-0.05, **kwargs)
    assert np.all(mu_soft >= mu_hard - 1e-8)


def test_collapse_criterion_freezes_at_threshold():
    """A record driven far past mu_coll must be reported at the collapse
    ductility, not diverge to inf/nan."""
    accel, dt = _sine_bank(n_records=4, seed=3)
    mu_coll = 10.0
    mu = sdof_max_ductility(accel * 50.0, alpha=-0.1, T1=0.5, zeta=0.05,
                             eta=0.1, dt=dt, mu_coll=mu_coll)
    assert np.all(np.isfinite(mu))
    assert np.all(mu >= mu_coll - 1e-6)


def test_loss_from_ductility_bounds():
    mu = np.array([0.0, 1.0, 5.0, 100.0, 1e6])
    loss = loss_from_ductility(mu, mu_coll=100.0)
    assert loss[0] == 0.0
    assert loss[-2] == 1.0 and loss[-1] == 1.0  # collapse -> total loss
    assert np.all((loss >= 0) & (loss <= 1))
    assert np.all(np.diff(loss[:-2]) >= 0)  # monotone non-decreasing pre-collapse


def test_response_surface_monotone_in_intensity():
    """IDA sanity: for a hardening/ductile system, median ductility should be
    non-decreasing as scaled intensity Sa increases along the ladder."""
    accel, dt = _sine_bank(n_records=20, seed=4)
    Sa0 = np.full(accel.shape[0], 0.3)  # arbitrary reference elastic Sa (g)
    Sa_grid = np.logspace(np.log10(0.05), np.log10(2.0), 15)
    surf = response_surface(accel, Sa_grid, Sa0, alpha=0.05, T1=0.5, zeta=0.05,
                             eta=0.15, dt=dt, mu_coll=50.0)
    med = np.median(surf, axis=1)
    assert np.all(np.diff(med) >= -1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
