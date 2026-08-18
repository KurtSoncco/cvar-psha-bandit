"""Compatibility shim. Import from ``cvar_psha.core.disaggregation``."""

from cvar_psha.core.disaggregation import *  # noqa: F403
from cvar_psha.core.disaggregation import (
    cvar_disaggregation_weights,
    disaggregation_weights,
    exact_var_cvar,
    lognormal_exceedance_prob,
    lognormal_tail_conditional_mean,
    lognormal_tail_mean_unnormalized,
    mixture_cdf,
    solve_mixture_var,
)

__all__ = [
    "lognormal_exceedance_prob",
    "lognormal_tail_mean_unnormalized",
    "lognormal_tail_conditional_mean",
    "mixture_cdf",
    "solve_mixture_var",
    "disaggregation_weights",
    "cvar_disaggregation_weights",
    "exact_var_cvar",
]
