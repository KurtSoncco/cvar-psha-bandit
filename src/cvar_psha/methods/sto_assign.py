"""CO-STC: control-oriented scenario-tree construction (Pavirani et al. 2026)
for discrete PSHA importance sampling.

Source paper
------------
Pavirani, F., Claessens, B., Pinson, P. & Develder, C. (2026).
Control-Oriented Scenario Tree Construction through Reinforcement Learning.
https://arxiv.org/abs/2608.09335

That paper is multistage stochastic MPC for battery arbitrage, not PSHA:
they fix a tree topology, treat construction as sequential assignment of a
forecast *fan* onto the leaves, and train an attention-based assignment
policy with PPO so that the *realized closed-loop control profit* (a
risk-averse CVaR program on the constructed tree) is the only objective —
not Wasserstein / nested-distance fidelity to the forecast.

What transfers here, and what does not
--------------------------------------
Used, faithfully:
  * Fixed topology whose leaves are the discrete IS support (1-node: K GMM
    arms; 3-node: the 12 complete Source→Magnitude→GMM paths).
  * Sequential grouped assignment of a sampled fan onto those leaves, with
    atypical scenarios first (their §4.1).
  * DeepSets / mean-pool set encoder over the fan (a numpy stand-in for
    their cross-attention set encoder; no PyTorch).
  * PPO clipped surrogate, entropy bonus, and KL early-stop (their §4.3).
  * Asymmetric critic that sees realized ``Y`` at train time only (their
    privileged future-price critic).
  * Leaf masses from the assigned fan (their Eq. 5), which become the IS
    proposal ``q``. Empty leaves keep a tiny floor rather than being
    dropped from a categorical support that must stay K- or 12-dimensional.

Not used (does not map):
  * Receding-horizon battery LP / Rockafellar–Uryasev CVaR program. There
    is no inner solver; the constructed leaf distribution *is* the IS
    proposal, and the analog of closed-loop profit is ``tail_reward``.
  * Multi-head attention, Set Transformer, or a fan of length-H price
    trajectories. A scenario here is one epistemic leaf + one aleatory
    draw, not a path of electricity prices.
  * Forward/backward Wasserstein reduction baselines (those are their
    *competitors*; CEM-IS is the closest analog already in this repo).

Sign / privilege: **never** given ``P(Y>v|leaf)``. Actor tokens carry the
leaf's ``(μ, σ)`` (the analog of their forecast trajectory ``ξ``) but not
the realized ``Y``; the critic sees ``Y``. Aggregation multiplies Eq. 5's
prior mass by ``(y − v)_+`` so that an identity assignment recovers
categorical CE toward ``q_star`` — without that kernel, occupancy of a
fan drawn from ``q`` just copies ``q`` and the proposal never moves.
"""

from __future__ import annotations

import math

import numpy as np

from cvar_psha.env import LogicTreeEnv
from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward, tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.tree_env import TreeLogicEnv

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))


def _gelu_prime(x: np.ndarray) -> np.ndarray:
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    th = np.tanh(inner)
    sech2 = 1.0 - th * th
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + th) + 0.5 * x * sech2 * d_inner


def _sample_from_mix(
    rng: np.random.Generator,
    q: np.ndarray,
    prior: np.ndarray,
    n: int,
    defensive_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` leaf indices from ``(1-ε)q + ε p``; return indices and IW."""
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    prior = np.asarray(prior, dtype=float)
    prior = prior / prior.sum()
    use_prior = rng.random(n) < defensive_eps
    idx = np.empty(n, dtype=int)
    n_prior = int(use_prior.sum())
    if n_prior:
        idx[use_prior] = rng.choice(len(prior), size=n_prior, p=prior)
    n_prop = n - n_prior
    if n_prop:
        idx[~use_prior] = rng.choice(len(q), size=n_prop, p=q)
    q_mix = (1.0 - defensive_eps) * q + defensive_eps * prior
    iw = prior[idx] / np.clip(q_mix[idx], 1e-12, None)
    return idx, iw


def _actor_tokens(
    native: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    prior_p: np.ndarray,
    assigned: np.ndarray,
    current_mask: np.ndarray,
    n_leaves: int,
) -> np.ndarray:
    """Paper Eq. 12 analog: forecast features + assignment bookkeeping, no Y."""
    s = native.shape[0]
    nat = np.zeros((s, n_leaves), dtype=float)
    nat[np.arange(s), native] = 1.0
    asg = np.zeros((s, n_leaves + 1), dtype=float)
    unassigned = assigned < 0
    asg[unassigned, n_leaves] = 1.0
    done = ~unassigned
    if done.any():
        asg[done, assigned[done]] = 1.0
    fan_mu = np.full((s, 1), float(mu.mean()))
    fan_sig = np.full((s, 1), float(sigma.mean()))
    return np.concatenate(
        [
            nat,
            (mu / 2.0).reshape(-1, 1),
            sigma.reshape(-1, 1),
            prior_p.reshape(-1, 1),
            asg,
            current_mask.reshape(-1, 1).astype(float),
            fan_mu,
            fan_sig,
        ],
        axis=1,
    )


def _critic_tokens(actor_x: np.ndarray, y: np.ndarray, v95: float) -> np.ndarray:
    """Asymmetric critic: actor tokens plus privileged realized Y."""
    y = np.asarray(y, dtype=float)
    extra = np.column_stack(
        [
            np.tanh(np.log1p(np.clip(y, 0.0, None))),
            (y > v95).astype(float),
        ]
    )
    return np.concatenate([actor_x, extra], axis=1)


def _token_dim(n_leaves: int, critic: bool) -> int:
    # native L + mu + sigma + prior + assigned (L+1) + mask + fan_mu + fan_sig
    d = n_leaves + 2 + 1 + (n_leaves + 1) + 1 + 2
    return d + 2 if critic else d


class _AssignmentNet:
    """One-hidden-layer DeepSets readout with a linear privileged critic."""

    def __init__(self, n_leaves: int, hidden: int, rng: np.random.Generator):
        self.n_leaves = n_leaves
        self.hidden = hidden
        d_a = _token_dim(n_leaves, critic=False)
        d_c = _token_dim(n_leaves, critic=True)
        scale = 0.4
        self.W1 = rng.normal(0.0, scale / math.sqrt(d_a), (d_a, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, scale / math.sqrt(hidden), (hidden, n_leaves))
        self.b2 = np.zeros(n_leaves)
        self.Wc = rng.normal(0.0, scale / math.sqrt(d_c), d_c)
        self.bc = 0.0

    def logits(self, x: np.ndarray, native: np.ndarray, identity_bias: float) -> np.ndarray:
        pre = x @ self.W1 + self.b1
        z = _gelu(pre)
        out = z @ self.W2 + self.b2
        if identity_bias != 0.0:
            out = out.copy()
            out[np.arange(x.shape[0]), native] += identity_bias
        return out, pre, z

    def value(self, x_priv: np.ndarray) -> float:
        return float(x_priv.mean(axis=0) @ self.Wc + self.bc)

    def ppo_step(
        self,
        x: np.ndarray,
        native: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        advantage: np.ndarray,
        x_priv: np.ndarray,
        ret: float,
        identity_bias: float,
        lr: float,
        clip: float,
        entropy_coef: float,
        critic_coef: float = 0.5,
    ) -> tuple[float, float]:
        """One PPO epoch on a single group. Returns (approx KL, entropy)."""
        logits, pre, z = self.logits(x, native, identity_bias)
        pi = _softmax_rows(logits)
        logp = np.log(np.clip(pi[np.arange(pi.shape[0]), actions], 1e-12, None))
        ratio = np.exp(np.clip(logp - old_logp, -20.0, 20.0))
        clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip)
        unclipped_term = ratio * advantage
        clipped_term = clipped * advantage
        # Gradient of min(unclipped, clipped): unclipped path iff it is smaller.
        use_unclipped = unclipped_term <= clipped_term
        dlogp = -pi
        dlogp[np.arange(pi.shape[0]), actions] += 1.0
        scale = np.where(use_unclipped, ratio * advantage, 0.0)
        # Ascent on the clipped surrogate (same sign as REINFORCE in this repo).
        dlogits = dlogp * scale[:, None]

        log_pi = np.log(np.clip(pi, 1e-12, None))
        row_h = -np.sum(pi * log_pi, axis=1, keepdims=True)
        # ∂H/∂z_i = -π_i (log π_i + H); add entropy bonus to the ascent.
        dent = -pi * (log_pi + row_h)
        dlogits = dlogits + entropy_coef * dent
        dlogits /= max(x.shape[0], 1)

        dW2 = z.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dz = dlogits @ self.W2.T
        dpre = dz * _gelu_prime(pre)
        dW1 = x.T @ dpre
        db1 = dpre.sum(axis=0)

        self.W2 += lr * dW2
        self.b2 += lr * db2
        self.W1 += lr * dW1
        self.b1 += lr * db1

        v = self.value(x_priv)
        resid = np.clip(v - ret, -2.0, 2.0)
        mean_priv = x_priv.mean(axis=0)
        self.Wc -= lr * critic_coef * resid * mean_priv
        self.bc -= lr * critic_coef * resid

        ent = float(row_h.mean())

        kl = float(np.mean(old_logp - logp))
        return kl, float(ent)


def _assign_fan(
    net: _AssignmentNet,
    native: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    prior_p: np.ndarray,
    y: np.ndarray,
    v95: float,
    group_size: int,
    identity_bias: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Sequential group assignment. Returns leaf index per scenario + PPO buffer."""
    s = native.shape[0]
    n_leaves = net.n_leaves
    assigned = np.full(s, -1, dtype=int)
    # Atypical first (paper §4.1): deviation from the mean *forecast*,
    # here |μ − mean(μ)|. Sorting on realized Y would leak the critic's
    # privileged information into the actor's group order.
    order = np.argsort(np.abs(mu - float(mu.mean())))[::-1]
    groups: list[np.ndarray] = []
    for start in range(0, s, group_size):
        groups.append(order[start : start + group_size])

    buffer: list[dict] = []
    for g_idx in groups:
        mask = np.zeros(s, dtype=bool)
        mask[g_idx] = True
        x_all = _actor_tokens(native, mu, sigma, prior_p, assigned, mask, n_leaves)
        x = x_all[g_idx]
        nat_g = native[g_idx]
        logits, _, _ = net.logits(x, nat_g, identity_bias)
        pi = _softmax_rows(logits)
        actions = np.array(
            [int(rng.choice(n_leaves, p=pi[i])) for i in range(pi.shape[0])],
            dtype=int,
        )
        logp = np.log(np.clip(pi[np.arange(pi.shape[0]), actions], 1e-12, None))
        x_priv = _critic_tokens(x_all, y, v95)
        v = net.value(x_priv)
        assigned[g_idx] = actions
        buffer.append(
            {
                "x": x.copy(),
                "native": nat_g.copy(),
                "actions": actions.copy(),
                "old_logp": logp.copy(),
                "x_priv": x_priv.copy(),
                "value": v,
                "g_idx": np.asarray(g_idx, dtype=int).copy(),
            }
        )
    return assigned, buffer


def _leaf_masses(
    assigned: np.ndarray,
    iw: np.ndarray,
    y: np.ndarray,
    v95: float,
    n_leaves: int,
) -> np.ndarray | None:
    """Paper Eq. 5 with a problem-driven tail kernel.

    ``q_hat[ℓ] ∝ Σ_{i: a_i=ℓ} (p/q)_i y_i 1{y_i>v}``. Identity assignment
    is categorical CE toward ``q_star``. Returns None when the fan has
    fewer than 3 exceedances, so a single lucky body-arm tail hit cannot
    lock ``q``.
    """
    tail = np.asarray(y, dtype=float) > v95
    if int(tail.sum()) < 3:
        return None
    w = np.asarray(iw, dtype=float) * np.asarray(y, dtype=float) * tail.astype(float)
    mass = np.zeros(n_leaves, dtype=float)
    np.add.at(mass, assigned, w)
    s = float(mass.sum())
    if s <= 0:
        return None
    return mass / s


def _run_categorical_sto(
    *,
    rng: np.random.Generator,
    n_leaves: int,
    prior: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    sample_y,
    v95: float,
    budget: int,
    fan_size: int,
    group_size: int,
    smoothing: float,
    defensive_eps: float,
    identity_bias: float,
    hidden: int,
    ppo_lr: float,
    ppo_clip: float,
    ppo_epochs: int,
    entropy_coef: float,
    kl_stop: float,
    eval_every: int,
    name: str,
    reward_fn,
) -> MethodResult:
    if identity_bias <= 0.0:
        # Keep ~99% probability on the native leaf at init so q_hat is a
        # CE update. A 90% identity rate still let PPO scramble the 12-path
        # tree onto a mid-μ pessimistic GMM.
        identity_bias = math.log(max(n_leaves - 1, 1) * 99.0)

    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = prior.copy()
    q = q / q.sum()
    net = _AssignmentNet(n_leaves, hidden, rng)
    history = [q.copy()]
    n_done = 0
    n_effective = []

    while n_done < budget:
        m = min(fan_size, budget - n_done)
        idx, iw = _sample_from_mix(rng, q, prior, m, defensive_eps)
        ys = np.array([sample_y(int(i)) for i in idx], dtype=float)
        for y_i, w_i in zip(ys, iw):
            tracker.update(float(y_i), float(w_i))
        n_done += m

        mu_s = mus[idx]
        sig_s = sigmas[idx]
        prior_s = prior[idx]
        assigned, buf = _assign_fan(
            net, idx, mu_s, sig_s, prior_s, ys, v95, group_size, identity_bias, rng
        )
        q_hat = _leaf_masses(assigned, iw, ys, v95, n_leaves)
        if q_hat is not None:
            q_hat = (1.0 - defensive_eps) * q_hat + defensive_eps * prior
            q_hat = np.clip(q_hat, 1e-8, None)
            q_hat = q_hat / q_hat.sum()
            q = smoothing * q_hat + (1.0 - smoothing) * q
            q = np.clip(q, 1e-8, None)
            q = q / q.sum()
            n_effective.append(int((q_hat > (1.0 / max(n_leaves, 1) * 0.25)).sum()))
        history.append(q.copy())

        rewards = np.array([reward_fn(float(y_i), float(w_i), v95) for y_i, w_i in zip(ys, iw)])
        r_max = max(float(np.max(np.abs(rewards))), 1e-8)
        r_scaled = rewards / r_max
        g_ret = float(r_scaled.mean())
        # Per-scenario advantage: only the assignments that produced a tail
        # contribution are reinforced. A shared batch scalar was collapsing
        # the 12-path tree onto whichever leaf got lucky co-occurrence.
        if r_scaled.max() > 0:
            for _ in range(ppo_epochs):
                kls = []
                for b in buf:
                    a = r_scaled[b["g_idx"]] - float(b["value"])
                    kl, _ent = net.ppo_step(
                        b["x"],
                        b["native"],
                        b["actions"],
                        b["old_logp"],
                        a,
                        b["x_priv"],
                        g_ret,
                        identity_bias,
                        ppo_lr,
                        ppo_clip,
                        entropy_coef,
                    )
                    kls.append(kl)
                if float(np.mean(kls)) > kl_stop:
                    break

    extras = {
        "q_history": np.asarray(history),
        "mean_effective_leaves": float(np.mean(n_effective)) if n_effective else float(n_leaves),
    }
    return MethodResult(name=name, metrics=tracker.finalize(), final_q=q, extras=extras)


def run_sto_assign(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    fan_size: int = 128,
    group_size: int = 4,
    smoothing: float = 0.35,
    defensive_eps: float = 0.1,
    identity_bias: float = 0.0,
    hidden: int = 32,
    ppo_lr: float = 0.03,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 4,
    entropy_coef: float = 0.02,
    kl_stop: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    """1-node CO-STC: assign a fan of GMM-arm samples onto K leaves."""
    return _run_categorical_sto(
        rng=env.rng,
        n_leaves=env.n_arms,
        prior=env.weights.copy(),
        mus=env.mus.copy(),
        sigmas=env.sigmas.copy(),
        sample_y=env.sample_ground_motion,
        v95=v95,
        budget=budget,
        fan_size=fan_size,
        group_size=group_size,
        smoothing=smoothing,
        defensive_eps=defensive_eps,
        identity_bias=identity_bias,
        hidden=hidden,
        ppo_lr=ppo_lr,
        ppo_clip=ppo_clip,
        ppo_epochs=ppo_epochs,
        entropy_coef=entropy_coef,
        kl_stop=kl_stop,
        eval_every=eval_every,
        name="CO-STC",
        reward_fn=tail_reward,
    )


def run_tree_sto_assign(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    fan_size: int = 128,
    group_size: int = 4,
    smoothing: float = 0.35,
    defensive_eps: float = 0.1,
    identity_bias: float = 0.0,
    hidden: int = 32,
    ppo_lr: float = 0.03,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 4,
    entropy_coef: float = 0.02,
    kl_stop: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    """3-node CO-STC: assign a fan of complete paths onto the 12 tree leaves."""
    mus = np.array([env.leaf_params(p)[0] for p in env.paths], dtype=float)
    sigmas = np.array([env.leaf_params(p)[1] for p in env.paths], dtype=float)

    def sample_y(i: int) -> float:
        return env.sample_y(env.paths[int(i)])

    return _run_categorical_sto(
        rng=env.rng,
        n_leaves=env.n_paths,
        prior=env.path_priors.copy(),
        mus=mus,
        sigmas=sigmas,
        sample_y=sample_y,
        v95=v95,
        budget=budget,
        fan_size=fan_size,
        group_size=group_size,
        smoothing=smoothing,
        defensive_eps=defensive_eps,
        identity_bias=identity_bias,
        hidden=hidden,
        ppo_lr=ppo_lr,
        ppo_clip=ppo_clip,
        ppo_epochs=ppo_epochs,
        entropy_coef=entropy_coef,
        kl_stop=kl_stop,
        eval_every=eval_every,
        name="CO-STC",
        reward_fn=path_tail_reward,
    )
