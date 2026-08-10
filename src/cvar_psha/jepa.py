"""Lightweight numpy Joint-Embedding Predictive Architecture (JEPA).

A small, from-scratch (no autograd, no new dependency) analog of
self-supervised joint-embedding predictive learning (LeCun; BYOL/I-JEPA
family): rather than regressing raw noisy rollout outcomes directly, two
encoders map (a) the epistemic parameter theta ("context") and (b) an
empirical, rollout-derived outcome summary ("target") into a shared latent
space, and a predictor learns to predict one embedding from the other. The
target branch is only ever updated as an exponential moving average (EMA)
of a separately-trained online copy, with gradients stopped through the EMA
path -- the standard BYOL/I-JEPA anti-collapse mechanism -- reinforced here
with an explicit per-dimension variance regularizer (VICReg-style) since
this small numpy model has no batch norm to lean on.

Both directions are trained (context->target and target->context), so both
online encoders receive gradient signal; only the EMA copies are frozen.

Nothing here ever sees the closed-form disaggregation formulas in
`disaggregation.py` -- the target view is built purely from realized
rollout quantities (`build_target_features`), matching the "scalar reward
only" constraint used throughout this project. The resulting context
embedding z_c(theta) is what `jepa_cvar.py` conditions its hierarchical
sampling policy on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _init_linear(rng: np.random.Generator, d_in: int, d_out: int, scale: float = 0.3):
    w = rng.normal(0.0, scale / np.sqrt(d_in), size=(d_in, d_out))
    b = np.zeros(d_out)
    return w, b


@dataclass
class _Linear:
    w: np.ndarray
    b: np.ndarray

    def copy(self) -> "_Linear":
        return _Linear(self.w.copy(), self.b.copy())


def build_target_features(y: float, reward: float, iw: float) -> np.ndarray:
    """Purely empirical, rollout-derived outcome summary -- no closed-form
    hazard/disaggregation formula involved. 3-D: [scaled reward, scaled
    log ground motion, scaled log importance weight]."""
    return np.array(
        [
            np.tanh(reward),
            np.tanh(0.3 * np.log1p(max(y, 0.0))),
            np.tanh(0.3 * np.log(max(iw, 1e-8))),
        ],
        dtype=float,
    )


class LightweightJEPA:
    """theta_dim -> embed_dim context encoder, feat_dim -> embed_dim target
    encoder, linear cross-predictors, EMA target copies, trained via
    hand-derived gradients (linear + tanh only, so backprop is closed-form).
    """

    def __init__(
        self,
        theta_dim: int = 2,
        feat_dim: int = 3,
        embed_dim: int = 4,
        ema_tau: float = 0.98,
        var_weight: float = 1.0,
        var_target: float = 0.5,
        lr: float = 0.05,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.embed_dim = embed_dim
        self.ema_tau = ema_tau
        self.var_weight = var_weight
        self.var_target = var_target
        self.lr = lr

        self.ctx = _Linear(*_init_linear(rng, theta_dim, embed_dim))
        self.tgt = _Linear(*_init_linear(rng, feat_dim, embed_dim))
        self.pred_c2t = _Linear(*_init_linear(rng, embed_dim, embed_dim))
        self.pred_t2c = _Linear(*_init_linear(rng, embed_dim, embed_dim))

        self.ctx_ema = self.ctx.copy()
        self.tgt_ema = self.tgt.copy()

    # --- forward passes -------------------------------------------------
    def encode_context(self, theta: np.ndarray, use_ema: bool = False) -> np.ndarray:
        theta = np.atleast_2d(theta)
        layer = self.ctx_ema if use_ema else self.ctx
        return np.tanh(theta @ layer.w + layer.b)

    def encode_target(self, feat: np.ndarray, use_ema: bool = False) -> np.ndarray:
        feat = np.atleast_2d(feat)
        layer = self.tgt_ema if use_ema else self.tgt
        return np.tanh(feat @ layer.w + layer.b)

    # --- training ---------------------------------------------------------
    def train_step(self, theta_batch: np.ndarray, feat_batch: np.ndarray) -> dict[str, float]:
        theta_batch = np.atleast_2d(theta_batch)
        feat_batch = np.atleast_2d(feat_batch)
        n = theta_batch.shape[0]

        # Forward (online).
        pre_c = theta_batch @ self.ctx.w + self.ctx.b
        zc = np.tanh(pre_c)
        pre_t = feat_batch @ self.tgt.w + self.tgt.b
        zt = np.tanh(pre_t)

        pred_t = zc @ self.pred_c2t.w + self.pred_c2t.b  # predicts zt_ema
        pred_c = zt @ self.pred_t2c.w + self.pred_t2c.b  # predicts zc_ema

        # Forward (EMA, frozen / stop-gradient targets).
        zt_ema = self.encode_target(feat_batch, use_ema=True)
        zc_ema = self.encode_context(theta_batch, use_ema=True)

        err_t = pred_t - zt_ema  # (n, d)
        err_c = pred_c - zc_ema
        loss_pred = float(np.mean(np.sum(err_t**2, axis=1)) + np.mean(np.sum(err_c**2, axis=1)))

        # VICReg-style variance regularizer on the online embeddings, to
        # keep the online branches (and hence their EMA copies) from
        # collapsing to a constant.
        std_zc = np.sqrt(np.var(zc, axis=0) + 1e-8)
        std_zt = np.sqrt(np.var(zt, axis=0) + 1e-8)
        hinge_c = np.maximum(0.0, self.var_target - std_zc)
        hinge_t = np.maximum(0.0, self.var_target - std_zt)
        loss_var = float(np.sum(hinge_c) + np.sum(hinge_t))

        # --- gradients (hand-derived, linear+tanh only) ---
        d_pred_t = (2.0 / n) * err_t  # dL/d(pred_t)
        d_pred_c = (2.0 / n) * err_c

        g_pred_c2t_w = zc.T @ d_pred_t
        g_pred_c2t_b = d_pred_t.sum(axis=0)
        g_pred_t2c_w = zt.T @ d_pred_c
        g_pred_t2c_b = d_pred_c.sum(axis=0)

        d_zc_from_pred = d_pred_t @ self.pred_c2t.w.T  # (n,d)
        d_zt_from_pred = d_pred_c @ self.pred_t2c.w.T

        # Variance-hinge gradient: d/dzc[i,k] of -std_k when std_k < target.
        def _var_grad(z: np.ndarray, std: np.ndarray, hinge: np.ndarray) -> np.ndarray:
            active = (hinge > 0).astype(float)  # (d,)
            mean_z = z.mean(axis=0, keepdims=True)
            centered = z - mean_z
            # d(std_k)/d(z_ik) = centered_ik / (n * std_k)
            dstd = centered / (n * np.clip(std, 1e-6, None))[None, :]
            return -self.var_weight * active[None, :] * dstd

        d_zc = d_zc_from_pred + _var_grad(zc, std_zc, hinge_c)
        d_zt = d_zt_from_pred + _var_grad(zt, std_zt, hinge_t)

        d_pre_c = d_zc * (1.0 - zc**2)
        d_pre_t = d_zt * (1.0 - zt**2)

        g_ctx_w = theta_batch.T @ d_pre_c
        g_ctx_b = d_pre_c.sum(axis=0)
        g_tgt_w = feat_batch.T @ d_pre_t
        g_tgt_b = d_pre_t.sum(axis=0)

        # --- SGD update (online branches only) ---
        lr = self.lr
        self.pred_c2t.w -= lr * g_pred_c2t_w
        self.pred_c2t.b -= lr * g_pred_c2t_b
        self.pred_t2c.w -= lr * g_pred_t2c_w
        self.pred_t2c.b -= lr * g_pred_t2c_b
        self.ctx.w -= lr * g_ctx_w
        self.ctx.b -= lr * g_ctx_b
        self.tgt.w -= lr * g_tgt_w
        self.tgt.b -= lr * g_tgt_b

        # --- EMA update of target copies ---
        tau = self.ema_tau
        self.ctx_ema.w = tau * self.ctx_ema.w + (1 - tau) * self.ctx.w
        self.ctx_ema.b = tau * self.ctx_ema.b + (1 - tau) * self.ctx.b
        self.tgt_ema.w = tau * self.tgt_ema.w + (1 - tau) * self.tgt.w
        self.tgt_ema.b = tau * self.tgt_ema.b + (1 - tau) * self.tgt.b

        return {"loss_pred": loss_pred, "loss_var": loss_var}

    # --- gradient utility for downstream policy use ------------------------
    def context_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """d z_c / d theta at a single point theta (d_embed, d_theta), for
        policies that want to move theta along a latent direction."""
        theta = np.atleast_2d(theta)
        pre = theta @ self.ctx.w + self.ctx.b
        z = np.tanh(pre)
        # dz/dtheta = diag(1-z^2) @ W^T
        return ((1.0 - z**2).T) * self.ctx.w.T  # (embed_dim, theta_dim), single-row theta
