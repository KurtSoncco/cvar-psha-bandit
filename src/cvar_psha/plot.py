"""Plot learning curves: CVaR, variance, and ESS vs computational budget."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvar_psha.core.result import MethodResult


def _aggregate_replications(
    results: list[MethodResult],
) -> dict[str, np.ndarray]:
    """Mean/std over replications that share the same budget grid."""
    budgets = results[0].metrics["budget"]
    keys = ("cvar", "variance", "ess")
    out: dict[str, np.ndarray] = {"budget": budgets}
    for key in keys:
        stacked = np.vstack([r.metrics[key] for r in results])
        out[f"{key}_mean"] = np.nanmean(stacked, axis=0)
        out[f"{key}_std"] = np.nanstd(stacked, axis=0)
    return out


def plot_analysis_comparisons(
    method_runs: dict[str, list[MethodResult]],
    true_cvar: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the two primary analysis figures: CVaR convergence and ESS vs N."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = {
        "Naive MC": "#7a7a7a",
        "q* oracle": "#1f1f1f",
        "Disagg-IS oracle": "#555555",
        "CEM-IS": "#c4a35a",
        "Flat Exp3": "#6c8ebf",
        "Flat REINFORCE": "#8eb0d9",
        "Exp3": "#6c8ebf",
        "REINFORCE": "#8eb0d9",
        "Hierarchical": "#c0392b",
        "CVaR-CPO": "#1a7a4c",
        "G-PMC AIS": "#8e44ad",
        "G-PMC AIS (CVaR)": "#8e44ad",
        "Hierarchical JEPA-CVaR": "#e08283",
        "Hierarchical JEPA-CVaR v2": "#c0392b",
        "CVaR-BF AIS": "#0e7c7b",
        "QR-SRM AIS": "#e67e22",
        "CO-STC": "#2c6fbb",
    }

    # Emphasize Hierarchical vs baselines.
    highlight = {
        "Hierarchical",
        "q* oracle",
        "Naive MC",
        "Hierarchical JEPA-CVaR v2",
        "G-PMC AIS",
        "CVaR-BF AIS",
        "QR-SRM AIS",
        "CO-STC",
    }

    # --- Plot 1: CVaR convergence ---
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for name, runs in method_runs.items():
        agg = _aggregate_replications(runs)
        x = agg["budget"]
        mean = agg["cvar_mean"]
        std = agg["cvar_std"]
        color = palette.get(name, None)
        lw = 2.6 if name in highlight else 1.6
        alpha_line = 1.0 if name in highlight else 0.85
        ax1.plot(x, mean, label=name, linewidth=lw, color=color, alpha=alpha_line)
        ax1.fill_between(x, mean - std, mean + std, alpha=0.12 if name in highlight else 0.06, color=color)
    ax1.axhline(true_cvar, color="black", linestyle="--", linewidth=1.8, label="True CVaR")
    ax1.set_title("CVaR convergence vs computational budget")
    ax1.set_xlabel("Leaf samples N")
    ax1.set_ylabel("CVaR estimate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, framealpha=0.95)
    cvar_path = output_dir / "cvar_convergence.png"
    fig1.savefig(cvar_path, dpi=160)
    plt.close(fig1)

    # --- Plot 2: ESS (policy collapse diagnostic) ---
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for name, runs in method_runs.items():
        agg = _aggregate_replications(runs)
        x = agg["budget"]
        mean = agg["ess_mean"]
        std = agg["ess_std"]
        color = palette.get(name, None)
        lw = 2.6 if name in highlight else 1.6
        alpha_line = 1.0 if name in highlight else 0.85
        ax2.plot(x, mean, label=name, linewidth=lw, color=color, alpha=alpha_line)
        ax2.fill_between(x, mean - std, mean + std, alpha=0.12 if name in highlight else 0.06, color=color)
    ax2.set_title("Effective sample size vs computational budget")
    ax2.set_xlabel("Leaf samples N")
    ax2.set_ylabel("ESS")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, framealpha=0.95)
    ess_path = output_dir / "ess_comparison.png"
    fig2.savefig(ess_path, dpi=160)
    plt.close(fig2)

    return cvar_path, ess_path


def plot_mag_rupture_heatmap(
    env,
    method_qs: dict[str, np.ndarray],
    q_star: np.ndarray,
    output_dir: Path,
) -> Path:
    """Marginal mag × rupture mass heatmaps for selected methods vs q*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_m = len(env.mags)
    n_r = env.n_rupture_bins

    def marginal(q: np.ndarray) -> np.ndarray:
        heat = np.zeros((n_m, n_r), dtype=float)
        for idx, path in enumerate(env.paths):
            _g, _gmm, m, r = path
            heat[m, r] += q[idx]
        s = heat.sum()
        return heat / s if s > 0 else heat

    panels = [("q*", q_star)] + list(method_qs.items())
    # Cap panels for readability
    panels = panels[:5]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, q) in zip(axes, panels):
        heat = marginal(q)
        im = ax.imshow(heat, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Rupture bin")
        ax.set_ylabel("Magnitude bin")
        yt = np.linspace(0, n_m - 1, min(6, n_m)).astype(int)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{env.mags[i]:.1f}" for i in yt], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = output_dir / "fault_mag_mass.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_learning_curves(
    method_runs: dict[str, list[MethodResult]],
    true_cvar: float,
    output_dir: Path,
    q_star: np.ndarray | None = None,
    path_labels: list[str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    titles = ("CVaR Convergence", "Estimator Variance", "Effective Sample Size")
    ylabels = ("CVaR estimate", "Rolling variance", "ESS")
    metric_keys = ("cvar", "variance", "ess")

    for name, runs in method_runs.items():
        agg = _aggregate_replications(runs)
        x = agg["budget"]
        for ax, key, title, ylabel in zip(axes, metric_keys, titles, ylabels):
            mean = agg[f"{key}_mean"]
            std = agg[f"{key}_std"]
            ax.plot(x, mean, label=name, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            ax.set_title(title)
            ax.set_xlabel("Computational budget N")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0].axhline(true_cvar, color="black", linestyle="--", linewidth=1.5, label="True CVaR")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[2].legend(fontsize=8)

    fig_path = output_dir / "learning_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Final policy comparison if available.
    final_qs = {}
    for name, runs in method_runs.items():
        qs = [r.final_q for r in runs if r.final_q is not None]
        if qs:
            final_qs[name] = np.mean(np.vstack(qs), axis=0)

    if final_qs:
        plot_final_policies(
            final_qs,
            output_dir,
            q_star=q_star,
            path_labels=path_labels,
        )

    return fig_path


def plot_final_policies(
    final_qs: dict[str, np.ndarray],
    output_dir: Path,
    q_star: np.ndarray | None = None,
    path_labels: list[str] | None = None,
    top_n: int = 15,
) -> Path:
    """Bar chart of final path / arm probabilities (top-N when the space is large)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_arms = len(next(iter(final_qs.values())))
    score = np.zeros(n_arms, dtype=float)
    if q_star is not None:
        score = np.maximum(score, np.asarray(q_star, dtype=float))
    for q in final_qs.values():
        score = np.maximum(score, np.asarray(q, dtype=float))

    if n_arms > top_n:
        idx = np.argsort(score)[::-1][:top_n]
        idx = np.sort(idx)  # keep a stable left-to-right order among the top set
        # Prefer ranking order for readability
        idx = np.argsort(score)[::-1][:top_n]
    else:
        idx = np.arange(n_arms)

    n_show = len(idx)
    series = {}
    if q_star is not None:
        series["q*"] = np.asarray(q_star, dtype=float)[idx]
    for name, q in final_qs.items():
        series[name] = np.asarray(q, dtype=float)[idx]

    if path_labels is not None and len(path_labels) == n_arms:
        labels = [path_labels[i] for i in idx]
    elif path_labels is not None and len(path_labels) == n_show:
        labels = list(path_labels)
    else:
        labels = [f"Path {i}" for i in idx]

    fig_w = max(8.0, 0.55 * n_show + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), constrained_layout=True)
    x = np.arange(n_show, dtype=float)
    width = 0.8 / max(len(series), 1)
    offset = 0.0
    ymax = 0.0
    for name, vals in series.items():
        ax.bar(x + offset, vals, width, label=name)
        ymax = max(ymax, float(np.max(vals)) if vals.size else 0.0)
        offset += width

    ax.set_xticks(x + width * (len(series) - 1) / 2.0)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Probability")
    title = "Final / reference sampling distributions"
    if n_arms > top_n:
        title += f" (top {top_n} of {n_arms} paths)"
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    policy_path = output_dir / "final_policies.png"
    fig.savefig(policy_path, dpi=150)
    plt.close(fig)
    return policy_path


def plot_hazard_curves(
    pga_grid: np.ndarray,
    hazard_curves: dict[str, np.ndarray],
    output_dir: Path,
    v_threshold: float | None = None,
) -> Path:
    """Log-log mean and fractile hazard curves (annual exceedance rate vs PGA)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    styles = {
        "mean": ("-", "#1f1f1f", 2.2, "Mean hazard"),
        "p16": ("--", "#6c8ebf", 1.6, "16th fractile"),
        "p50": ("--", "#7a7a7a", 1.6, "50th fractile"),
        "p84": ("--", "#c0392b", 1.6, "84th fractile"),
    }
    for key, (ls, color, lw, label) in styles.items():
        if key not in hazard_curves:
            continue
        lam = np.clip(hazard_curves[key], 1e-12, None)
        ax.loglog(pga_grid, lam, ls=ls, color=color, lw=lw, label=label)

    if v_threshold is not None:
        ax.axvline(v_threshold, color="#555555", ls=":", lw=1.4, label=f"Threshold v={v_threshold:.3g} g")

    ax.set_xlabel("Peak ground acceleration PGA (g)")
    ax.set_ylabel("Annual exceedance rate (1/yr)")
    ax.set_title("PSHA hazard curves (Houng et al. 2025 four-parameter model)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95)
    out = output_dir / "hazard_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_continuous_theta_marginals(
    nodes: np.ndarray,
    q_disagg: np.ndarray,
    method_means: dict[str, np.ndarray],
    theta_names: tuple[str, ...],
    output_dir: Path,
) -> Path:
    """Four 1-D marginal panels: q_disagg vs each method's Gaussian marginal mean."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dim = nodes.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    axes = axes.ravel()

    palette = {
        "Naive MC": "#7a7a7a",
        "Disagg-IS oracle": "#1f1f1f",
        "G-PMC AIS": "#8e44ad",
        "Hierarchical JEPA-CVaR": "#c0392b",
    }
    labels = {
        "b": r"$b$-value",
        "m_max": r"$m_{\max}$",
        "dmu": r"$\Delta\mu$ (ln median GMM)",
        "dsigma": r"$\Delta\sigma$ (ln sigma GMM)",
    }

    for d in range(min(dim, 4)):
        ax = axes[d]
        name = theta_names[d] if d < len(theta_names) else f"theta_{d}"
        x = nodes[:, d]
        order = np.argsort(x)
        ax.fill_between(x[order], 0, q_disagg[order], alpha=0.35, color="#c4a35a", label="q_disagg")
        for mname, mean in method_means.items():
            ax.axvline(
                mean[d],
                color=palette.get(mname, "black"),
                lw=1.8,
                ls="--",
                label=mname if d == 0 else None,
            )
        ax.set_xlabel(labels.get(name, name))
        ax.set_ylabel("Disagg mass" if d == 0 else "")
        ax.grid(True, alpha=0.25)

    if method_means:
        axes[0].legend(fontsize=7, framealpha=0.95, loc="best")
    fig.suptitle("Epistemic disaggregation marginals vs learned proposal means", fontsize=11)
    out = output_dir / "continuous_theta_marginals.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_continuous_theta_comparison(
    nodes: np.ndarray,
    q_disagg: np.ndarray,
    method_means: dict[str, np.ndarray],
    output_dir: Path,
    q_star: np.ndarray | None = None,
    theta_names: tuple[str, ...] | None = None,
) -> Path:
    """Backward-compatible alias: 4-D marginals for Houng PSHA, 2-D scatter otherwise."""
    if nodes.shape[1] > 2:
        names = theta_names or ("b", "m_max", "dmu", "dsigma")
        return plot_continuous_theta_marginals(nodes, q_disagg, method_means, names, output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    sc = ax.scatter(
        nodes[:, 0], nodes[:, 1], c=q_disagg, cmap="magma", s=14, alpha=0.85, linewidths=0
    )
    fig.colorbar(sc, ax=ax, label="q_disagg mass (paper-1 proven-optimal)")

    palette = {
        "Naive MC": "#7a7a7a",
        "Disagg-IS oracle": "#1f1f1f",
        "G-PMC AIS": "#8e44ad",
        "Hierarchical JEPA-CVaR": "#e08283",
        "Hierarchical JEPA-CVaR v2": "#c0392b",
        "CVaR-BF AIS": "#0e7c7b",
        "QR-SRM AIS": "#e67e22",
    }
    markers = {
        "Naive MC": "o",
        "Disagg-IS oracle": "*",
        "G-PMC AIS": "s",
        "Hierarchical JEPA-CVaR": "^",
        "Hierarchical JEPA-CVaR v2": "D",
        "CVaR-BF AIS": "P",
        "QR-SRM AIS": "X",
    }
    for name, mean in method_means.items():
        ax.scatter(
            [mean[0]],
            [mean[1]],
            color=palette.get(name, "white"),
            marker=markers.get(name, "P"),
            s=170,
            edgecolor="white",
            linewidth=1.3,
            label=name,
            zorder=5,
        )

    ax.set_xlabel(r"$\theta_\mu$ (median-GMPE epistemic offset)")
    ax.set_ylabel(r"$\theta_\sigma$ (aleatory-sigma epistemic scale)")
    ax.set_title("Continuous epistemic space: disaggregation mass vs learned proposals")
    ax.legend(fontsize=8, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.25)
    out = output_dir / "continuous_theta_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
