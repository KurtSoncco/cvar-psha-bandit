"""Build one verdict-panel figure per experiment setting from summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verdict_panels"
PAPER_KS_HI = 0.113


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ks_bar(ax, labels, q_disagg, q_star, title: str, winner: str) -> None:
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, q_disagg, w, label="KS → q_disagg", color="#4C78A8")
    ax.bar(x + w / 2, q_star, w, label="KS → q_star", color="#9E9E9E")
    ax.axhline(PAPER_KS_HI, color="#F58518", ls="--", lw=1.2, label=f"paper KS hi ({PAPER_KS_HI})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("KS distance")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    for i, lab in enumerate(labels):
        if lab == winner:
            ax.get_xticklabels()[i].set_fontweight("bold")


def panel_1node(summary: dict, out: Path) -> None:
    dist = summary["distance_to_targets"]
    order = ["Naive MC", "CEM-IS", "REINFORCE", "Exp3", "Disagg-IS oracle", "q* oracle"]
    qd = [dist[m]["q_disagg"]["ks"] for m in order]
    qs = [dist[m]["q_star"]["ks"] for m in order]
    short = ["Naive MC", "CEM-IS", "REINFORCE", "Exp3", "Disagg-IS", "q* oracle"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    _ks_bar(axes[0], short, qd, qs, "1-node: KS vs targets (verdict = Exp3 succeeds)", "Exp3")

    arms = ["optimistic", "average", "pessimistic"]
    pol = summary["final_policies"]
    series = {
        "q_disagg": summary["q_disagg"],
        "Exp3": pol["Exp3"],
        "REINFORCE": pol["REINFORCE"],
        "CEM-IS": pol["CEM-IS"],
        "Naive MC": pol["Naive MC"],
    }
    x = np.arange(len(arms))
    w = 0.15
    for i, (name, vals) in enumerate(series.items()):
        axes[1].bar(x + (i - 2) * w, vals, w, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(arms)
    axes[1].set_ylabel("policy mass")
    axes[1].set_title("Final arm mass (collapse check)")
    axes[1].legend(fontsize=7)

    fig.suptitle("Verdict panel — 1-node GMM: SUCCEEDS", fontsize=13, fontweight="bold")
    fig.savefig(out / "1node_verdict.png", dpi=160)
    plt.close(fig)


def panel_3node(summary: dict, out: Path) -> None:
    dist = summary["distance_to_targets"]
    order = [
        "Naive MC",
        "CEM-IS",
        "CVaR-CPO",
        "Flat Exp3",
        "Flat REINFORCE",
        "Hierarchical",
        "Disagg-IS oracle",
        "q* oracle",
    ]
    qd = [dist[m]["q_disagg"]["ks"] for m in order]
    qs = [dist[m]["q_star"]["ks"] for m in order]
    short = [
        "Naive MC",
        "CEM-IS",
        "CVaR-CPO",
        "Flat Exp3",
        "Flat REINFORCE",
        "Hierarchical",
        "Disagg-IS",
        "q* oracle",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    _ks_bar(
        axes[0],
        short,
        qd,
        qs,
        "3-node: KS vs targets (verdict = Hierarchical succeeds)",
        "Hierarchical",
    )

    # Four heaviest q_disagg paths
    q_disagg = np.asarray(summary["q_disagg"])
    labels = summary["path_labels"]
    top_idx = np.argsort(q_disagg)[::-1][:4]
    top_labs = [labels[i].replace("/", "\n") for i in top_idx]
    pol = summary["final_policies"]
    series = {
        "q_disagg": q_disagg[top_idx],
        "Hierarchical": np.asarray(pol["Hierarchical"])[top_idx],
        "CEM-IS": np.asarray(pol["CEM-IS"])[top_idx],
        "Naive MC": np.asarray(pol["Naive MC"])[top_idx],
    }
    x = np.arange(len(top_idx))
    w = 0.18
    for i, (name, vals) in enumerate(series.items()):
        axes[1].bar(x + (i - 1.5) * w, vals, w, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top_labs, fontsize=8)
    axes[1].set_ylabel("path mass")
    axes[1].set_title("Top-4 q_disagg paths (CEM collapse)")
    axes[1].legend(fontsize=7)

    fig.suptitle("Verdict panel — 3-node: SUCCEEDS", fontsize=13, fontweight="bold")
    fig.savefig(out / "3node_verdict.png", dpi=160)
    plt.close(fig)


def panel_continuous(summary: dict, out: Path) -> None:
    dist = summary["distance_to_targets"]
    order = ["Naive MC", "Hierarchical JEPA-CVaR", "G-PMC AIS", "Disagg-IS oracle"]
    qd = [dist[m]["q_disagg"]["ks"] for m in order]
    qs = [dist[m]["q_star"]["ks"] for m in order]
    short = ["Naive MC", "JEPA-CVaR", "G-PMC AIS", "Disagg-IS"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    _ks_bar(
        axes[0],
        short,
        qd,
        qs,
        "Continuous: KS vs targets (verdict = partial)",
        "JEPA-CVaR",
    )

    means = summary["final_theta_mean"]
    cats = ["θμ", "θσ"]
    series = {
        "q_disagg mean": summary["mean_theta_q_disagg"],
        "q_star mean": summary["mean_theta_q_star"],
        "JEPA-CVaR": means["Hierarchical JEPA-CVaR"],
        "G-PMC AIS": means["G-PMC AIS"],
        "Naive MC": means["Naive MC"],
    }
    x = np.arange(2)
    w = 0.15
    for i, (name, vals) in enumerate(series.items()):
        axes[1].bar(x + (i - 2) * w, vals, w, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].set_ylabel("mean")
    axes[1].set_title("Proposal mean vs target means")
    axes[1].legend(fontsize=7)

    fig.suptitle(
        "Verdict panel — Continuous θ: PARTIAL (JEPA in band, trails G-PMC)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out / "continuous_verdict.png", dpi=160)
    plt.close(fig)


def panel_spatial(summary: dict, out: Path) -> None:
    dist = summary["distance_to_qstar"]
    order = ["Naive MC", "Flat REINFORCE", "CVaR-CPO", "Hierarchical"]
    mass = [100 * dist[m]["mass_on_top20_qstar"] for m in order]
    kl = [dist[m]["kl"] for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].bar(order, mass, color="#E45756")
    axes[0].set_ylabel("mass on top-20 q* (%)")
    axes[0].set_title("Spatial: top-20 q* coverage (verdict = fails)")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(mass):
        axes[0].text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=8)

    axes[1].bar(order, kl, color="#F58518")
    axes[1].set_ylabel("KL → q_star")
    axes[1].set_title("KL still near prior (~16)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Verdict panel — Spatial: FAILS (policies miss risky mag/rupture modes)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out / "spatial_verdict.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    one_node = ROOT / "results" / "1node" / "summary.json"
    if not one_node.exists():
        one_node = ROOT / "results" / "summary.json"
    panel_1node(_load(one_node), OUT)
    panel_3node(_load(ROOT / "results" / "3node" / "summary.json"), OUT)
    panel_continuous(_load(ROOT / "results" / "continuous" / "summary.json"), OUT)
    panel_spatial(_load(ROOT / "results" / "spatial" / "summary.json"), OUT)
    print(f"Wrote verdict panels to {OUT}")


if __name__ == "__main__":
    main()
