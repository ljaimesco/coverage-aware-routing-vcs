"""Figures for Section VII from results/*_<stamp>_*.

Usage (from the repo root):  python analysis/paper_figures.py <results_dir> <out_dir> [stamp]
Writes fig_ablation.pdf, fig_travel.pdf, fig_certificate.pdf (and .png previews).
Palette: blue #2a78d6 (LDD), orange #eb6834 (uniform), aqua #1baf7a, yellow #eda100 (validated).
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#333333", "#666666", "#dddddd"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def load(res, stamp, key, kind="results"):
    return pd.read_csv(glob.glob(os.path.join(res, f"{key}_{stamp}_*", f"{kind}.csv"))[0])


def mean_ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


SERIES = [("ldd", "greedy_ls", BLUE, "o", "-", True, "LDD + coverage-aware"),
          ("ldd", "closest", BLUE, "o", "--", False, "LDD + closest-length"),
          ("uniform", "greedy_ls", ORANGE, "s", "-", True, "Uniform + coverage-aware"),
          ("uniform", "closest", ORANGE, "s", "--", False, "Uniform + closest-length")]


def fig_ablation(res, out, stamp):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=True)
    panels = [("main_london_equal", "(a) London, equal costs"), ("main_london_hetero", "(b) London, heterogeneous costs"),
              ("main_monaco_equal", "(c) Monaco, equal costs"), ("main_monaco_hetero", "(d) Monaco, heterogeneous costs")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        r = load(res, stamp, key)
        Rs = sorted(r.R.unique())
        for dev, sel, col, mk, ls, filled, lab in SERIES:
            m, h = zip(*[mean_ci(r[(r.R == R) & (r.dev_rule == dev) & (r.sel_rule == sel)].J_reduction_pct) for R in Rs])
            ax.errorbar(np.array(Rs) / 1000, m, yerr=h, color=col, ls=ls, lw=2, marker=mk, ms=6, capsize=2.5,
                        mfc=col if filled else "white", mec=col, mew=1.5, label=lab, zorder=3)
        ax.set_title(title, fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks([5, 15, 25, 35])
        ax.set_ylim(0, 95)
    for ax in axes[1]:
        ax.set_xlabel("Budget R (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reduction in J (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "fig_ablation")


def fig_travel(res, out, stamp):
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, [("main_london_hetero", "(a) London"), ("main_monaco_hetero", "(b) Monaco")]):
        r = load(res, stamp, key)
        r = r[r.sel_rule == "greedy_ls"]
        for dev, col, mk, lab in [("ldd", BLUE, "o", "LDD"), ("uniform", ORANGE, "s", "Uniform")]:
            g = r[r.dev_rule == dev].groupby("R")[["added_travel_pct", "J_reduction_pct"]].mean()
            ax.plot(g.added_travel_pct, g.J_reduction_pct, color=col, lw=2, marker=mk, ms=6, label=lab, zorder=3)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylabel("Reduction in J (%)")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 95)
    axes[1].set_xlabel("Added travel (% of shortest routes)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig_travel")


def fig_certificate(res, out, stamp):
    """London, LDD with greedy + local search, runs without unrealized vehicles."""
    rows = [("Largest best-response gain", "max_br_gain_active", BLUE, "o"), ("Prop. 3 bound", "eps2_max_active", ORANGE, "s"),
            ("Thm. 2 bound", "eps_max_active", AQUA, "D"), ("Mean utility, active", "util_active", YELLOW, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, [("main_london_equal", "(a) Equal costs"), ("main_london_hetero", "(b) Heterogeneous costs")]):
        r = load(res, stamp, key)
        v = load(res, stamp, key, "vehicles")
        r = r[(r.dev_rule == "ldd") & (r.sel_rule == "greedy_ls") & (r.n_unrealized == 0)].copy()
        va = v[(v.dev_rule == "ldd") & (v.sel_rule == "greedy_ls") & v.active_stage1]
        ua = va.groupby(["scenario", "R"]).utility.mean().rename("util_active")
        r = r.join(ua, on=["scenario", "R"])
        for y, (lab, col, c, mk) in enumerate(rows):
            x = r[col].dropna()
            q1, md, q3 = np.percentile(x, [25, 50, 75])
            ax.plot([q1, q3], [y, y], color=c, lw=4, solid_capstyle="round", zorder=2)
            ax.plot(md, y, marker=mk, color=c, ms=8, mec="white", mew=1, zorder=3)
            ax.annotate(f"{md:,.2f}" if md < 10 else f"{md:,.1f}" if md < 100 else f"{md:,.0f}", (q3, y),
                        textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 2e4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r_[0] for r_ in rows])
        if ax is axes[0]:                       # shared y axis: set (inverted) limits once
            ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_title(f"{title}  (n = {len(r)} runs)", fontsize=9, color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Utility units (log scale)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig_certificate")


if __name__ == "__main__":
    res, out = sys.argv[1], sys.argv[2]
    stamp = sys.argv[3] if len(sys.argv) > 3 else "20261104"
    os.makedirs(out, exist_ok=True)
    fig_ablation(res, out, stamp)
    fig_travel(res, out, stamp)
    fig_certificate(res, out, stamp)
    print("figures in", out)
