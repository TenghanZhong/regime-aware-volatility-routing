#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Conference-paper figure script for the regime-aware specialist forecasting paper.

Updated version:
- Figure 1 is simplified to a clean 5-step schematic.
- Figure 3 is split into two separate figures:
    * Figure 3a: main 5-asset comparison excluding TLT
    * Figure 3b: TLT robustness only

Expected input files in input_dir:
    - cross_asset_method_aggregate.csv
    - cross_asset_method_win_counts.csv
    - cross_asset_relative_aggregate.csv
    - cross_asset_relative_to_baselines.csv

Default input path:
    C:\Users\26876\Desktop\Models_selections\regime_overlay_cross_asset_ablation

Example
-------
python regime_conference_figures_v2.py
python regime_conference_figures_v2.py --input-dir "C:\Users\26876\Desktop\Models_selections\regime_overlay_cross_asset_ablation"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =========================================================
# Global style
# =========================================================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

DEFAULT_INPUT_DIR = r"C:\Users\26876\Desktop\Models_selections\regime_overlay_cross_asset_ablation"
DEFAULT_OUTPUT_SUBDIR = "conference_figures"

ASSET_ORDER = ["EEM", "GLD", "IWM", "QQQ", "SPY", "TLT"]
ASSET_ORDER_EX_TLT = ["EEM", "GLD", "IWM", "QQQ", "SPY"]
REGIME_ORDER = ["low", "mid", "high", "all"]

DISPLAY_NAME = {
    "har": "HAR",
    "rolling_best": "Rolling best",
    "static_best": "Static best",
    "gru": "GRU",
    "garch_t": "GARCH-t",
    "figarch": "FIGARCH",
    "xgb": "XGBoost",
    "regime_set_calm": "Set calm",
    "regime_set_stress": "Set stress",
    "regime_set_combo": "Set combo",
    "regime_low_branch": "Low branch",
    "regime_high_branch": "High branch",
    "regime_overlay": "Overlay",
}

WINNER_METHOD_ORDER = [
    "gru",
    "xgb",
    "har",
    "garch_t",
    "figarch",
    "static_best",
    "rolling_best",
    "regime_set_combo",
    "regime_low_branch",
    "regime_high_branch",
    "regime_overlay",
]


# =========================================================
# I/O and helpers
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate simplified conference-ready figures.")
    p.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the cross-asset CSV result files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures. Default: <input-dir>/conference_figures",
    )
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation setting to visualize. Default: full",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def method_label(method: str) -> str:
    return DISPLAY_NAME.get(method, method)


def save_fig(fig: plt.Figure, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def load_tables(input_dir: Path) -> Dict[str, pd.DataFrame]:
    req = {
        "method_agg": input_dir / "cross_asset_method_aggregate.csv",
        "win_counts": input_dir / "cross_asset_method_win_counts.csv",
        "rel_agg": input_dir / "cross_asset_relative_aggregate.csv",
        "rel_detail": input_dir / "cross_asset_relative_to_baselines.csv",
    }
    missing = [str(v) for v in req.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return {k: pd.read_csv(v) for k, v in req.items()}


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: int = 11,
    fc: str = "#F5F7FA",
    ec: str = "#2F3B52",
    rounding: float = 0.02,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#2F3B52"),
    )


def robust_text_color(v: float, vmax: float) -> str:
    if not np.isfinite(v):
        return "black"
    return "white" if v >= 0.55 * vmax else "black"


def annotate_vertical_bars(
    ax: plt.Axes,
    bars: Iterable,
    fmt: str = "{:.3f}",
    dx: float = 0.0,
    dy: float = 0.004,
    fontsize: int = 7,
) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(
                b.get_x() + b.get_width() / 2 + dx,
                h + dy,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                clip_on=False,
            )


# =========================================================
# Figure 1: simplified method schematic
# =========================================================
def plot_figure1_schematic_simple(outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.28
    h = 0.34
    w = 0.16

    xs = [0.03, 0.23, 0.43, 0.63, 0.83]
    labels = [
        "Daily inputs\nETF panel + macro",
        "Candidate pool\nHAR / GRU / XGB\nGARCH-t / FIGARCH",
        "Online scoring\nQLIKE + underprediction\n+ regime similarity",
        "State routing\ncalm / stress specialists\nlocal-shrunk selection",
        "Final overlay\nstress-gated blend\n+ HAR floor",
    ]
    colors = ["#EAF3FF", "#F3F8E8", "#FFF4DD", "#EAF7F0", "#F3EFFF"]

    for x, text, color in zip(xs, labels, colors):
        draw_box(ax, x, y, w, h, text, fontsize=10, fc=color)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.set_title("Figure 1. Simplified regime-aware specialist routing architecture", pad=10)
    save_fig(fig, outpath)


# =========================================================
# Figure 2: winner heatmap + optional mean-rank heatmap
# =========================================================
def compute_mean_rank_heatmap(
    rel_detail: pd.DataFrame,
    ablation: str,
    regimes: List[str],
    methods: List[str],
) -> pd.DataFrame:
    df = rel_detail[(rel_detail["ablation"] == ablation) & (rel_detail["regime"].isin(regimes))].copy()
    df = df[df["method"].isin(methods)].copy()

    pieces = []
    for regime in regimes:
        sub = df[df["regime"] == regime].copy()
        if sub.empty:
            continue
        rank_df = sub.copy()
        rank_df["rank_qlike"] = rank_df.groupby("asset")["qlike_var"].rank(method="average", ascending=True)
        agg = rank_df.groupby("method", as_index=False)["rank_qlike"].mean()
        agg["regime"] = regime
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame(index=methods, columns=regimes, dtype=float)

    out = pd.concat(pieces, ignore_index=True)
    pivot = out.pivot(index="method", columns="regime", values="rank_qlike")
    pivot = pivot.reindex(index=methods, columns=regimes)
    return pivot


def plot_heatmap(
    values: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    outpath: Path,
    value_fmt: str,
    cmap: str,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, pad=10)

    vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    value_fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=robust_text_color(v, vmax),
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)
    save_fig(fig, outpath)


def plot_figure2_winner_heatmap(
    win_counts: pd.DataFrame,
    rel_detail: pd.DataFrame,
    ablation: str,
    outpath_main: Path,
    outpath_bonus_mean_rank: Path,
) -> None:
    df = win_counts[win_counts["ablation"] == ablation].copy()
    df = df[df["regime"].isin(REGIME_ORDER)].copy()

    methods_present = [m for m in WINNER_METHOD_ORDER if m in set(df["winner_method"])]
    if not methods_present:
        methods_present = sorted(df["winner_method"].dropna().unique().tolist())

    pivot = (
        df.pivot_table(
            index="winner_method",
            columns="regime",
            values="n_asset_wins",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=methods_present, columns=REGIME_ORDER)
        .fillna(0.0)
    )

    plot_heatmap(
        values=pivot.values,
        row_labels=[method_label(m) for m in pivot.index],
        col_labels=[r.capitalize() for r in pivot.columns],
        title=f"Figure 2. Regime-conditioned winner heatmap ({ablation} design)",
        outpath=outpath_main,
        value_fmt="{:.0f}",
        cmap="Blues",
        cbar_label="Number of asset wins",
    )

    mean_rank = compute_mean_rank_heatmap(
        rel_detail=rel_detail,
        ablation=ablation,
        regimes=REGIME_ORDER,
        methods=methods_present,
    )

    if not mean_rank.empty:
        plot_heatmap(
            values=mean_rank.values,
            row_labels=[method_label(m) for m in mean_rank.index],
            col_labels=[r.capitalize() for r in mean_rank.columns],
            title=f"Bonus. Regime-conditioned mean-rank heatmap ({ablation} design)",
            outpath=outpath_bonus_mean_rank,
            value_fmt="{:.2f}",
            cmap="viridis_r",
            cbar_label="Average rank (lower is better)",
        )


# =========================================================
# Figure 3a: main 5-asset comparison excluding TLT
# =========================================================
def plot_figure3a_asset_bars(rel_detail: pd.DataFrame, ablation: str, outpath: Path) -> None:
    methods = ["regime_overlay", "har", "rolling_best"]
    df = rel_detail[
        (rel_detail["ablation"] == ablation)
        & (rel_detail["regime"] == "all")
        & (rel_detail["method"].isin(methods))
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for Figure 3a with ablation={ablation}")

    pivot = df.pivot(index="asset", columns="method", values="qlike_var")
    pivot = pivot.reindex(ASSET_ORDER_EX_TLT)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.23

    b1 = ax.bar(x - width, pivot["regime_overlay"].values, width, label="Overlay")
    b2 = ax.bar(x, pivot["har"].values, width, label="HAR")
    b3 = ax.bar(x + width, pivot["rolling_best"].values, width, label="Rolling best")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("QLIKE")
    ax.set_title(f"Figure 3a. Overlay vs HAR vs rolling best by asset ({ablation} design)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    annotate_vertical_bars(ax, b1, dx=-0.025, dy=0.003)
    annotate_vertical_bars(ax, b2, dx=0.000, dy=0.008)
    annotate_vertical_bars(ax, b3, dx=0.025, dy=0.003)

    note = "Five-asset main panel excludes TLT to avoid scale distortion."
    ax.text(0.01, -0.16, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    save_fig(fig, outpath)


# =========================================================
# Figure 3b: separate TLT robustness panel
# =========================================================
def plot_figure3b_tlt_robustness(rel_detail: pd.DataFrame, ablation: str, outpath: Path) -> None:
    methods = ["regime_overlay", "har", "rolling_best"]
    df = rel_detail[
        (rel_detail["ablation"] == ablation)
        & (rel_detail["regime"] == "all")
        & (rel_detail["method"].isin(methods))
        & (rel_detail["asset"] == "TLT")
    ].copy()

    if df.empty:
        raise ValueError(f"No TLT rows found for Figure 3b with ablation={ablation}")

    vals = (
        df.set_index("method")
        .reindex(methods)["qlike_var"]
        .astype(float)
        .values
    )
    labels = ["Overlay", "HAR", "Rolling best"]

    # 更宽、更矮，适合单栏
    fig, ax = plt.subplots(figsize=(7.0, 2.9))

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.55)

    ax.set_yscale("log")
    ax.set_ylabel("QLIKE (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)

    # 不要图内标题，交给 LaTeX caption
    # ax.set_title(...)

    # 给上下留一点呼吸空间，但不要太高
    positive_vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(positive_vals) > 0:
        ymin = min(positive_vals.min() * 0.6, 0.2)
        ymax = positive_vals.max() * 1.6
        ax.set_ylim(ymin, ymax)

    # 顶部数字简洁一点
    for b, v in zip(bars, vals):
        if np.isfinite(v) and v > 0:
            label = f"{v:.2e}" if v >= 1000 else f"{v:.3f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.05,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )

    # 不要图内底部说明，交给 LaTeX caption
    # note = "Overlay remains finite while HAR and rolling-best blow up on TLT."
    # ax.text(...)

    fig.tight_layout()
    save_fig(fig, outpath)


# =========================================================
# Figure 4: delta QLIKE by regime vs baselines
# =========================================================
def plot_figure4_delta_by_regime(rel_agg: pd.DataFrame, ablation: str, outpath: Path) -> None:
    df = rel_agg[
        (rel_agg["ablation"] == ablation)
        & (rel_agg["method"] == "regime_overlay")
        & (rel_agg["regime"].isin(REGIME_ORDER))
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for Figure 4 with ablation={ablation}")

    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    df = df.sort_values("regime")

    x = np.arange(len(REGIME_ORDER))
    width = 0.34

    y1 = df["delta_qlike_vs_rolling_best_median"].values.astype(float)
    y2 = df["delta_qlike_vs_har_median"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    b1 = ax.bar(x - width / 2, y1, width, label="vs rolling best")
    b2 = ax.bar(x + width / 2, y2, width, label="vs HAR")

    vals = np.concatenate([y1[np.isfinite(y1)], y2[np.isfinite(y2)], np.array([0.0])])
    y_min = float(np.min(vals))
    y_max = float(np.max(vals))
    y_span = max(y_max - y_min, 0.04)
    pad = 0.12 * y_span
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in REGIME_ORDER])
    ax.set_ylabel("ΔQLIKE (median across assets)")
    ax.set_title(f"Figure 4. Overlay ΔQLIKE by regime relative to baselines ({ablation} design)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    offset = 0.04 * y_span
    for bars in [b1, b2]:
        for bar in bars:
            h = float(bar.get_height())
            y = h + offset if h >= 0 else h - offset
            va = "bottom" if h >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{h:.3f}",
                ha="center",
                va=va,
                fontsize=8,
                clip_on=False,
            )

    note = "Negative ΔQLIKE means Overlay is better than the baseline."
    ax.text(0.01, -0.16, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    save_fig(fig, outpath)


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / DEFAULT_OUTPUT_SUBDIR
    ablation = args.ablation

    ensure_dir(output_dir)
    tables = load_tables(input_dir)

    plot_figure1_schematic_simple(output_dir / "fig1_method_schematic_simple.png")

    plot_figure2_winner_heatmap(
        win_counts=tables["win_counts"],
        rel_detail=tables["rel_detail"],
        ablation=ablation,
        outpath_main=output_dir / "fig2_regime_winner_heatmap.png",
        outpath_bonus_mean_rank=output_dir / "fig2b_regime_mean_rank_heatmap.png",
    )

    plot_figure3a_asset_bars(
        rel_detail=tables["rel_detail"],
        ablation=ablation,
        outpath=output_dir / "fig3a_overlay_vs_har_vs_rolling_5assets.png",
    )

    plot_figure3b_tlt_robustness(
        rel_detail=tables["rel_detail"],
        ablation=ablation,
        outpath=output_dir / "fig3b_tlt_robustness.png",
    )

    plot_figure4_delta_by_regime(
        rel_agg=tables["rel_agg"],
        ablation=ablation,
        outpath=output_dir / "fig4_overlay_delta_qlike_by_regime.png",
    )

    print("=" * 88)
    print("Saved conference figures to:")
    print(output_dir)
    print("- fig1_method_schematic_simple.png")
    print("- fig2_regime_winner_heatmap.png")
    print("- fig2b_regime_mean_rank_heatmap.png   [bonus]")
    print("- fig3a_overlay_vs_har_vs_rolling_5assets.png")
    print("- fig3b_tlt_robustness.png")
    print("- fig4_overlay_delta_qlike_by_regime.png")
    print("=" * 88)


if __name__ == "__main__":
    main()
