
"""Paper-style plotting utilities for trajectory reconstruction outputs.

The palette, hatching, font sizes, and grouped-boxplot layout follow the older
``Recon_paper_figures_v4_with_supplementary_rmse`` notebook, but the readers are
adapted to the cleaned package outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# Color-blind-friendly palette inherited from the reference figure notebook
# ---------------------------------------------------------------------
C_TRUTH_LINE = "#0072B2"   # blue
C_TRUTH_PTS  = "#56B4E9"   # light blue
C_ANCHOR     = "#D55E00"   # vermillion
C_STRAIGHT   = "#6E6E6E"   # dark gray
C_MOVE       = "#CC79A7"   # reddish purple
C_ENV        = "#E69F00"   # orange
C_TOPK       = "#8C564B"   # muted brown
C_CONTEXT    = "#BDBDBD"   # light gray
C_KIN        = "#009E73"   # green
C_RTG        = "#7F3C8D"   # deep purple
C_BB         = "#00A6A6"   # teal

# Feature colors from the temporal-feature figure family.
FEATURE_COLORS = {
    "displacement": "#0072B2",
    "step": "#009E73",
    "directness": "#E69F00",
    "turning": "#CC79A7",
    "housing": "#D55E00",
    "stream": "#56B4E9",
    "slope": "#7F3C8D",
    "elevation": "#8C6D31",
}

METHOD_ORDER = [
    "linear",
    "heading_hermite",
    "rtg_bridge",
    "brownian_bridge",
    "pretrained_motif_adaptive_selector_v2",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
    "pretrained_motif_robust_global",
    "pretrained_motif_guarded",
    "pretrained_motif_weighted_cost_rank_K20_b0.5",
    "pretrained_motif_weighted_probability_K20_b0.5",
    "pretrained_motif_top1",
]

METHOD_LABELS = {
    "linear": "Straight line",
    "straight": "Straight line",
    "heading_hermite": "Heading Hermite",
    "cubic_hermite": "Heading Hermite",
    "rtg_bridge": "RTG bridge",
    "brownian_bridge": "Brownian bridge",
    "pretrained_motif": "Pretrained motif",
    "pretrained_motif": "Pretrained motif",
    "pretrained_motif_gb_residual_rescue_selector": "Pretrained motif",
    "pretrained_motif_robust_global": "Pretrained motif",
    "pretrained_motif_guarded": "Pretrained motif",
    "pretrained_motif_K20_b0.25": "Motif K20 β=0.25",
    "pretrained_motif_K20_b0.35": "Motif K20 β=0.35",
    "pretrained_motif_K20_b0.5": "Motif K20 β=0.5",
    "pretrained_motif_K20_b0.75": "Motif K20 β=0.75",
    "pretrained_motif_weighted_cost_rank_K20_b0.25": "Motif K20 cost-rank β=0.25",
    "pretrained_motif_weighted_cost_rank_K20_b0.35": "Motif K20 cost-rank β=0.35",
    "pretrained_motif_weighted_cost_rank_K20_b0.5": "Motif K20 cost-rank β=0.5",
    "pretrained_motif_weighted_cost_rank_K20_b0.75": "Motif K20 cost-rank β=0.75",
    "pretrained_motif_weighted_probability_K20_b0.25": "Motif K20 prob. β=0.25",
    "pretrained_motif_weighted_probability_K20_b0.35": "Motif K20 prob. β=0.35",
    "pretrained_motif_weighted_probability_K20_b0.5": "Motif K20 prob. β=0.5",
    "pretrained_motif_weighted_probability_K20_b0.75": "Motif K20 prob. β=0.75",
    "pretrained_motif_paper": "Pretrained motif",
    "pretrained_motif_top1": "Motif Top-1",
    "motif_top1_diagnostic": "Motif Top-1",
    "fewshot_weighted": "Few-shot motif",
}

METHOD_COLORS = {
    "linear": C_STRAIGHT,
    "straight": C_STRAIGHT,
    "heading_hermite": C_KIN,
    "cubic_hermite": C_KIN,
    "rtg_bridge": C_RTG,
    "brownian_bridge": C_BB,
    "pretrained_motif": C_ENV,
    "pretrained_motif": C_ENV,
    "pretrained_motif_robust_global": C_ENV,
    "pretrained_motif_guarded": C_ENV,
    "pretrained_motif_K20_b0.25": C_ENV,
    "pretrained_motif_K20_b0.35": C_ENV,
    "pretrained_motif_K20_b0.5": C_ENV,
    "pretrained_motif_K20_b0.75": C_ENV,
    "pretrained_motif_paper": C_ENV,
    "pretrained_motif_top1": C_MOVE,
    "motif_top1_diagnostic": C_MOVE,
    "fewshot_weighted": C_TOPK,
}

METHOD_HATCHES = {
    "linear": "",
    "straight": "",
    "heading_hermite": "///",
    "cubic_hermite": "///",
    "rtg_bridge": "\\",
    "brownian_bridge": "xx",
    "pretrained_motif": "",
    "pretrained_motif": "",
    "pretrained_motif_robust_global": "",
    "pretrained_motif_guarded": "",
    "pretrained_motif_K20_b0.25": "",
    "pretrained_motif_K20_b0.35": "",
    "pretrained_motif_K20_b0.5": "",
    "pretrained_motif_K20_b0.75": "",
    "pretrained_motif_paper": "",
    "pretrained_motif_top1": "",
    "motif_top1_diagnostic": "",
    "fewshot_weighted": "",
}

PROPOSED_METHODS = {
    "pretrained_motif_adaptive_selector_v2",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
    "pretrained_motif",
    "pretrained_motif",
    "pretrained_motif_robust_global",
    "pretrained_motif_guarded",
    "pretrained_motif_K20_b0.25",
    "pretrained_motif_K20_b0.35",
    "pretrained_motif_K20_b0.5",
    "pretrained_motif_K20_b0.75",
}

RNG = np.random.default_rng(42)
MAX_POINTS_PER_BOX = 3000


def set_paper_style() -> None:
    """Apply the manuscript plotting style from the reference figure notebook."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_label(method: str) -> str:
    method = str(method)
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method.startswith("pretrained_motif_blend_top1"):
        return "Motif blend Top-1/K20"
    if method.startswith("pretrained_motif_blend_direct"):
        return "Motif blend direct/K20"
    if method.startswith("pretrained_motif_weighted_cost_rank"):
        return "Motif weighted cost-rank"
    if method.startswith("pretrained_motif_weighted_probability"):
        return "Motif weighted probability"
    return method.replace("_", " ")


def method_color(method: str) -> str:
    method = str(method)
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    if method.startswith("pretrained_motif"):
        return C_ENV
    return "lightgray"


def method_hatch(method: str) -> str:
    method = str(method)
    if method in METHOD_HATCHES:
        return METHOD_HATCHES[method]
    if method.startswith("pretrained_motif"):
        return ""
    return ""


def setting_label(setting: str) -> str:
    """Short display label such as ``240 → 60 min``."""
    import re
    s = str(setting)
    m = re.search(r"(\d+)\s*min[_-]to[_-](\d+)\s*min", s)
    if m:
        return f"{m.group(1)} → {m.group(2)} min"
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return f"{nums[0]} → {nums[1]} min"
    return s.replace("_", " ")


def save_fig(fig, output_dir: str | Path, basename: str, save_pdf: bool = True, dpi: int = 300) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{basename}.png"
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    print("Saved:", png, flush=True)
    if save_pdf:
        pdf = output_dir / f"{basename}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("Saved:", pdf, flush=True)


def finite_values(df: pd.DataFrame, value_col: str) -> np.ndarray:
    if value_col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


def clip_values_for_display(vals: Sequence[float], clip_quantiles=(1, 99)) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if clip_quantiles is not None and len(vals) >= 20:
        qlo, qhi = np.percentile(vals, clip_quantiles)
        clipped = vals[(vals >= qlo) & (vals <= qhi)]
        if len(clipped) >= 5:
            vals = clipped
    if len(vals) > MAX_POINTS_PER_BOX:
        vals = RNG.choice(vals, size=MAX_POINTS_PER_BOX, replace=False)
    return vals


def metric_ylim(data: pd.DataFrame, value_cols: Sequence[str], clip_quantiles=(1, 99), include_zero=True, pad_frac=0.08):
    vals_all = []
    for col in value_cols:
        if col in data.columns:
            vals_all.append(clip_values_for_display(finite_values(data, col), clip_quantiles))
    vals_all = [v for v in vals_all if len(v)]
    if not vals_all:
        return None
    vals = np.concatenate(vals_all)
    if not len(vals):
        return None
    ymin = float(np.nanmin(vals))
    ymax = float(np.nanmax(vals))
    if include_zero:
        ymin = min(0.0, ymin)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * pad_frac
    return ymin - pad, ymax + pad


def ordered_methods(data: pd.DataFrame, preferred: Sequence[str] | None = None) -> list[str]:
    available = list(pd.Series(data["method"].dropna().astype(str).unique()).values) if "method" in data.columns else []
    pref = list(preferred) if preferred is not None else METHOD_ORDER
    out = [m for m in pref if m in available]
    out.extend([m for m in available if m not in out])
    return out


def ordered_settings(data: pd.DataFrame, preferred: Sequence[str] | None = None) -> list[str]:
    if "setting_name" not in data.columns:
        return []
    available = list(pd.Series(data["setting_name"].dropna().astype(str).unique()).values)
    if preferred is None:
        # Prefer increasing coarse interval while preserving nonstandard settings.
        def key(s):
            import re
            nums = [int(x) for x in re.findall(r"\d+", str(s))]
            return (nums[1] if len(nums) > 1 else 999, nums[0] if nums else 999, str(s))
        return sorted(available, key=key)
    out = [s for s in preferred if s in available]
    out.extend([s for s in available if s not in out])
    return out



PAPER_PROPOSED_METHOD = "pretrained_motif_paper"

PAPER_BASELINE_METHODS = ["linear", "heading_hermite", "rtg_bridge", "brownian_bridge"]

PREFERRED_PROPOSED_METHODS = [
    "pretrained_motif_gb_residual_rescue_selector",
    "pretrained_motif_conservative_oracle_switcher",
    "pretrained_motif_oracle_distilled_selector",
    "pretrained_motif_adaptive_selector_v2",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.5_lamconf",
    "pretrained_motif_robust_global",
    "pretrained_motif_guarded",
    "pretrained_motif",
    "pretrained_motif_weighted_probability_K20_b0.25",
    "pretrained_motif_weighted_probability_K20_b0.35",
    "pretrained_motif_weighted_probability_K20_b0.5",
    "pretrained_motif_weighted_probability_K20_b0.75",
    "pretrained_motif_weighted_cost_rank_K20_b0.25",
    "pretrained_motif_weighted_cost_rank_K20_b0.35",
    "pretrained_motif_weighted_cost_rank_K20_b0.5",
    "pretrained_motif_weighted_cost_rank_K20_b0.75",
    "pretrained_motif_top1",
]


def is_proposed_method(method: str) -> bool:
    """Return True for any internal pretrained motif variant."""
    return str(method).startswith("pretrained_motif") or str(method).startswith("motif_")


def _method_summary_lookup(method_summary: pd.DataFrame | None) -> pd.DataFrame:
    if method_summary is None or method_summary.empty or "method" not in method_summary.columns:
        return pd.DataFrame()
    return method_summary.copy()


def choose_paper_proposed_method(metrics: pd.DataFrame, method_summary: pd.DataFrame | None = None, preferred: str | None = None) -> str | None:
    """Choose one proposed method to display as the manuscript method.

    Validation and full-test tables may contain dozens of tuning variants. The
    paper figures should show only one proposed method, not every candidate in
    the tuning grid. If the guarded method exists, it is preferred. Otherwise we
    choose the proposed method with the lowest median ADE in the supplied method
    summary/metrics.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return None
    available = set(metrics["method"].dropna().astype(str))
    if preferred and preferred in available:
        return preferred
    for m in PREFERRED_PROPOSED_METHODS:
        if m in available:
            return m
    proposed = sorted([m for m in available if is_proposed_method(m) and "guarded_q" not in m])
    if not proposed:
        proposed = sorted([m for m in available if is_proposed_method(m)])
    if not proposed:
        return None
    summary = _method_summary_lookup(method_summary)
    if not summary.empty:
        sub = summary[summary["method"].astype(str).isin(proposed)].copy()
        if not sub.empty:
            sort_col = "ade_ratio_to_linear_median" if "ade_ratio_to_linear_median" in sub.columns else "ADE_median"
            sub[sort_col] = pd.to_numeric(sub[sort_col], errors="coerce")
            sub = sub.sort_values([sort_col, "method"], na_position="last")
            return str(sub.iloc[0]["method"])
    tmp = metrics[metrics["method"].astype(str).isin(proposed)].copy()
    sort_col = "ade_ratio_to_linear" if "ade_ratio_to_linear" in tmp.columns else "ADE"
    tmp[sort_col] = pd.to_numeric(tmp.get(sort_col, np.nan), errors="coerce")
    med = tmp.groupby("method", sort=False)[sort_col].median().sort_values()
    return str(med.index[0]) if len(med) else proposed[0]


def prepare_metrics_for_paper_figures(
    metrics: pd.DataFrame,
    method_summary: pd.DataFrame | None = None,
    proposed_method: str | None = None,
    include_top1_diagnostic: bool = False,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Filter/rename metrics for readable manuscript figures.

    Returns a copy containing only the main baselines plus one selected proposed
    method. The selected proposed method is renamed to
    ``pretrained_motif_paper`` so legends remain short and stable even when the
    internal tuning-grid method name is long.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns:
        return pd.DataFrame(), [], None
    chosen = choose_paper_proposed_method(metrics, method_summary=method_summary, preferred=proposed_method)
    keep = [m for m in PAPER_BASELINE_METHODS if m in set(metrics["method"].astype(str))]
    if chosen is not None:
        keep.append(chosen)
    if include_top1_diagnostic and "pretrained_motif_top1" in set(metrics["method"].astype(str)) and "pretrained_motif_top1" not in keep:
        keep.append("pretrained_motif_top1")
    out = metrics[metrics["method"].astype(str).isin(keep)].copy()
    if chosen is not None:
        out.loc[out["method"].astype(str).eq(chosen), "method"] = PAPER_PROPOSED_METHOD
        keep = [PAPER_PROPOSED_METHOD if m == chosen else m for m in keep]
    # Drop duplicate rows caused by multiple candidate IDs after collapsing.
    if "task_uid" in out.columns:
        out = out.drop_duplicates(["task_uid", "method"], keep="first")
    return out, keep, chosen


def selected_method_note(chosen: str | None) -> str:
    if not chosen:
        return "No proposed method found."
    if chosen == "pretrained_motif_conservative_oracle_switcher":
        return "Displayed proposed method: V20.9 conservative oracle-gap switcher."
    if chosen == "pretrained_motif_adaptive_selector_v2":
        return "Displayed proposed method: V20.7 residual-flow adaptive selector."
    if chosen == "pretrained_motif_robust_global":
        return "Displayed proposed method: validation-selected robust global pretrained motif."
    if chosen == "pretrained_motif_guarded":
        return "Displayed proposed method: validation-guarded pretrained motif."
    return f"Displayed proposed method: {chosen} (shown as Pretrained motif in figures)."



def legend_handles(methods: Sequence[str], include_hatch: bool = True) -> list[Patch]:
    handles = []
    for method in methods:
        handles.append(Patch(
            facecolor=method_color(method),
            edgecolor="black",
            hatch=method_hatch(method) if include_hatch else "",
            label=method_label(method),
            linewidth=0.8,
        ))
    return handles


def draw_grouped_boxplot_by_setting_method(
    ax,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str | None = None,
    title: str | None = None,
    methods: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    ylim=None,
    clip_quantiles=(1, 99),
    show_xlabels: bool = True,
):
    """Grouped boxplot using reference notebook's restrained style and hatches."""
    if value_col not in data.columns:
        ax.text(0.5, 0.5, f"Missing {value_col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    methods = ordered_methods(data, methods)
    settings = ordered_settings(data, settings)
    if not methods or not settings:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions = []
    values = []
    method_for_box = []
    tick_pos = []
    tick_labels = []
    group_gap = 1.3
    n_methods = len(methods)
    for gi, setting in enumerate(settings):
        base = gi * (n_methods + group_gap)
        tick_pos.append(base + (n_methods - 1) / 2)
        tick_labels.append(setting_label(setting))
        for mi, method in enumerate(methods):
            vals = finite_values(data[(data["setting_name"].astype(str) == str(setting)) & (data["method"].astype(str) == str(method))], value_col)
            vals = clip_values_for_display(vals, clip_quantiles)
            if len(vals) == 0:
                vals = np.array([np.nan])
            positions.append(base + mi)
            values.append(vals)
            method_for_box.append(method)

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.70,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
    )
    for patch, method in zip(bp["boxes"], method_for_box):
        patch.set_facecolor(method_color(method))
        patch.set_hatch(method_hatch(method))
        patch.set_alpha(0.95)

    ax.set_title(title or value_col)
    ax.set_ylabel(ylabel or value_col)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_xticks(tick_pos)
    if show_xlabels:
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    else:
        ax.set_xticklabels([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_primary_performance(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    methods: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    save_pdf: bool = True,
    method_summary: pd.DataFrame | None = None,
    proposed_method: str | None = None,
):
    """Figure 3-style primary performance figure.

    If ``methods`` is omitted, dozens of tuning variants are collapsed into one
    manuscript method labeled ``Pretrained motif``.
    """
    set_paper_style()
    if methods is None:
        metrics, methods, chosen = prepare_metrics_for_paper_figures(metrics, method_summary=method_summary, proposed_method=proposed_method)
        print(selected_method_note(chosen), flush=True)
    else:
        methods = ordered_methods(metrics, methods)
    settings = ordered_settings(metrics, settings)

    if "ade_ratio_to_linear" in metrics.columns:
        metric_specs = [
            ("ade_ratio_to_linear", "ADE / linear", "Timestamp error relative to linear"),
            ("rmse_ratio_to_linear", "RMSE / linear", "RMSE relative to linear"),
            ("dtw_ratio_to_linear", "DTW / linear", "Shape similarity relative to linear"),
            ("path_length_ratio_error", "Path-length ratio error", "Path length preservation"),
        ]
    else:
        metric_specs = [
            ("ADE", "ADE (m)", "Average displacement error"),
            ("RMSE", "Time-indexed RMSE (m)", "Time-indexed RMSE"),
            ("path_length_log_error", "Path-length log error", "Path length preservation"),
            ("directness_error", "Directness-score error", "Directness preservation"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6), sharex=False)
    axes = axes.ravel()
    for ax, (col, ylabel, title) in zip(axes, metric_specs):
        ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=True)
        if col.endswith("_ratio_to_linear"):
            # Ratios are most interpretable around the linear reference line.
            ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=False)
        draw_grouped_boxplot_by_setting_method(ax, metrics, col, ylabel=ylabel, title=title, methods=methods, settings=settings, ylim=ylim)
        if col.endswith("_ratio_to_linear"):
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)

    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, output_dir, "Fig3_reconstruction_performance", save_pdf=save_pdf)
    return fig


def plot_supplementary_rmse(metrics: pd.DataFrame, output_dir: str | Path, methods: Sequence[str] | None = None, settings: Sequence[str] | None = None, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    """Supplementary Figure S1-style spatial and time-indexed RMSE."""
    set_paper_style()
    if methods is None:
        metrics, methods, _ = prepare_metrics_for_paper_figures(metrics, method_summary=method_summary, proposed_method=proposed_method)
    else:
        methods = ordered_methods(metrics, methods)
    col_spatial = "spatial_RMSE" if "spatial_RMSE" in metrics.columns else "spatial_rmse_m"
    col_time = "RMSE" if "RMSE" in metrics.columns else "time_indexed_rmse_m"
    settings = ordered_settings(metrics, settings)
    ylim = metric_ylim(metrics, [col_spatial, col_time], clip_quantiles=(1, 99), include_zero=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
    draw_grouped_boxplot_by_setting_method(axes[0], metrics, col_spatial, ylabel="RMSE (m)", title="Spatial RMSE to truth path", methods=methods, settings=settings, ylim=ylim)
    draw_grouped_boxplot_by_setting_method(axes[1], metrics, col_time, ylabel=None, title="Time-indexed RMSE", methods=methods, settings=settings, ylim=ylim)
    axes[1].set_ylabel("")
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save_fig(fig, output_dir, "FigS1_spatial_time_indexed_RMSE", save_pdf=save_pdf)
    return fig


def plot_environmental_exposure(metrics: pd.DataFrame, output_dir: str | Path, methods: Sequence[str] | None = None, settings: Sequence[str] | None = None, save_pdf: bool = True):
    """Figure 4-style environmental exposure errors.

    This release shows all available annotated/raster-derived layers rather than only the
    first matching generic variable.  It supports the puma layers
    (Dist_housing, Dist_stream, Elevation, Slope) and Thailand layers
    (slope, bt, gr, sb), using absolute exposure-error columns when available.
    """
    preferred = [
        ("env_abs_error_Dist_housing", "Housing distance"),
        ("env_abs_error_Dist_stream", "Stream distance"),
        ("env_abs_error_Elevation", "Elevation"),
        ("env_abs_error_Slope", "Slope\n(puma)"),
        ("env_abs_error_slope", "Slope\n(Thailand)"),
        ("env_abs_error_bt", "Banteng"),
        ("env_abs_error_gr", "Gaur"),
        ("env_abs_error_sb", "Sambar"),
    ]
    env_cols = []
    for c, label in preferred:
        if c in metrics.columns and pd.to_numeric(metrics[c], errors="coerce").notna().any():
            env_cols.append((c, label))
    if not env_cols:
        possible = [c for c in metrics.columns if c.lower().startswith("env_abs_error_") or (c.lower().startswith("env_error_") and c.lower() != "env_error_mean")]
        for c in possible:
            if pd.to_numeric(metrics[c], errors="coerce").notna().any():
                label = c.replace("env_abs_error_", "").replace("env_error_", "").replace("_", " ").title()
                env_cols.append((c, label))
    if not env_cols:
        print("No environmental exposure error columns found; skipping Fig4_environmental_exposure.", flush=True)
        return None

    set_paper_style()
    methods = ordered_methods(metrics, methods)
    settings = ordered_settings(metrics, settings)
    n = len(env_cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.55 * nrows), sharey=False, squeeze=False)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, env_cols):
        # Only show settings where this layer is defined for at least one method.
        valid_settings = []
        for st in settings:
            g = metrics[metrics["setting_name"].astype(str).eq(str(st))]
            if pd.to_numeric(g[col], errors="coerce").notna().any():
                valid_settings.append(st)
        if not valid_settings:
            ax.set_axis_off()
            continue
        ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=True)
        draw_grouped_boxplot_by_setting_method(
            ax,
            metrics,
            col,
            ylabel="Exposure error" if ax is axes[0] else None,
            title=label,
            methods=methods,
            settings=valid_settings,
            ylim=ylim,
        )
        if ax is not axes[0]:
            ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_fig(fig, output_dir, "Fig4_environmental_exposure", save_pdf=save_pdf)
    return fig

def plot_metadata_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    """Optional stratified performance figure by taxon/sex/age when metadata exists."""
    if not {"taxon", "method", "ADE"}.issubset(metrics.columns):
        print("Metadata performance figure skipped: required columns missing.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    if sub.empty:
        print(f"Metadata performance figure skipped: method {method} not found.", flush=True)
        return None
    group_col = "taxon"
    if "sex" in sub.columns and sub["sex"].notna().any():
        sub["taxon_sex"] = sub["taxon"].astype(str) + " / " + sub["sex"].fillna("unknown").astype(str)
        group_col = "taxon_sex"
    taxon_order = ["puma", "cougar", "bobcat", "tiger", "leopard"]
    groups = list(sub[group_col].dropna().astype(str).unique())
    def _gkey(g):
        g0 = str(g).split(" / ")[0]
        return (taxon_order.index(g0) if g0 in taxon_order else 999, str(g))
    groups = sorted(groups, key=_gkey)
    values = [clip_values_for_display(finite_values(sub[sub[group_col].astype(str).eq(g)], "ADE"), (1, 99)) for g in groups]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.70 * len(groups)), 4.2))
    bp = ax.boxplot(values, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
    for patch in bp["boxes"]:
        patch.set_facecolor(method_color(method))
        patch.set_edgecolor("black")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([g.replace(" / ", "\n") for g in groups], rotation=0, ha="center")
    ax.set_ylabel("ADE (m)")
    ax.set_title(f"{method_label(method)} accuracy by metadata group")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_fig(fig, output_dir, "FigS_metadata_stratified_ADE", save_pdf=save_pdf)
    return fig


def plot_reconstruction_examples(
    metrics: pd.DataFrame,
    task_points: pd.DataFrame,
    selected_paths: pd.DataFrame,
    output_dir: str | Path,
    method: str | None = None,
    n_examples: int = 4,
    save_pdf: bool = True,
):
    """Figure 2-style example truth/reconstruction maps for held-out tasks."""
    required = {"task_uid", "point_order", "x", "y"}
    if task_points is None or selected_paths is None or task_points.empty or selected_paths.empty:
        print("Example path figure skipped: missing task_points or selected_paths.", flush=True)
        return None
    if not required.issubset(task_points.columns) or not required.issubset(selected_paths.columns):
        print("Example path figure skipped: path tables do not have required columns.", flush=True)
        return None
    set_paper_style()
    available_methods = set(metrics["method"].astype(str)) if "method" in metrics.columns else set()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else None)
    method = method or (list(available_methods)[0] if available_methods else None)
    if method is None:
        print("Example path figure skipped: no method available.", flush=True)
        return None
    mdf = metrics[metrics["method"].astype(str).eq(method)].copy()
    if mdf.empty:
        print(f"Example path figure skipped: method {method} not found.", flush=True)
        return None
    # Select examples near the method's median ADE and across settings when possible.
    # When path-shape diagnostics are available, prefer examples whose selected
    # path has at least some detour so Fig. 2 does not visually collapse into a
    # duplicate of the straight-line baseline.
    mdf["ADE"] = pd.to_numeric(mdf["ADE"], errors="coerce")
    med = mdf["ADE"].median()
    mdf["_dist_to_med"] = (mdf["ADE"] - med).abs()
    if {"path_length_m", "truth_path_length_m"}.issubset(mdf.columns):
        mdf["_path_shape_score"] = pd.to_numeric(mdf["path_length_m"], errors="coerce") / pd.to_numeric(mdf["truth_path_length_m"], errors="coerce").replace(0, np.nan)
        mdf["_shape_ok"] = mdf["_path_shape_score"].fillna(0).ge(0.45)
    else:
        mdf["_shape_ok"] = True
    if "setting_name" in mdf.columns:
        chosen = []
        for _, g in mdf.sort_values(["_shape_ok", "_dist_to_med"], ascending=[False, True]).groupby("setting_name", sort=False):
            uid = str(g.iloc[0]["task_uid"])
            if uid in set(task_points["task_uid"].astype(str)) and uid in set(selected_paths["task_uid"].astype(str)):
                chosen.append(uid)
            if len(chosen) >= n_examples:
                break
        if len(chosen) < n_examples:
            for uid in mdf.sort_values(["_shape_ok", "_dist_to_med"], ascending=[False, True])["task_uid"].astype(str):
                if uid not in chosen and uid in set(task_points["task_uid"].astype(str)) and uid in set(selected_paths["task_uid"].astype(str)):
                    chosen.append(uid)
                if len(chosen) >= n_examples:
                    break
    else:
        chosen = list(mdf.sort_values("_dist_to_med")["task_uid"].astype(str).head(n_examples))
    if not chosen:
        print("Example path figure skipped: no overlapping task IDs in path tables.", flush=True)
        return None

    n = len(chosen)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.0 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, uid in zip(axes, chosen):
        tr = task_points[task_points["task_uid"].astype(str).eq(uid)].sort_values("point_order")
        paths = selected_paths[selected_paths["task_uid"].astype(str).eq(uid)].copy()
        # Plot truth.
        ax.plot(tr["x"], tr["y"], color=C_TRUTH_LINE, linewidth=2.2, label="Ground truth")
        ax.scatter(tr["x"], tr["y"], color=C_TRUTH_PTS, s=8, zorder=3)
        # Sparse endpoints.
        if len(tr) >= 2:
            anchors = tr.iloc[[0, -1]]
            ax.scatter(anchors["x"], anchors["y"], color=C_ANCHOR, s=48, zorder=4, marker="o", label="Sparse endpoints")
        # Plot one representative/proposed path. Different package versions write
        # selected paths with slightly different schemas: some have `path_id`,
        # some have `method`, some have `candidate_id`, and final selected-path
        # exports may contain exactly one path per task.  For paper figures, we
        # choose the requested method when possible and otherwise fall back to
        # the first available path for this task instead of failing.
        if paths.empty:
            continue

        if "method" in paths.columns:
            method_paths = paths[paths["method"].astype(str).eq(str(method))].copy()
            if not method_paths.empty:
                paths = method_paths

        path_key = None
        for candidate_key in ["path_id", "candidate_id", "selection_id", "variant", "method"]:
            if candidate_key in paths.columns:
                path_key = candidate_key
                break

        if path_key is not None:
            path_ids = list(paths[path_key].dropna().astype(str).unique())
            preferred_ids = [
                pid for pid in path_ids
                if str(method) in pid or pid in {str(method), "representative", "pretrained_motif", PAPER_PROPOSED_METHOD}
            ]
            pid = preferred_ids[0] if preferred_ids else (path_ids[0] if path_ids else None)
            rp = paths[paths[path_key].astype(str).eq(str(pid))].copy() if pid is not None else paths.copy()
        else:
            rp = paths.copy()

        sort_cols = [c for c in ["point_order", "step_index", "time", "timestamp", "t"] if c in rp.columns]
        if sort_cols:
            rp = rp.sort_values(sort_cols)
        if {"x", "y"}.issubset(rp.columns) and not rp.empty:
            ax.plot(rp["x"], rp["y"], color=method_color(method), linewidth=2.1, linestyle="--", label=method_label(method))
        row = mdf[mdf["task_uid"].astype(str).eq(uid)].iloc[0]
        title_bits = []
        for c in ["dataset", "taxon", "setting_name"]:
            if c in row.index:
                title_bits.append(str(row[c]))
        ade = row.get("ADE", np.nan)
        ax.set_title(" | ".join(title_bits) + (f"\nADE={ade:.1f} m" if pd.notna(ade) else ""))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[n:]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=C_TRUTH_LINE, lw=2.2, label="Ground truth"),
        Line2D([0], [0], color=method_color(method), lw=2.1, linestyle="--", label=method_label(method)),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ANCHOR, markersize=7, label="Sparse endpoints"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, output_dir, "Fig2_reconstruction_examples", save_pdf=save_pdf)
    return fig


def plot_runtime_summary(runtime: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True):
    """Optional runtime figure using the same calm style."""
    if runtime is None or runtime.empty or "total_seconds" not in runtime.columns:
        print("Runtime figure skipped: no total_seconds column.", flush=True)
        return None
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vals = finite_values(runtime, "total_seconds")
    ax.hist(vals, bins=24, color=C_ENV, edgecolor="black", alpha=0.92)
    ax.set_xlabel("Seconds per reconstruction gap")
    ax.set_ylabel("Number of tasks")
    ax.set_title("User-facing reconstruction runtime per gap")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_fig(fig, output_dir, "FigS_runtime_per_gap", save_pdf=save_pdf)
    return fig


def _transfer_relation_label(value: str) -> str:
    labels = {
        "same_species_same_habitat": "Same species\nsame habitat",
        "same_species_different_habitat": "Same species\ndifferent habitat",
        "different_species_same_habitat": "Different species\nsame habitat",
        "different_species_different_habitat": "Different species\ndifferent habitat",
        "no_training_source": "No training\nsource",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def plot_transfer_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    """Supplementary figure: proposed-method ADE by transfer-support relation.

    This uses the transfer labels written during model building and full testing.
    It is most useful for interpreting cross-species/cross-habitat support, not
    for claiming a strict leave-one-scenario-out transfer experiment.
    """
    # Prefer the actual selected motif/source relation when This release outputs it.
    # Fall back to the coarser available-training-source label from earlier versions.
    relation_candidates = [
        "dominant_source_transfer_relation",
        "top_source_transfer_relation",
        "source_transfer_relation",
        "best_train_transfer_relation",
    ]
    relation_col = next((c for c in relation_candidates if c in metrics.columns), None)
    if relation_col is None:
        print("Transfer performance figure skipped: no transfer-relation column found.", flush=True)
        return None
    if not {"method", "ADE"}.issubset(metrics.columns):
        print("Transfer performance figure skipped: method/ADE columns missing.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    sub = sub[sub[relation_col].notna()].copy()
    if sub.empty:
        print(f"Transfer performance figure skipped: no rows for {method}.", flush=True)
        return None
    relation_order = [
        "same_species_same_habitat",
        "same_species_different_habitat",
        "different_species_same_habitat",
        "different_species_different_habitat",
        "no_training_source",
    ]
    relations = [r for r in relation_order if r in set(sub[relation_col].astype(str))]
    relations += [r for r in sorted(sub[relation_col].dropna().astype(str).unique()) if r not in relations]
    values = [clip_values_for_display(finite_values(sub[sub[relation_col].astype(str).eq(r)], "ADE"), (1, 99)) for r in relations]
    fig, ax = plt.subplots(figsize=(max(6.8, 1.2 * len(relations)), 4.3))
    bp = ax.boxplot(values, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
    for patch in bp["boxes"]:
        patch.set_facecolor(method_color(method))
        patch.set_edgecolor("black")
        patch.set_alpha(0.95)
    ax.set_xticks(range(1, len(relations) + 1))
    ax.set_xticklabels([_transfer_relation_label(r) for r in relations], rotation=0, ha="center")
    ax.set_ylabel("ADE (m)")
    title_suffix = "selected-source relation" if relation_col != "best_train_transfer_relation" else "available-training relation"
    ax.set_title(f"{method_label(method)} accuracy by {title_suffix}")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Add sample sizes under categories.
    ymin, ymax = ax.get_ylim()
    for i, r in enumerate(relations, start=1):
        n = int(sub[sub[relation_col].astype(str).eq(r)]["task_uid"].nunique()) if "task_uid" in sub.columns else len(sub[sub[relation_col].astype(str).eq(r)])
        ax.text(i, ymin + 0.02 * (ymax - ymin), f"n={n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, "FigS_transfer_relation_ADE", save_pdf=save_pdf)
    return fig

# -----------------------------------------------------------------------------
# V20.3 manuscript plotting overrides
# -----------------------------------------------------------------------------
# These definitions intentionally override the earlier plotting functions above.
# They keep the model/scoring outputs unchanged, but improve manuscript figure
# readability and add supplementary violin plots requested after V20.1 review.

DISPLAY_TAXON_LABELS = {
    "puma": "Puma",
    "cougar": "Cougar",
    "bobcat": "Bobcat",
    "tiger": "Tiger",
    "leopard": "Leopard",
}

DISPLAY_HABITAT_LABELS = {
    "SantaCruz": "Santa Cruz",
    "OlympicPeninsula": "Olympic",
    "Thailand_WEFCOM": "Thailand",
}

INTERVAL_PANEL_ORDER = [
    ("240_to_60", "240 → 60 min"),
    ("120_to_5", "120 → 5 min"),
    ("60_to_15", "60 → 15 min"),
    ("60_to_5", "60 → 5 min"),
]


def _display_taxon(value) -> str:
    return DISPLAY_TAXON_LABELS.get(str(value), str(value).replace("_", " ").title())


def _display_habitat(value) -> str:
    return DISPLAY_HABITAT_LABELS.get(str(value), str(value).replace("_", " ").replace("-", " ").title())


def _interval_from_setting(setting: str) -> str:
    return setting_label(setting).replace(" ", "") if "→" in setting_label(setting) else setting_label(setting)


def _setting_display_from_row(row: pd.Series) -> str:
    taxon = _display_taxon(row.get("taxon", "")) if pd.notna(row.get("taxon", np.nan)) else ""
    habitat = _display_habitat(row.get("habitat_id", row.get("study_system", row.get("dataset", ""))))
    interval = setting_label(row.get("setting_name", ""))
    parts = [p for p in [taxon, habitat, interval] if p]
    return "\n".join(parts)


def add_figure_setting(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add a paper-facing setting label that distinguishes species/dataset/interval."""
    out = metrics.copy()
    if out.empty:
        out["figure_setting"] = []
        return out
    if "setting_name" not in out.columns:
        out["figure_setting"] = "All tasks"
        return out
    out["figure_setting"] = out.apply(_setting_display_from_row, axis=1)
    return out


def ordered_figure_settings(data: pd.DataFrame, preferred: Sequence[str] | None = None) -> list[str]:
    if data is None or data.empty:
        return []
    col = "figure_setting" if "figure_setting" in data.columns else "setting_name"
    available = list(pd.Series(data[col].dropna().astype(str).unique()).values)
    if preferred is not None:
        out = [s for s in preferred if s in available]
        out.extend([s for s in available if s not in out])
        return out
    taxon_order = {"Puma": 0, "Cougar": 1, "Bobcat": 2, "Tiger": 3, "Leopard": 4}
    habitat_order = {"Santa Cruz": 0, "Olympic": 1, "Thailand": 2}
    import re
    def key(label: str):
        parts = str(label).split("\n")
        taxon = parts[0] if len(parts) > 0 else ""
        habitat = parts[1] if len(parts) > 1 else ""
        nums = [int(x) for x in re.findall(r"\d+", str(label))]
        coarse = nums[0] if nums else 999
        fine = nums[1] if len(nums) > 1 else 999
        return (fine, coarse, habitat_order.get(habitat, 999), taxon_order.get(taxon, 999), label)
    return sorted(available, key=key)


def draw_grouped_boxplot_by_group_method(
    ax,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str | None = None,
    title: str | None = None,
    methods: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    group_col: str = "figure_setting",
    ylim=None,
    clip_quantiles=(1, 99),
    show_xlabels: bool = True,
):
    if value_col not in data.columns:
        ax.text(0.5, 0.5, f"Missing {value_col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = data.copy()
    if group_col not in data.columns:
        data = add_figure_setting(data)
    methods = ordered_methods(data, methods)
    groups = list(groups) if groups is not None else ordered_figure_settings(data)
    if not methods or not groups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions, values, method_for_box, tick_pos, tick_labels = [], [], [], [], []
    group_gap = 1.35
    n_methods = len(methods)
    for gi, group in enumerate(groups):
        base = gi * (n_methods + group_gap)
        tick_pos.append(base + (n_methods - 1) / 2)
        tick_labels.append(str(group))
        for mi, method in enumerate(methods):
            mask = data[group_col].astype(str).eq(str(group)) & data["method"].astype(str).eq(str(method))
            vals = clip_values_for_display(finite_values(data[mask], value_col), clip_quantiles)
            if len(vals) == 0:
                vals = np.array([np.nan])
            positions.append(base + mi)
            values.append(vals)
            method_for_box.append(method)

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.70,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
    )
    for patch, method in zip(bp["boxes"], method_for_box):
        patch.set_facecolor(method_color(method))
        patch.set_hatch(method_hatch(method))
        patch.set_alpha(0.95)
    ax.set_title(title or value_col)
    ax.set_ylabel(ylabel or value_col)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels if show_xlabels else [], rotation=0, ha="center")
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_grouped_violin_by_group_method(
    ax,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str | None = None,
    title: str | None = None,
    methods: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    group_col: str = "figure_setting",
    ylim=None,
    clip_quantiles=(1, 99),
    show_xlabels: bool = True,
):
    """Pure violin counterpart to the grouped boxplot, using same colors."""
    if value_col not in data.columns:
        ax.text(0.5, 0.5, f"Missing {value_col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = data.copy()
    if group_col not in data.columns:
        data = add_figure_setting(data)
    methods = ordered_methods(data, methods)
    groups = list(groups) if groups is not None else ordered_figure_settings(data)
    if not methods or not groups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions, values, method_for_body, tick_pos, tick_labels = [], [], [], [], []
    group_gap = 1.35
    n_methods = len(methods)
    for gi, group in enumerate(groups):
        base = gi * (n_methods + group_gap)
        tick_pos.append(base + (n_methods - 1) / 2)
        tick_labels.append(str(group))
        for mi, method in enumerate(methods):
            mask = data[group_col].astype(str).eq(str(group)) & data["method"].astype(str).eq(str(method))
            vals = clip_values_for_display(finite_values(data[mask], value_col), clip_quantiles)
            if len(vals) < 2:
                # Matplotlib violinplot needs at least two values; duplicate a singleton.
                if len(vals) == 1 and np.isfinite(vals[0]):
                    vals = np.array([vals[0], vals[0]])
                else:
                    vals = np.array([np.nan, np.nan])
            positions.append(base + mi)
            values.append(vals)
            method_for_body.append(method)
    finite_pairs = [(v, p, m) for v, p, m in zip(values, positions, method_for_body) if np.isfinite(v).any()]
    if not finite_pairs:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    values2, positions2, methods2 = zip(*finite_pairs)
    vp = ax.violinplot(values2, positions=positions2, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, method in zip(vp["bodies"], methods2):
        body.set_facecolor(method_color(method))
        body.set_edgecolor("black")
        body.set_alpha(0.80)
        body.set_linewidth(0.6)
    ax.set_title(title or value_col)
    ax.set_ylabel(ylabel or value_col)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels if show_xlabels else [], rotation=0, ha="center")
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_primary_performance(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    methods: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    save_pdf: bool = True,
    method_summary: pd.DataFrame | None = None,
    proposed_method: str | None = None,
):
    set_paper_style()
    if methods is None:
        metrics, methods, chosen = prepare_metrics_for_paper_figures(metrics, method_summary=method_summary, proposed_method=proposed_method)
        print(selected_method_note(chosen), flush=True)
    else:
        methods = ordered_methods(metrics, methods)
    metrics = add_figure_setting(metrics)
    groups = ordered_figure_settings(metrics, settings)
    if "ade_ratio_to_linear" in metrics.columns:
        metric_specs = [
            ("ade_ratio_to_linear", "ADE / linear", "Timestamp error relative to linear"),
            ("rmse_ratio_to_linear", "RMSE / linear", "RMSE relative to linear"),
            ("dtw_ratio_to_linear", "DTW / linear", "Shape similarity relative to linear"),
            ("path_length_ratio_error", "Path-length ratio error", "Path length preservation"),
        ]
    else:
        metric_specs = [
            ("ADE", "ADE (m)", "Average displacement error"),
            ("RMSE", "Time-indexed RMSE (m)", "Time-indexed RMSE"),
            ("path_length_log_error", "Path-length log error", "Path length preservation"),
            ("directness_error", "Directness-score error", "Directness preservation"),
        ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.3), sharex=False)
    axes = axes.ravel()
    for ax, (col, ylabel, title) in zip(axes, metric_specs):
        ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=not col.endswith("_ratio_to_linear"))
        draw_grouped_boxplot_by_group_method(ax, metrics, col, ylabel=ylabel, title=title, methods=methods, groups=groups, ylim=ylim)
        if col.endswith("_ratio_to_linear"):
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, output_dir, "Fig3_reconstruction_performance", save_pdf=save_pdf)
    return fig


def plot_supplementary_rmse(metrics: pd.DataFrame, output_dir: str | Path, methods: Sequence[str] | None = None, settings: Sequence[str] | None = None, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    set_paper_style()
    if methods is None:
        metrics, methods, _ = prepare_metrics_for_paper_figures(metrics, method_summary=method_summary, proposed_method=proposed_method)
    else:
        methods = ordered_methods(metrics, methods)
    metrics = add_figure_setting(metrics)
    groups = ordered_figure_settings(metrics, settings)
    col_spatial = "spatial_RMSE" if "spatial_RMSE" in metrics.columns else "spatial_rmse_m"
    col_time = "RMSE" if "RMSE" in metrics.columns else "time_indexed_rmse_m"
    ylim = metric_ylim(metrics, [col_spatial, col_time], clip_quantiles=(1, 99), include_zero=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    draw_grouped_boxplot_by_group_method(axes[0], metrics, col_spatial, ylabel="RMSE (m)", title="Spatial RMSE to truth path", methods=methods, groups=groups, ylim=ylim)
    draw_grouped_boxplot_by_group_method(axes[1], metrics, col_time, ylabel=None, title="Time-indexed RMSE", methods=methods, groups=groups, ylim=ylim)
    axes[1].set_ylabel("")
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    save_fig(fig, output_dir, "FigS1_spatial_time_indexed_RMSE", save_pdf=save_pdf)
    return fig


def _plot_environment_subset(metrics: pd.DataFrame, output_dir: str | Path, env_cols: Sequence[tuple[str, str]], basename: str, title_prefix: str, methods: Sequence[str], save_pdf: bool = True):
    valid_env = [(c, lab) for c, lab in env_cols if c in metrics.columns and pd.to_numeric(metrics[c], errors="coerce").notna().any()]
    if not valid_env or metrics.empty:
        print(f"{basename} skipped: no matching environmental columns.", flush=True)
        return None
    metrics = add_figure_setting(metrics)
    groups = ordered_figure_settings(metrics)
    n = len(valid_env)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, valid_env):
        # Only keep figure settings where the layer exists.
        valid_groups = []
        for group in groups:
            g = metrics[metrics["figure_setting"].astype(str).eq(str(group))]
            if pd.to_numeric(g[col], errors="coerce").notna().any():
                valid_groups.append(group)
        ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=True)
        draw_grouped_boxplot_by_group_method(ax, metrics, col, ylabel="Absolute exposure error" if ax is axes[0] else None, title=label, methods=methods, groups=valid_groups, ylim=ylim)
        if ax is not axes[0]:
            ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(title_prefix, y=1.02, fontsize=11)
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, output_dir, basename, save_pdf=save_pdf)
    return fig


def plot_environmental_exposure(metrics: pd.DataFrame, output_dir: str | Path, methods: Sequence[str] | None = None, settings: Sequence[str] | None = None, save_pdf: bool = True):
    """Split environmental exposure by study system to avoid empty categories."""
    set_paper_style()
    if methods is None:
        metrics, methods, _ = prepare_metrics_for_paper_figures(metrics)
    else:
        methods = ordered_methods(metrics, methods)
    if metrics is None or metrics.empty:
        print("Environmental exposure figures skipped: no metrics.", flush=True)
        return None
    puma_env = [
        ("env_abs_error_Dist_housing", "Distance to housing"),
        ("env_abs_error_Dist_stream", "Distance to stream"),
        ("env_abs_error_Elevation", "Elevation"),
        ("env_abs_error_Slope", "Slope"),
    ]
    thailand_env = [
        ("env_abs_error_slope", "Slope"),
        ("env_abs_error_bt", "Banteng"),
        ("env_abs_error_gr", "Gaur"),
        ("env_abs_error_sb", "Sambar"),
    ]
    dataset = metrics.get("dataset", pd.Series([""] * len(metrics))).astype(str)
    taxon = metrics.get("taxon", pd.Series([""] * len(metrics))).astype(str)
    puma_sub = metrics[dataset.str.contains("SantaCruz_puma", case=False, na=False) | taxon.eq("puma")].copy()
    thailand_sub = metrics[dataset.str.contains("Thailand", case=False, na=False) | taxon.isin(["tiger", "leopard"])].copy()
    figs = []
    fig_a = _plot_environment_subset(puma_sub, output_dir, puma_env, "Fig4a_puma_environmental_exposure", "Puma environmental exposure", methods, save_pdf=save_pdf)
    if fig_a is not None:
        figs.append(fig_a)
    fig_b = _plot_environment_subset(thailand_sub, output_dir, thailand_env, "Fig4b_tigerleopard_environmental_exposure", "Tiger/leopard environmental exposure", methods, save_pdf=save_pdf)
    if fig_b is not None:
        figs.append(fig_b)
    return figs if figs else None


def _truth_curvature_score(tr: pd.DataFrame) -> float:
    if tr is None or tr.empty or len(tr) < 3 or not {"x", "y"}.issubset(tr.columns):
        return 0.0
    xy = tr[["x", "y"]].to_numpy(dtype=float)
    a, b = xy[0], xy[-1]
    chord = b - a
    clen = float(np.linalg.norm(chord))
    if clen <= 0:
        return 0.0
    # perpendicular distance of all truth points to the endpoint chord
    rel = xy - a
    perp = np.abs(np.cross(chord, rel) / clen)
    return float(np.nanmax(perp) / max(clen, 1.0))


def _add_scale_bar(ax, xlim, ylim):
    width = abs(xlim[1] - xlim[0])
    if width <= 0:
        return
    target = width * 0.22
    if target <= 0:
        return
    pow10 = 10 ** np.floor(np.log10(target))
    candidates = np.array([1, 2, 5, 10]) * pow10
    length = float(candidates[np.argmin(np.abs(candidates - target))])
    x0 = xlim[0] + width * 0.08
    y0 = ylim[0] + abs(ylim[1] - ylim[0]) * 0.08
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=2.0, solid_capstyle="butt", zorder=10)
    label = f"{int(round(length))} m" if length < 1000 else f"{length/1000:.1f} km"
    ax.text(x0 + length / 2, y0 + abs(ylim[1] - ylim[0]) * 0.025, label, ha="center", va="bottom", fontsize=8)


def _add_north_arrow(ax):
    ax.annotate("N", xy=(0.92, 0.90), xytext=(0.92, 0.76), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))


def _linear_path_from_truth_times(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.sort_values("point_order").copy()
    if len(tr) < 2:
        return tr
    n = len(tr)
    t = np.linspace(0.0, 1.0, n)
    x0, y0 = float(tr.iloc[0]["x"]), float(tr.iloc[0]["y"])
    x1, y1 = float(tr.iloc[-1]["x"]), float(tr.iloc[-1]["y"])
    return pd.DataFrame({"x": x0 + t * (x1 - x0), "y": y0 + t * (y1 - y0), "point_order": tr["point_order"].values})


def _select_one_example_per_interval(mdf: pd.DataFrame, task_points: pd.DataFrame, selected_paths: pd.DataFrame, intervals=INTERVAL_PANEL_ORDER) -> list[str]:
    chosen = []
    task_ids_in_points = set(task_points["task_uid"].astype(str))
    task_ids_in_paths = set(selected_paths["task_uid"].astype(str))
    mdf = mdf.copy()
    if "ade_ratio_to_linear" in mdf.columns:
        mdf["_gain_score"] = 1.0 - pd.to_numeric(mdf["ade_ratio_to_linear"], errors="coerce")
    elif "ADE" in mdf.columns:
        # fallback: lower ADE is better
        mdf["_gain_score"] = -pd.to_numeric(mdf["ADE"], errors="coerce")
    else:
        mdf["_gain_score"] = 0.0
    if {"path_length_m", "truth_path_length_m"}.issubset(mdf.columns):
        mdf["_path_ratio_to_truth"] = pd.to_numeric(mdf["path_length_m"], errors="coerce") / pd.to_numeric(mdf["truth_path_length_m"], errors="coerce").replace(0, np.nan)
        mdf["_shape_ok"] = mdf["_path_ratio_to_truth"].between(0.45, 1.75).fillna(False)
    else:
        mdf["_shape_ok"] = True
    curvature = {}
    for uid, g in task_points.groupby(task_points["task_uid"].astype(str), sort=False):
        curvature[str(uid)] = _truth_curvature_score(g.sort_values("point_order"))
    mdf["_truth_curvature"] = mdf["task_uid"].astype(str).map(curvature).fillna(0.0)
    # Prefer visible turning and positive/near-positive gain, but avoid extreme outliers.
    for pattern, _label in intervals:
        sub = mdf[mdf.get("setting_name", "").astype(str).str.contains(pattern, na=False)].copy() if "setting_name" in mdf.columns else pd.DataFrame()
        if sub.empty:
            continue
        sub = sub[sub["task_uid"].astype(str).isin(task_ids_in_points & task_ids_in_paths)]
        if sub.empty:
            continue
        # Rank by: proposed not worse, visible truth curvature, selected path not degenerate, then moderate-good ADE ratio.
        sub["_beats_or_ties"] = pd.to_numeric(sub["_gain_score"], errors="coerce").fillna(-999).ge(-0.02)
        sub = sub.sort_values(["_beats_or_ties", "_shape_ok", "_truth_curvature", "_gain_score"], ascending=[False, False, False, False])
        uid = str(sub.iloc[0]["task_uid"])
        if uid not in chosen:
            chosen.append(uid)
    return chosen


def plot_reconstruction_examples(
    metrics: pd.DataFrame,
    task_points: pd.DataFrame,
    selected_paths: pd.DataFrame,
    output_dir: str | Path,
    method: str | None = None,
    n_examples: int = 4,
    save_pdf: bool = True,
):
    """Figure 2: one example per reconstruction interval with map annotations."""
    required = {"task_uid", "point_order", "x", "y"}
    if task_points is None or selected_paths is None or task_points.empty or selected_paths.empty:
        print("Example path figure skipped: missing task_points or selected_paths.", flush=True)
        return None
    if not required.issubset(task_points.columns) or not {"task_uid", "x", "y"}.issubset(selected_paths.columns):
        print("Example path figure skipped: path tables do not have required columns.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics_for_choice, paper_methods, chosen_method = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics_for_choice["method"].astype(str)) else (paper_methods[-1] if paper_methods else None)
        mdf = metrics_for_choice[metrics_for_choice["method"].astype(str).eq(str(method))].copy()
        original_method = chosen_method
    else:
        mdf = metrics[metrics["method"].astype(str).eq(str(method))].copy()
        original_method = method
    if mdf.empty:
        print(f"Example path figure skipped: method {method} not found.", flush=True)
        return None
    chosen = _select_one_example_per_interval(mdf, task_points, selected_paths)[:n_examples]
    if len(chosen) < n_examples:
        remaining = [u for u in mdf.sort_values("ADE")["task_uid"].astype(str).values if u not in chosen]
        for u in remaining:
            if u in set(task_points["task_uid"].astype(str)) and u in set(selected_paths["task_uid"].astype(str)):
                chosen.append(u)
            if len(chosen) >= n_examples:
                break
    if not chosen:
        print("Example path figure skipped: no overlapping task IDs in path tables.", flush=True)
        return None
    n = len(chosen)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 5.7 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, uid in zip(axes, chosen):
        tr = task_points[task_points["task_uid"].astype(str).eq(uid)].sort_values("point_order")
        paths = selected_paths[selected_paths["task_uid"].astype(str).eq(uid)].copy()
        ax.plot(tr["x"], tr["y"], color=C_TRUTH_LINE, linewidth=2.2, label="Ground truth")
        ax.scatter(tr["x"], tr["y"], color=C_TRUTH_PTS, s=8, zorder=3)
        lin = _linear_path_from_truth_times(tr)
        ax.plot(lin["x"], lin["y"], color=C_STRAIGHT, linewidth=1.6, linestyle=":", label="Straight line")
        if len(tr) >= 2:
            anchors = tr.iloc[[0, -1]]
            ax.scatter(anchors["x"], anchors["y"], color=C_ANCHOR, s=48, zorder=4, marker="o", label="Sparse endpoints")
        if not paths.empty:
            # Try original selected method first, then any proposed method, then first path.
            path_subset = pd.DataFrame()
            if "method" in paths.columns and original_method is not None:
                path_subset = paths[paths["method"].astype(str).eq(str(original_method))].copy()
            if path_subset.empty and "method" in paths.columns:
                proposed_methods = paths[paths["method"].astype(str).map(is_proposed_method)].copy()
                if not proposed_methods.empty:
                    # prefer robust global/guarded if present
                    for pm in ["pretrained_motif_robust_global", "pretrained_motif_guarded", "pretrained_motif_top1"]:
                        tmp = proposed_methods[proposed_methods["method"].astype(str).eq(pm)].copy()
                        if not tmp.empty:
                            proposed_methods = tmp
                            break
                    path_subset = proposed_methods
            if path_subset.empty:
                path_subset = paths.copy()
            path_key = next((c for c in ["path_id", "candidate_id", "selection_id", "variant", "method"] if c in path_subset.columns), None)
            if path_key is not None:
                pid = str(path_subset[path_key].dropna().astype(str).iloc[0]) if path_subset[path_key].notna().any() else None
                rp = path_subset[path_subset[path_key].astype(str).eq(pid)].copy() if pid is not None else path_subset.copy()
            else:
                rp = path_subset.copy()
            sort_cols = [c for c in ["point_order", "step_index", "time", "timestamp", "t"] if c in rp.columns]
            if sort_cols:
                rp = rp.sort_values(sort_cols)
            if {"x", "y"}.issubset(rp.columns) and not rp.empty:
                ax.plot(rp["x"], rp["y"], color=method_color(PAPER_PROPOSED_METHOD), linewidth=2.1, linestyle="--", label="Pretrained motif")
        row = mdf[mdf["task_uid"].astype(str).eq(uid)].iloc[0]
        taxon = _display_taxon(row.get("taxon", ""))
        habitat = _display_habitat(row.get("habitat_id", row.get("study_system", "")))
        interval = setting_label(row.get("setting_name", ""))
        gain = row.get("ade_gain_pct_vs_linear", np.nan)
        ade = row.get("ADE", np.nan)
        subtitle = f"ADE={ade:.1f} m" if pd.notna(ade) else ""
        if pd.notna(gain):
            subtitle += f", gain={gain:.1f}%"
        ax.set_title(f"{taxon}, {habitat}, {interval}\n{subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=True, axis="both")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        _add_scale_bar(ax, xlim, ylim)
        _add_north_arrow(ax)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[n:]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=C_TRUTH_LINE, lw=2.2, label="Ground truth"),
        Line2D([0], [0], color=C_STRAIGHT, lw=1.6, linestyle=":", label="Straight line"),
        Line2D([0], [0], color=method_color(PAPER_PROPOSED_METHOD), lw=2.1, linestyle="--", label="Pretrained motif"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ANCHOR, markersize=7, label="Sparse endpoints"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, output_dir, "Fig2_reconstruction_examples", save_pdf=save_pdf)
    return fig


def plot_transfer_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    """Species-separated transfer performance figure using normalized gain when available."""
    relation_candidates = [
        "dominant_source_transfer_relation",
        "top_source_transfer_relation",
        "source_transfer_relation",
        "best_train_transfer_relation",
    ]
    relation_col = next((c for c in relation_candidates if c in metrics.columns), None)
    if relation_col is None:
        print("Transfer performance figure skipped: no transfer-relation column found.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    sub = sub[sub[relation_col].notna()].copy()
    if sub.empty:
        print(f"Transfer performance figure skipped: no rows for {method}.", flush=True)
        return None
    value_col = "ade_gain_pct_vs_linear" if "ade_gain_pct_vs_linear" in sub.columns else ("ade_ratio_to_linear" if "ade_ratio_to_linear" in sub.columns else "ADE")
    ylabel = "ADE gain over linear (%)" if value_col == "ade_gain_pct_vs_linear" else ("ADE / linear" if value_col == "ade_ratio_to_linear" else "ADE (m)")
    species_col = "taxon" if "taxon" in sub.columns else "target_taxon"
    species_order = [x for x in ["puma", "cougar", "bobcat", "tiger", "leopard"] if x in set(sub.get(species_col, pd.Series(dtype=str)).astype(str))]
    species_order += [x for x in sorted(sub.get(species_col, pd.Series(dtype=str)).dropna().astype(str).unique()) if x not in species_order]
    relation_order = [
        "same_species_same_habitat",
        "same_species_different_habitat",
        "different_species_same_habitat",
        "different_species_different_habitat",
        "no_training_source",
    ]
    n = len(species_order)
    if n == 0:
        print("Transfer performance figure skipped: no species/taxon labels.", flush=True)
        return None
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, sp in zip(axes, species_order):
        ss = sub[sub[species_col].astype(str).eq(str(sp))].copy()
        relations = [r for r in relation_order if r in set(ss[relation_col].astype(str))]
        relations += [r for r in sorted(ss[relation_col].dropna().astype(str).unique()) if r not in relations]
        vals = [clip_values_for_display(finite_values(ss[ss[relation_col].astype(str).eq(r)], value_col), (1, 99)) for r in relations]
        if not vals:
            ax.set_axis_off(); continue
        bp = ax.boxplot(vals, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
        for patch in bp["boxes"]:
            patch.set_facecolor(method_color(method))
            patch.set_edgecolor("black")
            patch.set_alpha(0.95)
        ax.set_title(_display_taxon(sp))
        ax.set_xticks(range(1, len(relations) + 1))
        ax.set_xticklabels([_transfer_relation_label(r) for r in relations], rotation=0, ha="center")
        if value_col == "ade_gain_pct_vs_linear":
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
        elif value_col == "ade_ratio_to_linear":
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ymin, ymax = ax.get_ylim()
        for i, r in enumerate(relations, start=1):
            n_tasks = int(ss[ss[relation_col].astype(str).eq(r)]["task_uid"].nunique()) if "task_uid" in ss.columns else len(ss[ss[relation_col].astype(str).eq(r)])
            ax.text(i, ymin + 0.02 * (ymax - ymin), f"n={n_tasks}", ha="center", va="bottom", fontsize=7)
    for ax in axes[n:]:
        ax.set_axis_off()
    title_suffix = "selected-source relation" if relation_col != "best_train_transfer_relation" else "available-training relation"
    fig.suptitle(f"{method_label(method)} performance by species and {title_suffix}", y=1.02, fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, "FigS_transfer_relation_by_species", save_pdf=save_pdf)
    return fig


def plot_supplementary_violin_figures(metrics: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    """Generate pure-violin supplementary counterparts using the main palette."""
    set_paper_style()
    metrics, methods, chosen = prepare_metrics_for_paper_figures(metrics, method_summary=method_summary, proposed_method=proposed_method)
    metrics = add_figure_setting(metrics)
    groups = ordered_figure_settings(metrics)
    figs = []
    # Performance violin: same panels as Fig. 3.
    specs = [
        ("ade_ratio_to_linear", "ADE / linear", "Timestamp error relative to linear"),
        ("rmse_ratio_to_linear", "RMSE / linear", "RMSE relative to linear"),
        ("dtw_ratio_to_linear", "DTW / linear", "Shape similarity relative to linear"),
        ("path_length_ratio_error", "Path-length ratio error", "Path length preservation"),
    ] if "ade_ratio_to_linear" in metrics.columns else [
        ("ADE", "ADE (m)", "Average displacement error"),
        ("RMSE", "Time-indexed RMSE (m)", "Time-indexed RMSE"),
        ("path_length_log_error", "Path-length log error", "Path length preservation"),
        ("directness_error", "Directness-score error", "Directness preservation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.3), sharex=False)
    axes = axes.ravel()
    for ax, (col, ylabel, title) in zip(axes, specs):
        ylim = metric_ylim(metrics, [col], clip_quantiles=(1, 99), include_zero=not col.endswith("_ratio_to_linear"))
        draw_grouped_violin_by_group_method(ax, metrics, col, ylabel=ylabel, title=title, methods=methods, groups=groups, ylim=ylim)
        if col.endswith("_ratio_to_linear"):
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
    fig.legend(handles=legend_handles(methods, include_hatch=False), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, output_dir, "FigS_violin_reconstruction_performance", save_pdf=save_pdf)
    figs.append(fig)
    # RMSE violin counterpart.
    col_spatial = "spatial_RMSE" if "spatial_RMSE" in metrics.columns else "spatial_rmse_m"
    col_time = "RMSE" if "RMSE" in metrics.columns else "time_indexed_rmse_m"
    if col_spatial in metrics.columns and col_time in metrics.columns:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
        ylim = metric_ylim(metrics, [col_spatial, col_time], clip_quantiles=(1, 99), include_zero=True)
        draw_grouped_violin_by_group_method(axes2[0], metrics, col_spatial, ylabel="RMSE (m)", title="Spatial RMSE to truth path", methods=methods, groups=groups, ylim=ylim)
        draw_grouped_violin_by_group_method(axes2[1], metrics, col_time, ylabel=None, title="Time-indexed RMSE", methods=methods, groups=groups, ylim=ylim)
        axes2[1].set_ylabel("")
        fig2.legend(handles=legend_handles(methods, include_hatch=False), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.12), frameon=False)
        fig2.tight_layout(rect=[0, 0.14, 1, 1])
        save_fig(fig2, output_dir, "FigS_violin_spatial_time_indexed_RMSE", save_pdf=save_pdf)
        figs.append(fig2)
    # Environmental violin counterparts.
    puma_env = [
        ("env_abs_error_Dist_housing", "Distance to housing"),
        ("env_abs_error_Dist_stream", "Distance to stream"),
        ("env_abs_error_Elevation", "Elevation"),
        ("env_abs_error_Slope", "Slope"),
    ]
    thailand_env = [
        ("env_abs_error_slope", "Slope"),
        ("env_abs_error_bt", "Banteng"),
        ("env_abs_error_gr", "Gaur"),
        ("env_abs_error_sb", "Sambar"),
    ]
    def env_violin(sub, env_cols, basename, title_prefix):
        valid = [(c, lab) for c, lab in env_cols if c in sub.columns and pd.to_numeric(sub[c], errors="coerce").notna().any()]
        if not valid or sub.empty:
            return None
        sub = add_figure_setting(sub)
        grps = ordered_figure_settings(sub)
        n = len(valid); ncols = min(2, n); nrows = int(np.ceil(n / ncols))
        figv, axv = plt.subplots(nrows, ncols, figsize=(6.4*ncols, 3.8*nrows), squeeze=False)
        axv = axv.ravel()
        for ax, (col, lab) in zip(axv, valid):
            ylim = metric_ylim(sub, [col], clip_quantiles=(1, 99), include_zero=True)
            draw_grouped_violin_by_group_method(ax, sub, col, ylabel="Absolute exposure error" if ax is axv[0] else None, title=lab, methods=methods, groups=grps, ylim=ylim)
        for ax in axv[n:]:
            ax.set_axis_off()
        figv.suptitle(title_prefix, y=1.02, fontsize=11)
        figv.legend(handles=legend_handles(methods, include_hatch=False), loc="lower center", ncol=min(3, len(methods)), bbox_to_anchor=(0.5, -0.02), frameon=False)
        figv.tight_layout(rect=[0, 0.08, 1, 1])
        save_fig(figv, output_dir, basename, save_pdf=save_pdf)
        return figv
    dataset = metrics.get("dataset", pd.Series([""] * len(metrics))).astype(str)
    taxon = metrics.get("taxon", pd.Series([""] * len(metrics))).astype(str)
    figp = env_violin(metrics[dataset.str.contains("SantaCruz_puma", case=False, na=False) | taxon.eq("puma")].copy(), puma_env, "FigS_violin_puma_environmental_exposure", "Puma environmental exposure")
    if figp is not None: figs.append(figp)
    figt = env_violin(metrics[dataset.str.contains("Thailand", case=False, na=False) | taxon.isin(["tiger", "leopard"])].copy(), thailand_env, "FigS_violin_tigerleopard_environmental_exposure", "Tiger/leopard environmental exposure")
    if figt is not None: figs.append(figt)
    return figs

# Extend violin outputs with species-separated transfer violin plot.
def plot_transfer_violin_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    relation_candidates = [
        "dominant_source_transfer_relation",
        "top_source_transfer_relation",
        "source_transfer_relation",
        "best_train_transfer_relation",
    ]
    relation_col = next((c for c in relation_candidates if c in metrics.columns), None)
    if relation_col is None:
        print("Transfer violin figure skipped: no transfer-relation column found.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    sub = sub[sub[relation_col].notna()].copy()
    if sub.empty:
        print(f"Transfer violin figure skipped: no rows for {method}.", flush=True)
        return None
    value_col = "ade_gain_pct_vs_linear" if "ade_gain_pct_vs_linear" in sub.columns else ("ade_ratio_to_linear" if "ade_ratio_to_linear" in sub.columns else "ADE")
    ylabel = "ADE gain over linear (%)" if value_col == "ade_gain_pct_vs_linear" else ("ADE / linear" if value_col == "ade_ratio_to_linear" else "ADE (m)")
    species_col = "taxon" if "taxon" in sub.columns else "target_taxon"
    species_order = [x for x in ["puma", "cougar", "bobcat", "tiger", "leopard"] if x in set(sub.get(species_col, pd.Series(dtype=str)).astype(str))]
    species_order += [x for x in sorted(sub.get(species_col, pd.Series(dtype=str)).dropna().astype(str).unique()) if x not in species_order]
    relation_order = [
        "same_species_same_habitat",
        "same_species_different_habitat",
        "different_species_same_habitat",
        "different_species_different_habitat",
        "no_training_source",
    ]
    n = len(species_order)
    if n == 0:
        print("Transfer violin figure skipped: no species/taxon labels.", flush=True)
        return None
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, sp in zip(axes, species_order):
        ss = sub[sub[species_col].astype(str).eq(str(sp))].copy()
        relations = [r for r in relation_order if r in set(ss[relation_col].astype(str))]
        relations += [r for r in sorted(ss[relation_col].dropna().astype(str).unique()) if r not in relations]
        vals = [clip_values_for_display(finite_values(ss[ss[relation_col].astype(str).eq(r)], value_col), (1, 99)) for r in relations]
        finite = [(v, r) for v, r in zip(vals, relations) if len(v) and np.isfinite(v).any()]
        if not finite:
            ax.set_axis_off(); continue
        vals2, relations2 = zip(*finite)
        vals2 = [v if len(v) >= 2 else np.array([v[0], v[0]]) for v in vals2]
        vp = ax.violinplot(vals2, positions=list(range(1, len(vals2)+1)), widths=0.78, showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(method_color(method))
            body.set_edgecolor("black")
            body.set_alpha(0.80)
            body.set_linewidth(0.6)
        ax.set_title(_display_taxon(sp))
        ax.set_xticks(range(1, len(relations2) + 1))
        ax.set_xticklabels([_transfer_relation_label(r) for r in relations2], rotation=0, ha="center")
        if value_col == "ade_gain_pct_vs_linear":
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
        elif value_col == "ade_ratio_to_linear":
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(f"{method_label(method)} transfer performance by species", y=1.02, fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, "FigS_violin_transfer_relation_by_species", save_pdf=save_pdf)
    return fig

# Wrap the previous supplementary violin generator to also include transfer violin.
_previous_plot_supplementary_violin_figures = plot_supplementary_violin_figures

def plot_supplementary_violin_figures(metrics: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    figs = _previous_plot_supplementary_violin_figures(metrics, output_dir, save_pdf=save_pdf, method_summary=method_summary, proposed_method=proposed_method) or []
    figt = plot_transfer_violin_performance(metrics, output_dir, method=None, save_pdf=save_pdf)
    if figt is not None:
        figs.append(figt)
    return figs


# -----------------------------------------------------------------------------
# V20.4 manuscript plotting overrides
# -----------------------------------------------------------------------------
# These overrides implement two updates requested after V20.3 review:
# 1) transfer figure as a 2x2 relation-by-animal layout;
# 2) Figure 2 with one-day trajectory context around each reconstructed gap.


def _species_order_from_data(data: pd.DataFrame, species_col: str = "taxon") -> list[str]:
    preferred = ["bobcat", "cougar", "puma", "tiger", "leopard"]
    available = list(data.get(species_col, pd.Series(dtype=str)).dropna().astype(str).unique())
    out = [sp for sp in preferred if sp in available]
    out.extend([sp for sp in sorted(available) if sp not in out])
    return out


def _relation_order_for_panels(sub: pd.DataFrame, relation_col: str) -> list[str]:
    preferred = [
        "same_species_same_habitat",
        "same_species_different_habitat",
        "different_species_same_habitat",
        "different_species_different_habitat",
    ]
    available = set(sub[relation_col].dropna().astype(str))
    return [r for r in preferred if r in available]


def _value_label_for_transfer(sub: pd.DataFrame) -> tuple[str, str, float | None]:
    if "ade_gain_pct_vs_linear" in sub.columns:
        return "ade_gain_pct_vs_linear", "ADE gain over linear (%)", 0.0
    if "ade_ratio_to_linear" in sub.columns:
        return "ade_ratio_to_linear", "ADE / linear", 1.0
    return "ADE", "ADE (m)", None


def _draw_transfer_panel(ax, sub: pd.DataFrame, species_order: list[str], relation: str, relation_col: str, value_col: str, ylabel: str, method: str, violin: bool = False):
    positions = []
    values = []
    labels = []
    for i, sp in enumerate(species_order, start=1):
        ss = sub[(sub[relation_col].astype(str).eq(str(relation))) & (sub["_species_col"].astype(str).eq(str(sp)))].copy()
        vals = clip_values_for_display(finite_values(ss, value_col), (1, 99))
        positions.append(i)
        labels.append(_display_taxon(sp))
        values.append(vals)
    finite = [(p, lab, v) for p, lab, v in zip(positions, labels, values) if len(v) and np.isfinite(v).any()]
    if not finite:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    pos2, lab2, vals2 = zip(*finite)
    if violin:
        vv = []
        for v in vals2:
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 1:
                arr = np.array([arr[0], arr[0]])
            vv.append(arr)
        vp = ax.violinplot(vv, positions=list(pos2), widths=0.78, showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(method_color(method))
            body.set_edgecolor("black")
            body.set_alpha(0.80)
            body.set_linewidth(0.6)
    else:
        bp = ax.boxplot(vals2, positions=list(pos2), widths=0.72, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
        for patch in bp["boxes"]:
            patch.set_facecolor(method_color(method))
            patch.set_edgecolor("black")
            patch.set_alpha(0.95)
    ax.set_title(_transfer_relation_label(relation))
    ax.set_xticks(list(pos2))
    ax.set_xticklabels(list(lab2), rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ymin, ymax = ax.get_ylim()
    for p, sp in zip(pos2, lab2):
        ss = sub[(sub[relation_col].astype(str).eq(str(relation))) & (sub["_species_col"].map(_display_taxon).astype(str).eq(str(sp)))]
        n_tasks = int(ss["task_uid"].nunique()) if "task_uid" in ss.columns else len(ss)
        ax.text(p, ymin + 0.02 * (ymax - ymin), f"n={n_tasks}", ha="center", va="bottom", fontsize=7)


def plot_transfer_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    """Main supplementary transfer figure: 2x2 relation panels with animals on x-axis."""
    relation_candidates = [
        "dominant_source_transfer_relation",
        "top_source_transfer_relation",
        "source_transfer_relation",
        "best_train_transfer_relation",
    ]
    relation_col = next((c for c in relation_candidates if c in metrics.columns), None)
    if relation_col is None:
        print("Transfer performance figure skipped: no transfer-relation column found.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    sub = sub[sub[relation_col].notna()].copy()
    if sub.empty:
        print(f"Transfer performance figure skipped: no rows for {method}.", flush=True)
        return None
    species_col = "taxon" if "taxon" in sub.columns else ("target_taxon" if "target_taxon" in sub.columns else None)
    if species_col is None:
        print("Transfer performance figure skipped: no species/taxon labels.", flush=True)
        return None
    sub["_species_col"] = sub[species_col].astype(str)
    species_order = _species_order_from_data(sub, "_species_col")
    relations = _relation_order_for_panels(sub, relation_col)
    if not relations:
        print("Transfer performance figure skipped: none of the four paper transfer relations were found.", flush=True)
        return None
    value_col, ylabel, reference = _value_label_for_transfer(sub)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, relation in zip(axes, relations):
        _draw_transfer_panel(ax, sub, species_order, relation, relation_col, value_col, ylabel, method, violin=False)
        if reference is not None:
            ax.axhline(reference, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
    for ax in axes[len(relations):]:
        ax.set_axis_off()
    fig.suptitle(f"{method_label(method)} transfer performance by support relation", y=1.01, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "FigS_transfer_relation_by_relation", save_pdf=save_pdf)
    return fig


def plot_transfer_violin_performance(metrics: pd.DataFrame, output_dir: str | Path, method: str | None = None, save_pdf: bool = True):
    relation_candidates = [
        "dominant_source_transfer_relation",
        "top_source_transfer_relation",
        "source_transfer_relation",
        "best_train_transfer_relation",
    ]
    relation_col = next((c for c in relation_candidates if c in metrics.columns), None)
    if relation_col is None:
        print("Transfer violin figure skipped: no transfer-relation column found.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics, paper_methods, chosen = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics["method"].astype(str)) else (paper_methods[-1] if paper_methods else str(metrics["method"].iloc[0]))
    sub = metrics[metrics["method"].astype(str).eq(method)].copy()
    sub = sub[sub[relation_col].notna()].copy()
    if sub.empty:
        print(f"Transfer violin figure skipped: no rows for {method}.", flush=True)
        return None
    species_col = "taxon" if "taxon" in sub.columns else ("target_taxon" if "target_taxon" in sub.columns else None)
    if species_col is None:
        print("Transfer violin figure skipped: no species/taxon labels.", flush=True)
        return None
    sub["_species_col"] = sub[species_col].astype(str)
    species_order = _species_order_from_data(sub, "_species_col")
    relations = _relation_order_for_panels(sub, relation_col)
    if not relations:
        print("Transfer violin figure skipped: none of the four paper transfer relations were found.", flush=True)
        return None
    value_col, ylabel, reference = _value_label_for_transfer(sub)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, relation in zip(axes, relations):
        _draw_transfer_panel(ax, sub, species_order, relation, relation_col, value_col, ylabel, method, violin=True)
        if reference is not None:
            ax.axhline(reference, color="#444444", linestyle="--", linewidth=0.8, alpha=0.75)
    for ax in axes[len(relations):]:
        ax.set_axis_off()
    fig.suptitle(f"{method_label(method)} transfer performance by support relation", y=1.01, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "FigS_violin_transfer_relation_by_relation", save_pdf=save_pdf)
    return fig


def _daily_context_subset(tracks: pd.DataFrame | None, row: pd.Series, context_hours: float = 12.0) -> pd.DataFrame:
    if tracks is None or getattr(tracks, "empty", True):
        return pd.DataFrame()
    if "time" not in tracks.columns:
        return pd.DataFrame()
    try:
        start = pd.to_datetime(row.get("start_time"), errors="coerce")
        end = pd.to_datetime(row.get("end_time"), errors="coerce")
    except Exception:
        return pd.DataFrame()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    mid = start + (end - start) / 2
    w0 = mid - pd.Timedelta(hours=float(context_hours))
    w1 = mid + pd.Timedelta(hours=float(context_hours))
    mask = pd.Series(True, index=tracks.index)
    for col in ["dataset", "taxon", "animal_id"]:
        if col in tracks.columns and col in row.index and pd.notna(row.get(col, np.nan)):
            mask &= tracks[col].astype(str).eq(str(row[col]))
    times = pd.to_datetime(tracks["time"], errors="coerce")
    mask &= times.ge(w0) & times.le(w1)
    ctx = tracks.loc[mask].copy()
    if not ctx.empty:
        ctx = ctx.sort_values("time")
    return ctx


def plot_reconstruction_examples(
    metrics: pd.DataFrame,
    task_points: pd.DataFrame,
    selected_paths: pd.DataFrame,
    output_dir: str | Path,
    method: str | None = None,
    n_examples: int = 4,
    save_pdf: bool = True,
    task_table: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    context_hours: float = 12.0,
):
    """Figure 2: one example per interval, shown inside a 24 h trajectory context when available."""
    required = {"task_uid", "point_order", "x", "y"}
    if task_points is None or selected_paths is None or task_points.empty or selected_paths.empty:
        print("Example path figure skipped: missing task_points or selected_paths.", flush=True)
        return None
    if not required.issubset(task_points.columns) or not {"task_uid", "x", "y"}.issubset(selected_paths.columns):
        print("Example path figure skipped: path tables do not have required columns.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics_for_choice, paper_methods, chosen_method = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics_for_choice["method"].astype(str)) else (paper_methods[-1] if paper_methods else None)
        mdf = metrics_for_choice[metrics_for_choice["method"].astype(str).eq(str(method))].copy()
        original_method = chosen_method
    else:
        mdf = metrics[metrics["method"].astype(str).eq(str(method))].copy()
        original_method = method
    if mdf.empty:
        print(f"Example path figure skipped: method {method} not found.", flush=True)
        return None
    chosen = _select_one_example_per_interval(mdf, task_points, selected_paths)[:n_examples]
    if len(chosen) < n_examples:
        remaining = [u for u in mdf.sort_values("ADE")["task_uid"].astype(str).values if u not in chosen]
        for u in remaining:
            if u in set(task_points["task_uid"].astype(str)) and u in set(selected_paths["task_uid"].astype(str)):
                chosen.append(u)
            if len(chosen) >= n_examples:
                break
    if not chosen:
        print("Example path figure skipped: no overlapping task IDs in path tables.", flush=True)
        return None
    n = len(chosen)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1 * ncols, 6.0 * nrows), squeeze=False)
    axes = axes.ravel()
    tt = task_table.copy() if task_table is not None and not getattr(task_table, "empty", True) else pd.DataFrame()
    for ax, uid in zip(axes, chosen):
        tr = task_points[task_points["task_uid"].astype(str).eq(uid)].sort_values("point_order")
        paths = selected_paths[selected_paths["task_uid"].astype(str).eq(uid)].copy()
        row = mdf[mdf["task_uid"].astype(str).eq(uid)].iloc[0]
        trow = tt[tt["task_uid"].astype(str).eq(uid)].iloc[0] if not tt.empty and (tt["task_uid"].astype(str).eq(uid)).any() else row
        ctx = _daily_context_subset(tracks, trow, context_hours=context_hours)
        if not ctx.empty and {"x", "y"}.issubset(ctx.columns):
            ax.plot(ctx["x"], ctx["y"], color="#bfbfbf", linewidth=1.5, alpha=0.85, zorder=1)
            ax.scatter(ctx["x"], ctx["y"], color="#d7d7d7", s=5, alpha=0.75, zorder=1)
        ax.plot(tr["x"], tr["y"], color=C_TRUTH_LINE, linewidth=2.4, label="Gap truth", zorder=4)
        ax.scatter(tr["x"], tr["y"], color=C_TRUTH_PTS, s=9, zorder=5)
        lin = _linear_path_from_truth_times(tr)
        ax.plot(lin["x"], lin["y"], color=C_STRAIGHT, linewidth=1.6, linestyle=":", label="Straight line", zorder=3)
        if len(tr) >= 2:
            anchors = tr.iloc[[0, -1]]
            ax.scatter(anchors["x"], anchors["y"], color=C_ANCHOR, s=52, zorder=6, marker="o", label="Sparse endpoints")
        if not paths.empty:
            path_subset = pd.DataFrame()
            if "method" in paths.columns and original_method is not None:
                path_subset = paths[paths["method"].astype(str).eq(str(original_method))].copy()
            if path_subset.empty and "method" in paths.columns:
                proposed_methods = paths[paths["method"].astype(str).map(is_proposed_method)].copy()
                if not proposed_methods.empty:
                    for pm in ["pretrained_motif_robust_global", "pretrained_motif_guarded", "pretrained_motif_top1"]:
                        tmp = proposed_methods[proposed_methods["method"].astype(str).eq(pm)].copy()
                        if not tmp.empty:
                            proposed_methods = tmp
                            break
                    path_subset = proposed_methods
            if path_subset.empty:
                path_subset = paths.copy()
            path_key = next((c for c in ["path_id", "candidate_id", "selection_id", "variant", "method"] if c in path_subset.columns), None)
            if path_key is not None:
                pid = str(path_subset[path_key].dropna().astype(str).iloc[0]) if path_subset[path_key].notna().any() else None
                rp = path_subset[path_subset[path_key].astype(str).eq(pid)].copy() if pid is not None else path_subset.copy()
            else:
                rp = path_subset.copy()
            sort_cols = [c for c in ["point_order", "step_index", "time", "timestamp", "t"] if c in rp.columns]
            if sort_cols:
                rp = rp.sort_values(sort_cols)
            if {"x", "y"}.issubset(rp.columns) and not rp.empty:
                ax.plot(rp["x"], rp["y"], color=method_color(PAPER_PROPOSED_METHOD), linewidth=2.2, linestyle="--", label="Pretrained motif", zorder=5)
        taxon = _display_taxon(row.get("taxon", ""))
        habitat = _display_habitat(row.get("habitat_id", row.get("study_system", row.get("dataset", ""))))
        interval = setting_label(row.get("setting_name", ""))
        gain = row.get("ade_gain_pct_vs_linear", np.nan)
        ade = row.get("ADE", np.nan)
        subtitle = f"ADE={ade:.1f} m" if pd.notna(ade) else ""
        if pd.notna(gain):
            subtitle += f", gain={gain:.1f}%"
        if not ctx.empty:
            subtitle += " | 24 h context"
        ax.set_title(f"{taxon}, {habitat}, {interval}\n{subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=True, axis="both")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        _add_scale_bar(ax, xlim, ylim)
        _add_north_arrow(ax)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[n:]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color="#bfbfbf", lw=1.5, label="24 h context"),
        Line2D([0], [0], color=C_TRUTH_LINE, lw=2.4, label="Gap truth"),
        Line2D([0], [0], color=C_STRAIGHT, lw=1.6, linestyle=":", label="Straight line"),
        Line2D([0], [0], color=method_color(PAPER_PROPOSED_METHOD), lw=2.2, linestyle="--", label="Pretrained motif"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ANCHOR, markersize=7, label="Sparse endpoints"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, output_dir, "Fig2_reconstruction_examples", save_pdf=save_pdf)
    return fig

# Re-wrap violin exporter so the new transfer violin file is included.
_previous_plot_supplementary_violin_figures_v204 = plot_supplementary_violin_figures

def plot_supplementary_violin_figures(metrics: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    figs = _previous_plot_supplementary_violin_figures_v204(metrics, output_dir, save_pdf=save_pdf, method_summary=method_summary, proposed_method=proposed_method) or []
    # Remove older transfer-by-species figure from the list if already emitted; keep the new relation-panel figure.
    figt = plot_transfer_violin_performance(metrics, output_dir, method=None, save_pdf=save_pdf)
    if figt is not None:
        figs.append(figt)
    return figs

# Final no-duplicate wrapper after V20.4 overrides.
def plot_supplementary_violin_figures(metrics: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True, method_summary: pd.DataFrame | None = None, proposed_method: str | None = None):
    return _previous_plot_supplementary_violin_figures_v204(metrics, output_dir, save_pdf=save_pdf, method_summary=method_summary, proposed_method=proposed_method)


# -----------------------------------------------------------------------------
# V20.5 Figure 2 override: 24-hour truth vs 24-hour simulated trajectory
# -----------------------------------------------------------------------------

def _time_bounds_from_row_and_truth(row: pd.Series, tr: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.to_datetime(row.get("start_time", pd.NaT), errors="coerce")
    end = pd.to_datetime(row.get("end_time", pd.NaT), errors="coerce")
    if (pd.isna(start) or pd.isna(end)) and "time" in tr.columns:
        tt = pd.to_datetime(tr["time"], errors="coerce")
        if tt.notna().any():
            start = tt.min() if pd.isna(start) else start
            end = tt.max() if pd.isna(end) else end
    if pd.isna(start) or pd.isna(end):
        return None, None
    return start, end


def _path_with_gap_times(path: pd.DataFrame, tr: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Attach timestamps to a gap path so it can be inserted into a 24 h track."""
    out = path.copy()
    time_col = next((c for c in ["time", "timestamp", "datetime"] if c in out.columns), None)
    if time_col is not None:
        out["time"] = pd.to_datetime(out[time_col], errors="coerce")
    elif "time" in tr.columns and len(tr) == len(out):
        out["time"] = pd.to_datetime(tr["time"].values, errors="coerce")
    else:
        n = max(len(out), 2)
        out["time"] = pd.date_range(start=start, end=end, periods=n)[:len(out)]
    out = out[["time", "x", "y"]].copy()
    out = out[pd.to_datetime(out["time"], errors="coerce").notna()]
    return out.sort_values("time")


def _linear_gap_path_for_times(tr: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    lin = _linear_path_from_truth_times(tr)
    if "time" in tr.columns and len(lin) == len(tr):
        lin["time"] = pd.to_datetime(tr["time"].values, errors="coerce")
    else:
        lin["time"] = pd.date_range(start=start, end=end, periods=max(len(lin), 2))[:len(lin)]
    return lin[["time", "x", "y"]].copy().sort_values("time")


def _insert_gap_into_context(ctx: pd.DataFrame, gap: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return a 24 h track where only the hidden gap is replaced by `gap`."""
    if ctx is None or ctx.empty or "time" not in ctx.columns:
        return gap.copy()
    base = ctx.copy()
    base["time"] = pd.to_datetime(base["time"], errors="coerce")
    base = base[base["time"].notna()]
    if base.empty:
        return gap.copy()
    outside = base[(base["time"] < start) | (base["time"] > end)][["time", "x", "y"]].copy()
    out = pd.concat([outside, gap[["time", "x", "y"]]], ignore_index=True)
    return out.sort_values("time")


def _terrain_raster_candidates(project_root: str | Path | None, row: pd.Series) -> list[Path]:
    if project_root is None:
        project_root = Path.cwd()
    root = Path(project_root)
    raster_root = root / "data" / "rasters"
    if not raster_root.exists():
        return []
    dataset_bits = " ".join(str(row.get(c, "")) for c in ["dataset", "taxon", "habitat_id", "study_system"]).lower()
    files = sorted(list(raster_root.rglob("*.tif")) + list(raster_root.rglob("*.tiff")))
    if not files:
        return []

    def dataset_match(path: Path) -> int:
        s = str(path).lower()
        score = 0
        if "thailand" in dataset_bits and ("thai" in s or "wefcom" in s):
            score += 5
        if ("puma" in dataset_bits or "santacruz" in dataset_bits or "santa" in dataset_bits) and ("puma" in s or "santa" in s):
            score += 5
        if ("olympic" in dataset_bits or "cougar" in dataset_bits or "bobcat" in dataset_bits) and ("olympic" in s or "cougar" in s or "bobcat" in s):
            score += 5
        return score

    def terrain_score(path: Path) -> tuple[int, str]:
        s = path.name.lower()
        score = dataset_match(path)
        if any(k in s for k in ["hillshade", "shade"]):
            score += 8
        if any(k in s for k in ["elev", "dem", "altitude"]):
            score += 7
        if "slope" in s:
            score += 5
        if any(k in s for k in ["bt", "gr", "sb", "banteng", "gaur", "sambar", "housing", "stream"]):
            score -= 3
        return (score, str(path))
    return sorted(files, key=terrain_score, reverse=True)[:8]


def _add_grey_terrain_basemap(ax, row: pd.Series, project_root: str | Path | None = None, alpha: float = 0.26):
    """Add a quiet grayscale local raster backdrop if one is available."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except Exception:
        return False
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xmin, xmax = sorted([float(xlim[0]), float(xlim[1])])
    ymin, ymax = sorted([float(ylim[0]), float(ylim[1])])
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmax <= xmin or ymax <= ymin:
        return False
    padx = (xmax - xmin) * 0.08
    pady = (ymax - ymin) * 0.08
    bounds = (xmin - padx, ymin - pady, xmax + padx, ymax + pady)
    for rp in _terrain_raster_candidates(project_root, row):
        try:
            with rasterio.open(rp) as src:
                rb = src.bounds
                if bounds[2] < rb.left or bounds[0] > rb.right or bounds[3] < rb.bottom or bounds[1] > rb.top:
                    continue
                window = from_bounds(*bounds, transform=src.transform)
                arr = src.read(1, window=window, boundless=True, masked=True)
                if arr.size == 0:
                    continue
                arr = np.asarray(arr.filled(np.nan), dtype=float)
                if not np.isfinite(arr).any():
                    continue
                lo, hi = np.nanpercentile(arr, [2, 98])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    continue
                arr = np.clip((arr - lo) / (hi - lo), 0, 1)
                ax.imshow(arr, extent=bounds, cmap="Greys", origin="upper", alpha=alpha, zorder=0, interpolation="bilinear")
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                return True
        except Exception:
            continue
    return False


def plot_reconstruction_examples(
    metrics: pd.DataFrame,
    task_points: pd.DataFrame,
    selected_paths: pd.DataFrame,
    output_dir: str | Path,
    method: str | None = None,
    n_examples: int = 4,
    save_pdf: bool = True,
    task_table: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    context_hours: float = 12.0,
    project_root: str | Path | None = None,
    add_terrain: bool = True,
):
    """Figure 2: 24 h ground truth vs 24 h simulated track for four interval examples."""
    required = {"task_uid", "point_order", "x", "y"}
    if task_points is None or selected_paths is None or task_points.empty or selected_paths.empty:
        print("Example path figure skipped: missing task_points or selected_paths.", flush=True)
        return None
    if not required.issubset(task_points.columns) or not {"task_uid", "x", "y"}.issubset(selected_paths.columns):
        print("Example path figure skipped: path tables do not have required columns.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics_for_choice, paper_methods, chosen_method = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics_for_choice["method"].astype(str)) else (paper_methods[-1] if paper_methods else None)
        mdf = metrics_for_choice[metrics_for_choice["method"].astype(str).eq(str(method))].copy()
        original_method = chosen_method
    else:
        mdf = metrics[metrics["method"].astype(str).eq(str(method))].copy()
        original_method = method
    if mdf.empty:
        print(f"Example path figure skipped: method {method} not found.", flush=True)
        return None

    chosen = _select_one_example_per_interval(mdf, task_points, selected_paths)[:n_examples]
    if len(chosen) < n_examples:
        remaining = [u for u in mdf.sort_values("ADE")["task_uid"].astype(str).values if u not in chosen]
        for u in remaining:
            if u in set(task_points["task_uid"].astype(str)) and u in set(selected_paths["task_uid"].astype(str)):
                chosen.append(u)
            if len(chosen) >= n_examples:
                break
    if not chosen:
        print("Example path figure skipped: no overlapping task IDs in path tables.", flush=True)
        return None

    n = len(chosen); ncols = min(2, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.3 * ncols, 6.2 * nrows), squeeze=False)
    axes = axes.ravel()
    tt = task_table.copy() if task_table is not None and not getattr(task_table, "empty", True) else pd.DataFrame()

    for ax, uid in zip(axes, chosen):
        tr = task_points[task_points["task_uid"].astype(str).eq(uid)].sort_values("point_order").copy()
        paths = selected_paths[selected_paths["task_uid"].astype(str).eq(uid)].copy()
        row = mdf[mdf["task_uid"].astype(str).eq(uid)].iloc[0]
        trow = tt[tt["task_uid"].astype(str).eq(uid)].iloc[0] if not tt.empty and (tt["task_uid"].astype(str).eq(uid)).any() else row
        start, end = _time_bounds_from_row_and_truth(trow, tr)
        ctx = _daily_context_subset(tracks, trow, context_hours=context_hours) if start is not None else pd.DataFrame()
        if ctx.empty:
            ctx = tr.copy()
            if "time" not in ctx.columns and start is not None and end is not None:
                ctx["time"] = pd.date_range(start=start, end=end, periods=max(len(ctx), 2))[:len(ctx)]
        if "time" in ctx.columns:
            ctx["time"] = pd.to_datetime(ctx["time"], errors="coerce")
            ctx = ctx[ctx["time"].notna()].sort_values("time")

        gap_sim = pd.DataFrame()
        if not paths.empty and start is not None and end is not None:
            path_subset = pd.DataFrame()
            if "method" in paths.columns and original_method is not None:
                path_subset = paths[paths["method"].astype(str).eq(str(original_method))].copy()
            if path_subset.empty and "method" in paths.columns:
                proposed_methods = paths[paths["method"].astype(str).map(is_proposed_method)].copy()
                if not proposed_methods.empty:
                    for pm in ["pretrained_motif_robust_global", "pretrained_motif_guarded", "pretrained_motif_top1"]:
                        tmp = proposed_methods[proposed_methods["method"].astype(str).eq(pm)].copy()
                        if not tmp.empty:
                            proposed_methods = tmp; break
                    path_subset = proposed_methods
            if path_subset.empty:
                path_subset = paths.copy()
            path_key = next((c for c in ["path_id", "candidate_id", "selection_id", "variant", "method"] if c in path_subset.columns), None)
            if path_key is not None:
                pid = str(path_subset[path_key].dropna().astype(str).iloc[0]) if path_subset[path_key].notna().any() else None
                rp = path_subset[path_subset[path_key].astype(str).eq(pid)].copy() if pid is not None else path_subset.copy()
            else:
                rp = path_subset.copy()
            sort_cols = [c for c in ["point_order", "step_index", "time", "timestamp", "t"] if c in rp.columns]
            if sort_cols:
                rp = rp.sort_values(sort_cols)
            if {"x", "y"}.issubset(rp.columns) and not rp.empty:
                gap_sim = _path_with_gap_times(rp, tr, start, end)

        if start is not None and end is not None:
            gap_linear = _linear_gap_path_for_times(tr, start, end)
            sim_24h = _insert_gap_into_context(ctx, gap_sim, start, end) if not gap_sim.empty else pd.DataFrame()
            linear_24h = _insert_gap_into_context(ctx, gap_linear, start, end)
        else:
            sim_24h = gap_sim
            linear_24h = _linear_path_from_truth_times(tr)

        extent_frames = [df for df in [ctx, sim_24h, linear_24h, tr] if df is not None and not df.empty and {"x", "y"}.issubset(df.columns)]
        if extent_frames:
            allxy = pd.concat([df[["x", "y"]] for df in extent_frames], ignore_index=True)
            xmin, xmax = np.nanmin(allxy["x"]), np.nanmax(allxy["x"])
            ymin, ymax = np.nanmin(allxy["y"]), np.nanmax(allxy["y"])
            padx = max((xmax - xmin) * 0.10, 20.0); pady = max((ymax - ymin) * 0.10, 20.0)
            ax.set_xlim(xmin - padx, xmax + padx); ax.set_ylim(ymin - pady, ymax + pady)
        if add_terrain:
            _add_grey_terrain_basemap(ax, row, project_root=project_root, alpha=0.26)

        if not ctx.empty and {"x", "y"}.issubset(ctx.columns):
            ax.plot(ctx["x"], ctx["y"], color=C_TRUTH_LINE, linewidth=2.1, label="24 h ground truth", zorder=3)
        if linear_24h is not None and not getattr(linear_24h, "empty", True) and {"x", "y"}.issubset(linear_24h.columns):
            ax.plot(linear_24h["x"], linear_24h["y"], color=C_STRAIGHT, linewidth=1.7, linestyle=":", label="24 h straight-line baseline", zorder=4)
        if sim_24h is not None and not getattr(sim_24h, "empty", True) and {"x", "y"}.issubset(sim_24h.columns):
            ax.plot(sim_24h["x"], sim_24h["y"], color=method_color(PAPER_PROPOSED_METHOD), linewidth=2.2, linestyle="--", label="24 h simulated", zorder=5)
        ax.plot(tr["x"], tr["y"], color=C_TRUTH_LINE, linewidth=3.0, alpha=0.92, zorder=6)
        if not gap_sim.empty:
            ax.plot(gap_sim["x"], gap_sim["y"], color=method_color(PAPER_PROPOSED_METHOD), linewidth=2.6, linestyle="--", zorder=7)
        if len(tr) >= 2:
            anchors = tr.iloc[[0, -1]]
            ax.scatter(anchors["x"], anchors["y"], color=C_ANCHOR, s=58, zorder=8, marker="o", label="Sparse endpoints")

        taxon = _display_taxon(row.get("taxon", ""))
        habitat = _display_habitat(row.get("habitat_id", row.get("study_system", row.get("dataset", ""))))
        interval = setting_label(row.get("setting_name", ""))
        gain = row.get("ade_gain_pct_vs_linear", np.nan); ade = row.get("ADE", np.nan)
        subtitle = f"ADE={ade:.1f} m" if pd.notna(ade) else ""
        if pd.notna(gain): subtitle += f", gain={gain:.1f}%"
        subtitle += " | 24 h window"
        ax.set_title(f"{taxon}, {habitat}, {interval}\n{subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=True, axis="both")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        _add_scale_bar(ax, xlim, ylim); _add_north_arrow(ax)
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

    for ax in axes[n:]: ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=C_TRUTH_LINE, lw=2.1, label="24 h ground truth"),
        Line2D([0], [0], color=C_STRAIGHT, lw=1.7, linestyle=":", label="24 h straight-line baseline"),
        Line2D([0], [0], color=method_color(PAPER_PROPOSED_METHOD), lw=2.2, linestyle="--", label="24 h simulated"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ANCHOR, markersize=7, label="Sparse endpoints"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, output_dir, "Fig2_reconstruction_examples", save_pdf=save_pdf)
    return fig

# -----------------------------------------------------------------------------
# V20.6 Figure 2 override: complete one-day reconstruction, not a single-gap context
# -----------------------------------------------------------------------------

def _nearest_track_row(track: pd.DataFrame, target_time: pd.Timestamp, tolerance: pd.Timedelta | None = None):
    if track is None or track.empty or "time" not in track.columns:
        return None
    tt = pd.to_datetime(track["time"], errors="coerce")
    if tt.isna().all():
        return None
    idx = (tt - target_time).abs().idxmin()
    if tolerance is not None and abs(tt.loc[idx] - target_time) > tolerance:
        return None
    return track.loc[idx]


def _coarse_fixes_for_day(day_truth: pd.DataFrame, coarse_dt_min: float, fine_dt_min: float | None = None) -> pd.DataFrame:
    """Choose coarse observed fixes across the one-day truth window."""
    if day_truth is None or day_truth.empty or "time" not in day_truth.columns:
        return pd.DataFrame()
    d = day_truth.copy().sort_values("time")
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d[d["time"].notna()]
    if len(d) < 2:
        return d
    start = d["time"].iloc[0]
    end = d["time"].iloc[-1]
    coarse_td = pd.Timedelta(minutes=float(coarse_dt_min))
    tol_min = float(fine_dt_min) / 2 if fine_dt_min and np.isfinite(fine_dt_min) else max(1.0, float(coarse_dt_min) * 0.05)
    tol = pd.Timedelta(minutes=tol_min + 1e-6)
    targets = []
    t = start
    while t <= end + pd.Timedelta(seconds=1):
        targets.append(t)
        t = t + coarse_td
    rows = []
    used_idx = set()
    for t in targets:
        row = _nearest_track_row(d, t, tolerance=tol)
        if row is not None and row.name not in used_idx:
            rows.append(row)
            used_idx.add(row.name)
    # Ensure terminal point is included so the day trajectory reaches the end.
    if d.index[-1] not in used_idx:
        rows.append(d.iloc[-1])
    out = pd.DataFrame(rows).sort_values("time") if rows else d.iloc[[0, -1]].copy()
    return out.drop_duplicates(subset=["time"])


def _find_task_for_segment(task_table: pd.DataFrame, row: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp, fine_dt_min: float | None = None) -> str | None:
    if task_table is None or task_table.empty or "task_uid" not in task_table.columns:
        return None
    tt = task_table.copy()
    for col in ["dataset", "taxon", "animal_id", "setting_name"]:
        if col in tt.columns and col in row.index and pd.notna(row.get(col, np.nan)):
            tt = tt[tt[col].astype(str).eq(str(row[col]))]
    if tt.empty or "start_time" not in tt.columns or "end_time" not in tt.columns:
        return None
    st = pd.to_datetime(tt["start_time"], errors="coerce")
    et = pd.to_datetime(tt["end_time"], errors="coerce")
    tol_min = float(fine_dt_min) if fine_dt_min and np.isfinite(fine_dt_min) else 1.0
    tol = pd.Timedelta(minutes=max(tol_min, 1.0) + 1e-6)
    hit = tt[(st - seg_start).abs().le(tol) & (et - seg_end).abs().le(tol)]
    if hit.empty:
        return None
    # Prefer validation/test selected tasks if split exists, otherwise first.
    if "split" in hit.columns:
        pref = hit[hit["split"].astype(str).isin(["test", "validation"])]
        if not pref.empty:
            hit = pref
    return str(hit.iloc[0]["task_uid"])


def _selected_gap_path_for_task_uid(selected_paths: pd.DataFrame, task_uid: str, preferred_method: str | None, tr_segment: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> pd.DataFrame:
    if selected_paths is None or selected_paths.empty or task_uid is None:
        return pd.DataFrame()
    paths = selected_paths[selected_paths["task_uid"].astype(str).eq(str(task_uid))].copy()
    if paths.empty or not {"x", "y"}.issubset(paths.columns):
        return pd.DataFrame()
    path_subset = pd.DataFrame()
    if "method" in paths.columns and preferred_method is not None:
        path_subset = paths[paths["method"].astype(str).eq(str(preferred_method))].copy()
    if path_subset.empty and "method" in paths.columns:
        proposed_methods = paths[paths["method"].astype(str).map(is_proposed_method)].copy()
        if not proposed_methods.empty:
            for pm in ["pretrained_motif_adaptive_selector", "pretrained_motif_robust_global", "pretrained_motif_guarded", "pretrained_motif_top1"]:
                tmp = proposed_methods[proposed_methods["method"].astype(str).eq(pm)].copy()
                if not tmp.empty:
                    proposed_methods = tmp
                    break
            path_subset = proposed_methods
    if path_subset.empty:
        path_subset = paths.copy()
    path_key = next((c for c in ["path_id", "candidate_id", "selection_id", "variant", "method"] if c in path_subset.columns), None)
    if path_key is not None:
        pid = str(path_subset[path_key].dropna().astype(str).iloc[0]) if path_subset[path_key].notna().any() else None
        rp = path_subset[path_subset[path_key].astype(str).eq(pid)].copy() if pid is not None else path_subset.copy()
    else:
        rp = path_subset.copy()
    sort_cols = [c for c in ["point_order", "step_index", "time", "timestamp", "t"] if c in rp.columns]
    if sort_cols:
        rp = rp.sort_values(sort_cols)
    return _path_with_gap_times(rp, tr_segment, seg_start, seg_end)


def _linear_segment_between_rows(a: pd.Series, b: pd.Series, tr_segment: pd.DataFrame) -> pd.DataFrame:
    if tr_segment is not None and not tr_segment.empty and "time" in tr_segment.columns:
        times = pd.to_datetime(tr_segment["time"], errors="coerce")
        times = times[times.notna()].sort_values()
    else:
        times = pd.Series(pd.date_range(a["time"], b["time"], periods=2))
    n = max(len(times), 2)
    t = np.linspace(0.0, 1.0, n)
    x = float(a["x"]) + t * (float(b["x"]) - float(a["x"]))
    y = float(a["y"]) + t * (float(b["y"]) - float(a["y"]))
    return pd.DataFrame({"time": list(times)[:n], "x": x[:n], "y": y[:n]})


def _compose_one_day_reconstruction(day_truth: pd.DataFrame, row: pd.Series, task_table: pd.DataFrame, selected_paths: pd.DataFrame, preferred_method: str | None):
    """Compose a full-day reconstructed trajectory from all available gaps.

    Every consecutive pair of coarse observations in the 24 h window is replaced
    by the selected simulated gap if available; otherwise the segment uses a
    straight-line fallback. This visualizes a complete one-day reconstruction,
    not merely a single selected segment.
    """
    if day_truth is None or day_truth.empty or "time" not in day_truth.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = day_truth.copy().sort_values("time")
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d[d["time"].notna()]
    if len(d) < 2:
        return d, d, d
    coarse = float(row.get("coarse_dt_min", row.get("coarse", 60)))
    fine = row.get("fine_dt_min", np.nan)
    try:
        fine = float(fine)
    except Exception:
        fine = np.nan
    coarse_pts = _coarse_fixes_for_day(d, coarse, fine)
    if len(coarse_pts) < 2:
        return d, d, coarse_pts
    sim_segments = []
    lin_segments = []
    for i in range(len(coarse_pts) - 1):
        a = coarse_pts.iloc[i]
        b = coarse_pts.iloc[i + 1]
        seg_start = pd.to_datetime(a["time"])
        seg_end = pd.to_datetime(b["time"])
        tr_seg = d[d["time"].between(seg_start, seg_end)].copy()
        if tr_seg.empty:
            tr_seg = pd.DataFrame({"time": [seg_start, seg_end], "x": [a["x"], b["x"]], "y": [a["y"], b["y"]]})
        linear_seg = _linear_segment_between_rows(a, b, tr_seg)
        lin_segments.append(linear_seg.iloc[0 if i == 0 else 1:])
        uid = _find_task_for_segment(task_table, row, seg_start, seg_end, fine_dt_min=fine)
        sim_seg = _selected_gap_path_for_task_uid(selected_paths, uid, preferred_method, tr_seg, seg_start, seg_end) if uid is not None else pd.DataFrame()
        if sim_seg.empty:
            sim_seg = linear_seg.copy()
            sim_seg["segment_source"] = "linear_fallback"
        else:
            sim_seg["segment_source"] = "selected_simulation"
        sim_segments.append(sim_seg.iloc[0 if i == 0 else 1:])
    sim_day = pd.concat(sim_segments, ignore_index=True) if sim_segments else pd.DataFrame()
    lin_day = pd.concat(lin_segments, ignore_index=True) if lin_segments else pd.DataFrame()
    if not sim_day.empty:
        sim_day = sim_day.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    if not lin_day.empty:
        lin_day = lin_day.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    return sim_day, lin_day, coarse_pts


def plot_reconstruction_examples(
    metrics: pd.DataFrame,
    task_points: pd.DataFrame,
    selected_paths: pd.DataFrame,
    output_dir: str | Path,
    method: str | None = None,
    n_examples: int = 4,
    save_pdf: bool = True,
    task_table: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    context_hours: float = 12.0,
    project_root: str | Path | None = None,
    add_terrain: bool = True,
):
    """Figure 2: complete one-day ground truth vs one-day reconstructed trajectories."""
    required = {"task_uid", "point_order", "x", "y"}
    if task_points is None or selected_paths is None or task_points.empty or selected_paths.empty:
        print("Example path figure skipped: missing task_points or selected_paths.", flush=True)
        return None
    if not required.issubset(task_points.columns) or not {"task_uid", "x", "y"}.issubset(selected_paths.columns):
        print("Example path figure skipped: path tables do not have required columns.", flush=True)
        return None
    set_paper_style()
    if method is None:
        metrics_for_choice, paper_methods, chosen_method = prepare_metrics_for_paper_figures(metrics)
        method = PAPER_PROPOSED_METHOD if PAPER_PROPOSED_METHOD in set(metrics_for_choice["method"].astype(str)) else (paper_methods[-1] if paper_methods else None)
        mdf = metrics_for_choice[metrics_for_choice["method"].astype(str).eq(str(method))].copy()
        preferred_method = chosen_method or method
    else:
        mdf = metrics[metrics["method"].astype(str).eq(str(method))].copy()
        preferred_method = method
    if mdf.empty:
        print(f"Example path figure skipped: method {method} not found.", flush=True)
        return None
    # Prefer adaptive selector if it exists in selected path exports.
    if "method" in selected_paths.columns and "pretrained_motif_adaptive_selector" in set(selected_paths["method"].astype(str)):
        preferred_method = "pretrained_motif_adaptive_selector"

    chosen = _select_one_example_per_interval(mdf, task_points, selected_paths)[:n_examples]
    if len(chosen) < n_examples:
        remaining = [u for u in mdf.sort_values("ADE")["task_uid"].astype(str).values if u not in chosen]
        for u in remaining:
            if u in set(task_points["task_uid"].astype(str)):
                chosen.append(u)
            if len(chosen) >= n_examples:
                break
    if not chosen:
        print("Example path figure skipped: no overlapping task IDs in path tables.", flush=True)
        return None

    n = len(chosen)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 6.25 * nrows), squeeze=False)
    axes = axes.ravel()
    tt = task_table.copy() if task_table is not None and not getattr(task_table, "empty", True) else pd.DataFrame()

    for ax, uid in zip(axes, chosen):
        tr_gap = task_points[task_points["task_uid"].astype(str).eq(uid)].sort_values("point_order").copy()
        row = mdf[mdf["task_uid"].astype(str).eq(uid)].iloc[0]
        trow = tt[tt["task_uid"].astype(str).eq(uid)].iloc[0] if not tt.empty and (tt["task_uid"].astype(str).eq(uid)).any() else row
        gap_start, gap_end = _time_bounds_from_row_and_truth(trow, tr_gap)
        day_truth = _daily_context_subset(tracks, trow, context_hours=context_hours) if gap_start is not None else pd.DataFrame()
        if day_truth.empty:
            day_truth = tr_gap.copy()
            if "time" not in day_truth.columns and gap_start is not None and gap_end is not None:
                day_truth["time"] = pd.date_range(start=gap_start, end=gap_end, periods=max(len(day_truth), 2))[:len(day_truth)]
        if "time" in day_truth.columns:
            day_truth["time"] = pd.to_datetime(day_truth["time"], errors="coerce")
            day_truth = day_truth[day_truth["time"].notna()].sort_values("time")

        sim_day, linear_day, coarse_pts = _compose_one_day_reconstruction(day_truth, trow, tt, selected_paths, preferred_method)
        extent_frames = [df for df in [day_truth, sim_day, linear_day, coarse_pts] if df is not None and not df.empty and {"x", "y"}.issubset(df.columns)]
        if extent_frames:
            allxy = pd.concat([df[["x", "y"]] for df in extent_frames], ignore_index=True)
            xmin, xmax = np.nanmin(allxy["x"]), np.nanmax(allxy["x"])
            ymin, ymax = np.nanmin(allxy["y"]), np.nanmax(allxy["y"])
            padx = max((xmax - xmin) * 0.10, 20.0)
            pady = max((ymax - ymin) * 0.10, 20.0)
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymin - pady, ymax + pady)
        if add_terrain:
            _add_grey_terrain_basemap(ax, row, project_root=project_root, alpha=0.26)

        if not day_truth.empty and {"x", "y"}.issubset(day_truth.columns):
            ax.plot(day_truth["x"], day_truth["y"], color=C_TRUTH_LINE, linewidth=2.2, label="One-day ground truth", zorder=3)
        if linear_day is not None and not getattr(linear_day, "empty", True) and {"x", "y"}.issubset(linear_day.columns):
            ax.plot(linear_day["x"], linear_day["y"], color=C_STRAIGHT, linewidth=1.6, linestyle=":", label="One-day straight-line reconstruction", zorder=4)
        if sim_day is not None and not getattr(sim_day, "empty", True) and {"x", "y"}.issubset(sim_day.columns):
            ax.plot(sim_day["x"], sim_day["y"], color=method_color(PAPER_PROPOSED_METHOD), linewidth=2.2, linestyle="--", label="One-day reconstructed trajectory", zorder=5)
        if coarse_pts is not None and not coarse_pts.empty and {"x", "y"}.issubset(coarse_pts.columns):
            ax.scatter(coarse_pts["x"], coarse_pts["y"], color=C_ANCHOR, s=34, zorder=7, marker="o", label="Coarse observations")
        # Mark the anchor gap so readers can see which interval type selected this one-day example.
        if not tr_gap.empty and {"x", "y"}.issubset(tr_gap.columns):
            ax.plot(tr_gap["x"], tr_gap["y"], color=C_TRUTH_LINE, linewidth=3.0, alpha=0.70, zorder=6)

        taxon = _display_taxon(row.get("taxon", ""))
        habitat = _display_habitat(row.get("habitat_id", row.get("study_system", row.get("dataset", ""))))
        interval = setting_label(row.get("setting_name", ""))
        gain = row.get("ade_gain_pct_vs_linear", np.nan)
        ade = row.get("ADE", np.nan)
        subtitle = f"anchor-gap ADE={ade:.1f} m" if pd.notna(ade) else ""
        if pd.notna(gain):
            subtitle += f", gain={gain:.1f}%"
        subtitle += " | full one-day reconstruction"
        ax.set_title(f"{taxon}, {habitat}, {interval}\n{subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=True, axis="both")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        _add_scale_bar(ax, xlim, ylim)
        _add_north_arrow(ax)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[n:]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=C_TRUTH_LINE, lw=2.2, label="One-day ground truth"),
        Line2D([0], [0], color=C_STRAIGHT, lw=1.6, linestyle=":", label="One-day straight-line reconstruction"),
        Line2D([0], [0], color=method_color(PAPER_PROPOSED_METHOD), lw=2.2, linestyle="--", label="One-day reconstructed trajectory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ANCHOR, markersize=6, label="Coarse observations"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, output_dir, "Fig2_reconstruction_examples", save_pdf=save_pdf)
    return fig


# V20.7 plotting labels for residual-flow/adaptive selector v2.
METHOD_LABELS.update({
    "pretrained_motif_adaptive_selector_v2": "Pretrained motif",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf": "Pretrained motif",
})
METHOD_COLORS.update({
    "pretrained_motif_adaptive_selector_v2": C_ENV,
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf": C_ENV,
})
METHOD_HATCHES.update({
    "pretrained_motif_adaptive_selector_v2": "",
    "pretrained_motif_residual_flow_weighted_cost_rank_K20_b0.35_lamconf": "",
})
