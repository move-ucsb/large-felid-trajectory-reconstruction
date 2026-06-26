"""Deployable context features for motif retrieval and candidate scoring.

The functions in this module use only information that is available from the
coarse task context or from the training motif itself: endpoints/timestamps,
individual metadata, environmental values already attached to observed fixes,
and path-shape summaries from training motifs.  They deliberately avoid using
the hidden high-resolution geometry of the *target* task for selection.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def _safe_timestamp(x) -> pd.Timestamp | None:
    try:
        if x is None or pd.isna(x):
            return None
        return pd.Timestamp(x)
    except Exception:
        return None


def _sin_cos(value: float, period: float) -> tuple[float, float]:
    if not np.isfinite(value) or period <= 0:
        return 0.0, 1.0
    a = 2.0 * math.pi * (float(value) % period) / period
    return float(math.sin(a)), float(math.cos(a))


def _season_name(month: int) -> str:
    # Northern-hemisphere meteorological seasons. For Thailand this is only a
    # coarse cyclic label; monthly sin/cos remains the numeric feature.
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _diel_bin(hour: float) -> str:
    # Simple deployable diel categories. Dawn/dusk are broad because GPS fixes
    # are coarser than exact local solar time and datasets span different sites.
    h = float(hour) % 24.0
    if 5.0 <= h < 8.0:
        return "dawn"
    if 17.0 <= h < 20.0:
        return "dusk"
    if 8.0 <= h < 17.0:
        return "day"
    return "night"


def temporal_context_features(task) -> dict[str, float | str]:
    """Return timestamp, diel, and seasonal context features for a task."""
    st = _safe_timestamp(getattr(task, "start_time", None))
    et = _safe_timestamp(getattr(task, "end_time", None))
    if st is None:
        return {
            "start_hour_sin": 0.0, "start_hour_cos": 1.0,
            "mid_hour_sin": 0.0, "mid_hour_cos": 1.0,
            "month_sin": 0.0, "month_cos": 1.0,
            "dayofyear_sin": 0.0, "dayofyear_cos": 1.0,
            "diel_bin": "unknown", "season": "unknown",
        }
    if et is None:
        mid = st
    else:
        mid = st + (et - st) / 2
    start_hour = st.hour + st.minute / 60.0 + st.second / 3600.0
    mid_hour = mid.hour + mid.minute / 60.0 + mid.second / 3600.0
    hs, hc = _sin_cos(start_hour, 24.0)
    ms, mc = _sin_cos(mid_hour, 24.0)
    mons, monc = _sin_cos(float(st.month - 1), 12.0)
    doy = float(st.dayofyear - 1) if hasattr(st, "dayofyear") else 0.0
    ds, dc = _sin_cos(doy, 365.25)
    return {
        "start_hour_sin": hs, "start_hour_cos": hc,
        "mid_hour_sin": ms, "mid_hour_cos": mc,
        "month_sin": mons, "month_cos": monc,
        "dayofyear_sin": ds, "dayofyear_cos": dc,
        "diel_bin": _diel_bin(mid_hour),
        "season": _season_name(int(st.month)),
    }


def demographic_context_features(task) -> dict[str, float | str]:
    """Return sex/age context fields in both categorical and numeric forms."""
    sex = str(getattr(task, "sex", "unknown") or "unknown").lower()
    age = str(getattr(task, "age_class", "unknown") or "unknown").lower()
    return {
        "context_sex": sex,
        "context_age_class": age,
        "context_sex_female": 1.0 if sex in {"female", "f", "ft", "fl"} else 0.0,
        "context_sex_male": 1.0 if sex in {"male", "m", "mt", "ml"} else 0.0,
        "context_age_adult": 1.0 if "adult" in age and "sub" not in age else 0.0,
        "context_age_subadult": 1.0 if "sub" in age else 0.0,
    }


def env_endpoint_context(task) -> dict[str, float]:
    """Endpoint-only environmental context for a task.

    ``task.truth_env`` may contain full high-resolution environmental values for
    evaluation tasks, but this function intentionally uses only the first and
    last values. Those correspond to the coarse endpoints and are deployable.
    """
    out: dict[str, float] = {}
    truth_env = getattr(task, "truth_env", {}) or {}
    for name, vals in truth_env.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue
        start = float(arr[0]) if np.isfinite(arr[0]) else np.nan
        end = float(arr[-1]) if np.isfinite(arr[-1]) else np.nan
        if not np.isfinite(start) and not np.isfinite(end):
            continue
        endpoint_mean = float(np.nanmean([start, end]))
        endpoint_delta = float(end - start) if np.isfinite(start) and np.isfinite(end) else np.nan
        key = str(name)
        out[f"env_{key}_start"] = start
        out[f"env_{key}_end"] = end
        out[f"env_{key}_endpoint_mean"] = endpoint_mean
        out[f"env_{key}_endpoint_delta"] = endpoint_delta
    return out


def task_context_features(task) -> dict[str, float | str]:
    """All deployable context features for a reconstruction task."""
    out: dict[str, float | str] = {}
    out.update(temporal_context_features(task))
    out.update(demographic_context_features(task))
    out.update(env_endpoint_context(task))
    return out


def signed_lateral_mean(path, start_xy=None, end_xy=None) -> float:
    """Mean signed lateral deviation in endpoint-normalized coordinates."""
    from .geometry import endpoint_basis, as_xy
    xy = as_xy(path)
    if len(xy) == 0:
        return 0.0
    if start_xy is None:
        start_xy = xy[0]
    if end_xy is None:
        end_xy = xy[-1]
    start, u, v, d = endpoint_basis(start_xy, end_xy)
    lateral = (xy - start[None, :]) @ v / max(d, 1.0)
    if len(lateral) <= 2:
        vals = lateral
    else:
        vals = lateral[1:-1]
    return float(np.nanmean(vals)) if len(vals) else 0.0


def motif_shape_features(task, frame: np.ndarray | None = None) -> dict[str, float]:
    """Shape summaries from a training truth path.

    These become source-motif context used later to decide whether a target task
    should borrow a left/right detour or stay near the endpoint chord.
    """
    from .geometry import path_length, directness, path_ratio
    xy = getattr(task, "truth_xy", None)
    out = {
        "motif_path_ratio": np.nan,
        "motif_directness": np.nan,
        "motif_lateral_mean_norm": 0.0,
        "motif_lateral_abs_mean_norm": 0.0,
        "motif_lateral_abs_max_norm": 0.0,
        "motif_lateral_sign": 0.0,
    }
    if xy is None or len(xy) < 2:
        return out
    try:
        out["motif_path_ratio"] = float(path_ratio(xy))
        out["motif_directness"] = float(directness(xy))
    except Exception:
        pass
    if frame is None:
        from .geometry import normalize_to_endpoint_frame
        frame = normalize_to_endpoint_frame(xy)
    y = np.asarray(frame, dtype=float)[:, 1]
    internal = y[1:-1] if len(y) > 2 else y
    if len(internal):
        mean_lat = float(np.nanmean(internal))
        out["motif_lateral_mean_norm"] = mean_lat
        out["motif_lateral_abs_mean_norm"] = float(np.nanmean(np.abs(internal)))
        out["motif_lateral_abs_max_norm"] = float(np.nanmax(np.abs(internal)))
        out["motif_lateral_sign"] = float(np.sign(mean_lat)) if abs(mean_lat) >= 1e-4 else 0.0
    return out


def preferred_lateral_sign_from_task(task) -> float:
    """Infer a weak left/right preference from adjacent coarse headings.

    This is not a hard constraint. It only says whether the incoming and/or
    outgoing coarse motion suggests that the hidden path bends consistently to
    one side of the start--end chord.
    """
    try:
        start = np.asarray(task.start_xy, dtype=float)
        end = np.asarray(task.end_xy, dtype=float)
        base = end - start
        d = float(np.linalg.norm(base))
        if d <= 1e-9:
            return 0.0
        u = base / d
        v = np.array([-u[1], u[0]])
        signs = []
        prev_xy = getattr(task, "prev_xy", None)
        next_xy = getattr(task, "next_xy", None)
        if prev_xy is not None:
            incoming = start - np.asarray(prev_xy, dtype=float)
            if np.linalg.norm(incoming) > 1e-9:
                signs.append(float(np.sign(np.dot(incoming, v))))
        if next_xy is not None:
            outgoing = np.asarray(next_xy, dtype=float) - end
            if np.linalg.norm(outgoing) > 1e-9:
                signs.append(float(np.sign(np.dot(outgoing, v))))
        signs = [s for s in signs if s != 0]
        if not signs:
            return 0.0
        s = float(np.sign(np.nanmean(signs)))
        return s if np.isfinite(s) else 0.0
    except Exception:
        return 0.0


def build_context_scales(motif_table: pd.DataFrame) -> dict[str, float]:
    """Robust per-column scales for environmental context matching."""
    scales: dict[str, float] = {}
    for c in motif_table.columns:
        if not (str(c).startswith("env_") and (str(c).endswith("_endpoint_mean") or str(c).endswith("_endpoint_delta"))):
            continue
        vals = pd.to_numeric(motif_table[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 5:
            continue
        med = float(vals.median())
        mad = float(np.nanmedian(np.abs(vals.to_numpy(dtype=float) - med)))
        iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
        scale = max(mad * 1.4826, iqr / 1.349 if iqr > 0 else 0.0, 1e-6)
        if np.isfinite(scale) and scale > 0:
            scales[c] = scale
    return scales


def _circular_sincos_distance(a_sin, a_cos, b_sin, b_cos) -> float:
    try:
        return float(np.sqrt((float(a_sin) - float(b_sin)) ** 2 + (float(a_cos) - float(b_cos)) ** 2) / 2.0)
    except Exception:
        return 0.0


def context_match_costs(task, motif_row: Mapping, scales: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return deployable context mismatch costs for a source motif."""
    scales = scales or {}
    target = task_context_features(task)
    temporal = 0.0
    temporal += _circular_sincos_distance(target.get("mid_hour_sin", 0.0), target.get("mid_hour_cos", 1.0), motif_row.get("mid_hour_sin", 0.0), motif_row.get("mid_hour_cos", 1.0))
    temporal += 0.75 * _circular_sincos_distance(target.get("month_sin", 0.0), target.get("month_cos", 1.0), motif_row.get("month_sin", 0.0), motif_row.get("month_cos", 1.0))
    temporal += 0.25 * (0.0 if str(target.get("diel_bin", "unknown")) == str(motif_row.get("diel_bin", "unknown")) else 1.0)

    demo = 0.0
    tsex = str(target.get("context_sex", "unknown"))
    ssex = str(motif_row.get("context_sex", motif_row.get("sex", "unknown"))).lower()
    if tsex not in {"unknown", "nan", ""} and ssex not in {"unknown", "nan", ""} and tsex != ssex:
        demo += 1.0
    tage = str(target.get("context_age_class", "unknown"))
    sage = str(motif_row.get("context_age_class", motif_row.get("age_class", "unknown"))).lower()
    if tage not in {"unknown", "nan", ""} and sage not in {"unknown", "nan", ""} and tage != sage:
        demo += 0.6

    env_terms = []
    for c, tv in target.items():
        c = str(c)
        if not (c.startswith("env_") and (c.endswith("_endpoint_mean") or c.endswith("_endpoint_delta"))):
            continue
        if c not in motif_row:
            continue
        try:
            a = float(tv)
            b = float(motif_row.get(c, np.nan))
        except Exception:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        scale = float(scales.get(c, np.nan))
        if not np.isfinite(scale) or scale <= 0:
            scale = max(abs(a), abs(b), 1.0)
        env_terms.append(min(abs(a - b) / max(scale, 1e-6), 5.0))
    env = float(np.nanmean(env_terms)) if env_terms else 0.0
    return {
        "context_temporal_cost": float(temporal),
        "context_demographic_cost": float(demo),
        "context_environment_cost": float(env),
        "context_n_environment_matches": float(len(env_terms)),
        "context_total_cost": float(temporal + demo + env),
    }
