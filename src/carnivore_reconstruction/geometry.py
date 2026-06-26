"""Geometry, baselines, path normalization, and evaluation metrics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EPS = 1e-9


def as_xy(array_like) -> np.ndarray:
    arr = np.asarray(array_like, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected an array with shape (n_points, 2).")
    return arr


def path_length(xy) -> float:
    xy = as_xy(xy)
    if len(xy) < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def displacement(xy) -> float:
    xy = as_xy(xy)
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(xy[-1] - xy[0]))


def directness(xy) -> float:
    length = path_length(xy)
    return float(displacement(xy) / length) if length > EPS else np.nan


def path_ratio(xy) -> float:
    disp = displacement(xy)
    length = path_length(xy)
    return float(length / max(disp, EPS))


def ade(recon, truth) -> float:
    recon = as_xy(recon)
    truth = as_xy(truth)
    n = min(len(recon), len(truth))
    if n == 0:
        return np.nan
    return float(np.nanmean(np.linalg.norm(recon[:n] - truth[:n], axis=1)))


def rmse(recon, truth) -> float:
    recon = as_xy(recon)
    truth = as_xy(truth)
    n = min(len(recon), len(truth))
    if n == 0:
        return np.nan
    d = np.linalg.norm(recon[:n] - truth[:n], axis=1)
    return float(np.sqrt(np.nanmean(d * d)))


def point_to_segments_distance(point, path) -> float:
    p = np.asarray(point, dtype=float)
    path = as_xy(path)
    if len(path) == 0:
        return np.nan
    if len(path) == 1:
        return float(np.linalg.norm(p - path[0]))
    a = path[:-1]
    b = path[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom <= EPS, EPS, denom)
    t = np.sum((p - a) * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.nanmin(np.linalg.norm(p - proj, axis=1)))


def spatial_rmse(recon, truth) -> float:
    recon = as_xy(recon)
    truth = as_xy(truth)
    if len(recon) == 0 or len(truth) == 0:
        return np.nan
    d = np.array([point_to_segments_distance(p, truth) for p in recon], dtype=float)
    return float(np.sqrt(np.nanmean(d * d)))


def discrete_frechet(P, Q) -> float:
    P = as_xy(P)
    Q = as_xy(Q)
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return np.nan
    ca = np.full((n, m), np.nan, dtype=float)
    for i in range(n):
        for j in range(m):
            d = float(np.linalg.norm(P[i] - Q[j]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[-1, -1])


def discrete_dtw(P, Q, normalize: bool = True) -> float:
    """Discrete dynamic time warping distance between two 2D paths.

    The returned value is the cumulative Euclidean distance along the optimal
    warping path divided by the number of matched pairs when ``normalize`` is
    True.  This makes the scale comparable to ADE/RMSE in meters while allowing
    flexible temporal alignment.
    """
    P = as_xy(P)
    Q = as_xy(Q)
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return np.nan
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    steps = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    steps[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(P[i - 1] - Q[j - 1]))
            prev = [(cost[i - 1, j], steps[i - 1, j]), (cost[i, j - 1], steps[i, j - 1]), (cost[i - 1, j - 1], steps[i - 1, j - 1])]
            k = int(np.argmin([x[0] for x in prev]))
            cost[i, j] = d + prev[k][0]
            steps[i, j] = 1.0 + prev[k][1]
    if not np.isfinite(cost[n, m]):
        return np.nan
    if normalize:
        denom = max(float(steps[n, m]), 1.0)
        return float(cost[n, m] / denom)
    return float(cost[n, m])


def linear_path(start_xy, end_xy, n_points: int) -> np.ndarray:
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    if n_points <= 1:
        return start.reshape(1, 2)
    t = np.linspace(0.0, 1.0, int(n_points))[:, None]
    return start[None, :] * (1.0 - t) + end[None, :] * t


def hermite_path(start_xy, end_xy, n_points: int, prev_xy=None, next_xy=None, tension: float = 0.35) -> np.ndarray:
    """Cubic Hermite interpolation using optional incoming/outgoing headings.

    This is a fast kinematic fallback, not a learned candidate.
    """
    p0 = np.asarray(start_xy, dtype=float)
    p1 = np.asarray(end_xy, dtype=float)
    base = p1 - p0
    if prev_xy is None:
        m0 = base
    else:
        m0 = p0 - np.asarray(prev_xy, dtype=float)
        if np.linalg.norm(m0) <= EPS:
            m0 = base
    if next_xy is None:
        m1 = base
    else:
        m1 = np.asarray(next_xy, dtype=float) - p1
        if np.linalg.norm(m1) <= EPS:
            m1 = base
    m0 = tension * m0
    m1 = tension * m1
    t = np.linspace(0.0, 1.0, int(n_points))
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00[:, None] * p0 + h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1


def brownian_bridge_path(start_xy, end_xy, n_points: int, scale: float = 0.15, seed: int | None = None) -> np.ndarray:
    """Endpoint-conditioned Brownian bridge baseline."""
    rng = np.random.default_rng(seed)
    base = linear_path(start_xy, end_xy, n_points)
    if n_points <= 2:
        return base
    disp = max(float(np.linalg.norm(base[-1] - base[0])), 1.0)
    noise = rng.normal(0.0, scale * disp, size=base.shape)
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    bridge = noise - (1 - t) * noise[0] - t * noise[-1]
    return base + bridge


def rtg_bridge_path(start_xy, end_xy, n_points: int, lateral_scale: float = 0.15) -> np.ndarray:
    """Simple time-geographic curved bridge baseline."""
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    base = linear_path(start, end, n_points)
    vec = end - start
    d = float(np.linalg.norm(vec))
    if d <= EPS or n_points <= 2:
        return base
    u = vec / d
    v = np.array([-u[1], u[0]])
    t = np.linspace(0.0, 1.0, n_points)
    curve = np.sin(np.pi * t)[:, None] * v[None, :] * (lateral_scale * d)
    return base + curve


def endpoint_basis(start_xy, end_xy) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    vec = end - start
    d = float(np.linalg.norm(vec))
    if d <= EPS:
        u = np.array([1.0, 0.0])
    else:
        u = vec / d
    v = np.array([-u[1], u[0]])
    return start, u, v, d


def normalize_to_endpoint_frame(path) -> np.ndarray:
    """Represent a path in endpoint-relative coordinates.

    Returns columns ``x_norm`` and ``y_norm``.  ``x_norm`` is progress from start
    to end in endpoint units; ``y_norm`` is signed lateral deviation divided by
    endpoint displacement.
    """
    xy = as_xy(path)
    start, u, v, d = endpoint_basis(xy[0], xy[-1])
    rel = xy - start[None, :]
    scale = max(d, 1.0)
    x = rel @ u / scale
    y = rel @ v / scale
    return np.column_stack([x, y])


def denormalize_from_endpoint_frame(frame, start_xy, end_xy, lateral_scale: float = 1.0, mirror: bool = False) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    start, u, v, d = endpoint_basis(start_xy, end_xy)
    scale = max(d, 1.0)
    x = frame[:, 0]
    y = frame[:, 1] * float(lateral_scale) * (-1.0 if mirror else 1.0)
    return start[None, :] + (x[:, None] * scale * u[None, :]) + (y[:, None] * scale * v[None, :])


def resample_path_by_index(path, n_points: int) -> np.ndarray:
    """Resample a path to a target number of points by normalized index."""
    xy = as_xy(path)
    n_points = int(n_points)
    if len(xy) == n_points:
        return xy.copy()
    if len(xy) == 1:
        return np.repeat(xy, n_points, axis=0)
    old = np.linspace(0.0, 1.0, len(xy))
    new = np.linspace(0.0, 1.0, n_points)
    x = np.interp(new, old, xy[:, 0])
    y = np.interp(new, old, xy[:, 1])
    return np.column_stack([x, y])


def signed_lateral_stats(path, start_xy=None, end_xy=None) -> dict[str, float]:
    xy = as_xy(path)
    if start_xy is None:
        start_xy = xy[0]
    if end_xy is None:
        end_xy = xy[-1]
    start, u, v, d = endpoint_basis(start_xy, end_xy)
    rel = xy - start[None, :]
    scale = max(d, 1.0)
    progress = rel @ u / scale
    lateral = rel @ v / scale
    diffs = np.diff(progress) if len(progress) > 1 else np.array([0.0])
    internal = lateral[1:-1] if len(lateral) > 2 else lateral
    mean_lateral = float(np.nanmean(internal)) if len(internal) else 0.0
    return {
        "max_abs_lateral_norm": float(np.nanmax(np.abs(lateral))) if len(lateral) else 0.0,
        "mean_abs_lateral_norm": float(np.nanmean(np.abs(lateral))) if len(lateral) else 0.0,
        "mean_lateral_norm": mean_lateral,
        "lateral_sign": float(np.sign(mean_lateral)) if abs(mean_lateral) >= 1e-4 else 0.0,
        "backtrack_fraction": float(np.mean(diffs < -1e-6)) if len(diffs) else 0.0,
        "progress_start": float(progress[0]) if len(progress) else np.nan,
        "progress_end": float(progress[-1]) if len(progress) else np.nan,
    }


def path_step_stats(path) -> dict[str, float]:
    xy = as_xy(path)
    if len(xy) < 2:
        steps = np.array([0.0])
    else:
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "step_mean": float(np.nanmean(steps)),
        "step_median": float(np.nanmedian(steps)),
        "step_q90": float(np.nanquantile(steps, 0.90)),
        "step_max": float(np.nanmax(steps)),
    }
