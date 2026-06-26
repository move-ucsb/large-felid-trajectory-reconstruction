
"""Latent direct-generation trajectory super-resolution methods.

This module adds optional training-stage direct generators to the
publication package.  The implementations are intentionally lightweight and use
scikit-learn only, so they should run in the same environment as the existing
trajectory reconstruction code.

Active variants
---------------
a: conditional diffusion bridge, implemented as a denoising residual bridge.
b: conditional flow matching, implemented as a learned residual vector field.
c: CVAE-style conditional latent residual decoder using PCA latent codes.
d: goal-conditioned autoregressive movement-action policy.
e: implicit continuous trajectory super-resolution f(condition, tau)->residual.

Each variant learns from paired low-resolution/high-resolution training tasks
during `MotifReconstructionModel.fit` and then adds direct-generated paths during
validation/test. These are retained only as candidate sources for the balanced Top-10 diagnostic.
"""
from __future__ import annotations

from typing import Sequence
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from .tasks import ReconstructionTask
from .timing import status
from .utils import stable_hash01


latent_ACTIVE_VARIANT = "c"
latent_PREFIX_MAP = {
    "a": "direct_sr_conditional_diffusion_bridge",
    "b": "direct_sr_conditional_flow_matching",
    "c": "direct_sr_conditional_latent_decoder",
    "d": "direct_sr_goal_conditioned_action_policy",
    "e": "direct_sr_implicit_continuous_function",
}


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _linear_path(task: ReconstructionTask) -> np.ndarray:
    n = int(task.n_points)
    return np.column_stack([
        np.linspace(float(task.start_xy[0]), float(task.end_xy[0]), n),
        np.linspace(float(task.start_xy[1]), float(task.end_xy[1]), n),
    ])


def _unit_features(vec) -> tuple[float, float, float]:
    if vec is None:
        return 0.0, 0.0, 0.0
    v = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-9:
        return 0.0, 0.0, 0.0
    return float(v[0] / norm), float(v[1] / norm), norm


def _condition(task: ReconstructionTask) -> np.ndarray:
    start = np.asarray(task.start_xy, dtype=float)
    end = np.asarray(task.end_xy, dtype=float)
    disp_vec = end - start
    disp = float(np.linalg.norm(disp_vec))
    dur = max(_safe_float(task.coarse_dt_min, 0.0), 1e-6)
    base_step = disp / max(int(task.n_points) - 1, 1)
    dx, dy, _ = _unit_features(disp_vec)
    prev_vec = None if task.prev_xy is None else (start - np.asarray(task.prev_xy, dtype=float))
    next_vec = None if task.next_xy is None else (np.asarray(task.next_xy, dtype=float) - end)
    pc, ps, pn = _unit_features(prev_vec)
    nc, ns, nn = _unit_features(next_vec)
    cat = [
        stable_hash01(str(getattr(task, "dataset", "unknown"))),
        stable_hash01(str(getattr(task, "taxon", "unknown"))),
        stable_hash01(str(getattr(task, "setting_name", "unknown"))),
        stable_hash01(str(getattr(task, "species_group", "unknown"))),
        stable_hash01(str(getattr(task, "genus_group", "unknown"))),
        stable_hash01(str(getattr(task, "transfer_unit", "unknown"))),
        stable_hash01(str(getattr(task, "habitat_id", "unknown"))),
    ]
    return np.asarray([
        float(task.n_points),
        _safe_float(task.coarse_dt_min),
        _safe_float(task.fine_dt_min),
        disp,
        np.log1p(max(disp, 0.0)),
        base_step,
        disp / dur,
        dx, dy,
        pc, ps, np.log1p(max(pn, 0.0)), float(task.prev_xy is not None),
        nc, ns, np.log1p(max(nn, 0.0)), float(task.next_xy is not None),
        *cat,
    ], dtype=float)


def _truth_residual(task: ReconstructionTask) -> np.ndarray | None:
    if task.truth_xy is None:
        return None
    truth = np.asarray(task.truth_xy, dtype=float)
    if truth.ndim != 2 or truth.shape[0] != int(task.n_points):
        return None
    base = _linear_path(task)
    r = truth[:, :2] - base[:, :2]
    r[0, :] = 0.0
    r[-1, :] = 0.0
    return r.reshape(-1)


def _resid_to_path(task: ReconstructionTask, resid, shrink: float = 1.0) -> np.ndarray:
    base = _linear_path(task)
    r = np.asarray(resid, dtype=float).reshape((-1, 2))
    if len(r) != len(base):
        rr = np.zeros_like(base)
        m = min(len(base), len(r))
        rr[:m] = r[:m]
        r = rr
    xy = base.copy()
    xy += float(shrink) * r
    xy[0] = base[0]
    xy[-1] = base[-1]
    xy[~np.isfinite(xy)] = base[~np.isfinite(xy)]
    return xy


def _by_n_arrays(tasks: Sequence[ReconstructionTask]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    groups: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for task in tasks:
        y = _truth_residual(task)
        if y is None:
            continue
        groups.setdefault(int(task.n_points), []).append((_condition(task), y))
    out = {}
    for n, rows in groups.items():
        X = np.vstack([r[0] for r in rows])
        Y = np.vstack([r[1] for r in rows])
        out[int(n)] = (X, Y)
    return out


def _residual_scale(Y: np.ndarray) -> float:
    vals = np.asarray(Y, dtype=float)
    s = float(np.nanmedian(np.nanstd(vals, axis=0)))
    if not np.isfinite(s) or s <= 1e-6:
        s = float(np.nanmedian(np.abs(vals)))
    if not np.isfinite(s) or s <= 1e-6:
        s = 1.0
    return s


def _fit_denoiser(X: np.ndarray, Y: np.ndarray, random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state)
    scale = _residual_scale(Y)
    xs, ys = [], []
    noise_levels = [1.00, 0.70, 0.45, 0.25, 0.10]
    for sigma in noise_levels:
        repeats = 2 if len(X) < 120 else 1
        for _ in range(repeats):
            noise = rng.normal(0.0, sigma * scale, size=Y.shape)
            noisy = Y + noise
            xs.append(np.hstack([X, noisy, np.full((len(X), 1), sigma)]))
            ys.append(Y)
    Xtrain = np.vstack(xs)
    Ytrain = np.vstack(ys)
    model = ExtraTreesRegressor(
        n_estimators=220, max_depth=12, min_samples_leaf=2,
        random_state=random_state, n_jobs=1
    )
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(Xtrain, Ytrain)
    return {"kind": "conditional_denoising_bridge", "model": pipe, "scale": float(scale), "n_train": int(len(X)), "dim": int(Y.shape[1])}


def _fit_flow(X: np.ndarray, Y: np.ndarray, random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state + 31)
    scale = _residual_scale(Y)
    xs, ys = [], []
    n_repeats = 4 if len(X) < 150 else 2
    for _ in range(n_repeats):
        z0 = rng.normal(0.0, scale, size=Y.shape)
        t = rng.uniform(0.05, 0.95, size=(len(X), 1))
        xt = (1.0 - t) * z0 + t * Y
        vel = Y - z0
        xs.append(np.hstack([X, xt, t]))
        ys.append(vel)
    Xtrain = np.vstack(xs)
    Ytrain = np.vstack(ys)
    model = ExtraTreesRegressor(
        n_estimators=220, max_depth=12, min_samples_leaf=2,
        random_state=random_state, n_jobs=1
    )
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(Xtrain, Ytrain)
    return {"kind": "conditional_flow_matching", "model": pipe, "scale": float(scale), "n_train": int(len(X)), "dim": int(Y.shape[1])}


def _fit_latent_decoder(X: np.ndarray, Y: np.ndarray, random_state: int = 42) -> dict:
    n_latent = int(max(2, min(8, len(X) - 1, Y.shape[1] // 4 if Y.shape[1] >= 8 else 2)))
    if len(X) <= 3:
        return {}
    pca = PCA(n_components=n_latent, random_state=random_state).fit(Y)
    Z = pca.transform(Y)
    decoder = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    decoder.fit(np.hstack([X, Z]), Y)
    # Also fit a condition-only fallback for z=0.
    fallback = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    fallback.fit(X, Y)
    return {
        "kind": "conditional_latent_residual_decoder",
        "pca": pca,
        "decoder": decoder,
        "fallback": fallback,
        "latent_mean": np.asarray(np.nanmean(Z, axis=0), dtype=float),
        "latent_std": np.asarray(np.nanstd(Z, axis=0) + 1e-6, dtype=float),
        "n_train": int(len(X)),
        "dim": int(Y.shape[1]),
        "n_latent": int(n_latent),
    }


def _fit_action_policy(tasks: Sequence[ReconstructionTask], random_state: int = 42) -> dict:
    rows_x, rows_y = [], []
    group_counts = {}
    for task in tasks:
        if task.truth_xy is None:
            continue
        truth = np.asarray(task.truth_xy, dtype=float)
        if truth.ndim != 2 or len(truth) != int(task.n_points) or len(truth) < 3:
            continue
        base = _linear_path(task)
        resid = truth[:, :2] - base
        cond = _condition(task)
        n = int(task.n_points)
        group_counts[n] = group_counts.get(n, 0) + 1
        for j in range(1, n - 1):
            tau = j / max(n - 1, 1)
            prev_r = resid[j - 1]
            rem = 1.0 - tau
            rows_x.append(np.hstack([cond, [tau, rem], prev_r]))
            rows_y.append(resid[j])
    if len(rows_x) < 10:
        return {}
    X = np.vstack(rows_x)
    Y = np.vstack(rows_y)
    pipe = make_pipeline(
        StandardScaler(),
        ExtraTreesRegressor(
            n_estimators=220, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=1
        )
    )
    pipe.fit(X, Y)
    return {
        "kind": "goal_conditioned_autoregressive_action_policy",
        "model": pipe,
        "n_train_rows": int(len(X)),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
    }


def _fit_implicit(tasks: Sequence[ReconstructionTask], random_state: int = 42) -> dict:
    rows_x, rows_y = [], []
    group_counts = {}
    for task in tasks:
        if task.truth_xy is None:
            continue
        truth = np.asarray(task.truth_xy, dtype=float)
        if truth.ndim != 2 or len(truth) != int(task.n_points):
            continue
        base = _linear_path(task)
        resid = truth[:, :2] - base
        cond = _condition(task)
        n = int(task.n_points)
        group_counts[n] = group_counts.get(n, 0) + 1
        for j in range(n):
            tau = j / max(n - 1, 1)
            # Fourier-like time features let the model express curved residuals.
            tf = [
                tau, 1.0 - tau,
                np.sin(np.pi * tau), np.cos(np.pi * tau),
                np.sin(2 * np.pi * tau), np.cos(2 * np.pi * tau),
                float(j == 0), float(j == n - 1),
            ]
            rows_x.append(np.hstack([cond, tf]))
            rows_y.append(resid[j])
    if len(rows_x) < 10:
        return {}
    X = np.vstack(rows_x)
    Y = np.vstack(rows_y)
    pipe = make_pipeline(
        StandardScaler(),
        ExtraTreesRegressor(
            n_estimators=260, max_depth=12, min_samples_leaf=2,
            random_state=random_state, n_jobs=1
        )
    )
    pipe.fit(X, Y)
    return {
        "kind": "implicit_continuous_trajectory_sr",
        "model": pipe,
        "n_train_rows": int(len(X)),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
    }


def build_v31_artifacts(tasks: Sequence[ReconstructionTask], variant: str = latent_ACTIVE_VARIANT, random_state: int = 42) -> dict:
    """Train Latent direct-generation artifacts from paired LR/HR training tasks."""
    variant = str(variant or latent_ACTIVE_VARIANT).lower()
    tasks = list(tasks)
    artifacts = {
        "version": "latent",
        "variant": variant,
        "variant_name": {
            "a": "conditional_diffusion_bridge",
            "b": "conditional_flow_matching",
            "c": "conditional_latent_decoder",
            "d": "goal_conditioned_action_policy",
            "e": "implicit_continuous_trajectory_sr",
        }.get(variant, "unknown"),
        "prefix": latent_PREFIX_MAP.get(variant, "direct_sr_unknown"),
        "n_train_tasks_available": int(len(tasks)),
        "models_by_n": {},
        "global_model": {},
    }
    by_n = _by_n_arrays(tasks)

    if variant == "a":
        for n, (X, Y) in by_n.items():
            if len(X) >= 4:
                artifacts["models_by_n"][int(n)] = _fit_denoiser(X, Y, random_state=random_state + int(n))
        status(f"latenta trained denoising diffusion-bridge models for {len(artifacts['models_by_n'])} n_points group(s).")
        return artifacts

    if variant == "b":
        for n, (X, Y) in by_n.items():
            if len(X) >= 4:
                artifacts["models_by_n"][int(n)] = _fit_flow(X, Y, random_state=random_state + int(n))
        status(f"latentb trained conditional flow-matching models for {len(artifacts['models_by_n'])} n_points group(s).")
        return artifacts

    if variant == "c":
        for n, (X, Y) in by_n.items():
            if len(X) >= 4:
                m = _fit_latent_decoder(X, Y, random_state=random_state + int(n))
                if m:
                    artifacts["models_by_n"][int(n)] = m
        status(f"Trained conditional latent decoders for {len(artifacts['models_by_n'])} n_points group(s).")
        return artifacts

    if variant == "d":
        artifacts["global_model"] = _fit_action_policy(tasks, random_state=random_state)
        status("latentd trained goal-conditioned autoregressive action policy.")
        return artifacts

    if variant == "e":
        artifacts["global_model"] = _fit_implicit(tasks, random_state=random_state)
        status("latente trained implicit continuous trajectory function.")
        return artifacts

    status(f"latent training skipped: unknown variant={variant!r}")
    return artifacts


def _get_group(artifacts: dict, n: int):
    return artifacts.get("models_by_n", {}).get(n) or artifacts.get("models_by_n", {}).get(str(n))


def _sample_noise(dim: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, float(scale), size=int(dim))


def generate_v31_paths_for_task(model, task: ReconstructionTask, n_samples: int = 10) -> dict[str, tuple[np.ndarray, dict]]:
    """Generate latent direct paths for a validation/test task."""
    artifacts = getattr(model, "v31_artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return {}
    variant = str(artifacts.get("variant", latent_ACTIVE_VARIANT)).lower()
    prefix = str(artifacts.get("prefix", latent_PREFIX_MAP.get(variant, "direct_sr_unknown")))
    out: dict[str, tuple[np.ndarray, dict]] = {}
    n = int(task.n_points)
    cond = _condition(task).reshape(1, -1)
    seed0 = int(stable_hash01(str(task.task_uid)) * 1_000_000) + 173

    if variant == "a":
        group = _get_group(artifacts, n)
        if not group:
            return {}
        dim = int(group.get("dim", 2 * n))
        scale = float(group.get("scale", 1.0))
        for i in range(1, int(n_samples) + 1):
            resid = _sample_noise(dim, scale, seed0 + i)
            # Iterative denoising bridge. Explicit endpoint repair happens in _resid_to_path.
            for sigma in [1.00, 0.70, 0.45, 0.25, 0.10, 0.00]:
                xin = np.hstack([cond, resid.reshape(1, -1), [[sigma]]])
                clean = group["model"].predict(xin)[0]
                resid = 0.35 * resid + 0.65 * clean
            method = f"{prefix}_sample{i:02d}"
            out[method] = (_resid_to_path(task, resid, shrink=0.85), {
                "candidate_origin": "direct_conditional_diffusion_bridge",
                "source_method": method,
                "direct_generator_family": "conditional_diffusion_bridge",
                "direct_generator_sample": int(i),
                "n_source_candidates": int(group.get("n_train", 0)),
                "is_direct_generation": 1,
                "is_experimental_v31": 1,
            })
        return out

    if variant == "b":
        group = _get_group(artifacts, n)
        if not group:
            return {}
        dim = int(group.get("dim", 2 * n))
        scale = float(group.get("scale", 1.0))
        steps = 8
        dt = 1.0 / steps
        for i in range(1, int(n_samples) + 1):
            resid = _sample_noise(dim, scale, seed0 + 101 + i)
            for s in range(steps):
                t = (s + 0.5) / steps
                xin = np.hstack([cond, resid.reshape(1, -1), [[t]]])
                vel = group["model"].predict(xin)[0]
                resid = resid + dt * vel
            method = f"{prefix}_sample{i:02d}"
            out[method] = (_resid_to_path(task, resid, shrink=0.75), {
                "candidate_origin": "direct_conditional_flow_matching",
                "source_method": method,
                "direct_generator_family": "conditional_flow_matching",
                "direct_generator_sample": int(i),
                "n_source_candidates": int(group.get("n_train", 0)),
                "is_direct_generation": 1,
                "is_experimental_v31": 1,
            })
        return out

    if variant == "c":
        group = _get_group(artifacts, n)
        if not group:
            return {}
        rng = np.random.default_rng(seed0 + 211)
        zmean = np.asarray(group.get("latent_mean"), dtype=float)
        zstd = np.asarray(group.get("latent_std"), dtype=float)
        latent_grid = [np.zeros_like(zmean)]
        for _ in range(max(0, int(n_samples) - 1)):
            latent_grid.append(zmean + rng.normal(0.0, 1.0, size=len(zmean)) * zstd)
        for i, z in enumerate(latent_grid[:int(n_samples)], start=1):
            pred = group["decoder"].predict(np.hstack([cond, z.reshape(1, -1)]))[0]
            shrink = 0.80 if i == 1 else 0.90
            method = f"{prefix}_sample{i:02d}"
            out[method] = (_resid_to_path(task, pred, shrink=shrink), {
                "candidate_origin": "direct_conditional_latent_residual_decoder",
                "source_method": method,
                "direct_generator_family": "conditional_latent_decoder",
                "direct_generator_sample": int(i),
                "n_source_candidates": int(group.get("n_train", 0)),
                "latent_dim": int(group.get("n_latent", 0)),
                "is_direct_generation": 1,
                "is_experimental_v31": 1,
            })
        return out

    if variant == "d":
        group = artifacts.get("global_model", {})
        if not group:
            return {}
        pipe = group["model"]
        for i, shrink in enumerate([0.60, 0.75, 0.90, 1.00], start=1):
            resid = np.zeros((n, 2), dtype=float)
            for j in range(1, n - 1):
                tau = j / max(n - 1, 1)
                rem = 1.0 - tau
                xrow = np.hstack([cond.ravel(), [tau, rem], resid[j - 1]]).reshape(1, -1)
                pred = pipe.predict(xrow)[0]
                resid[j] = pred
            # mild endpoint tapering to avoid late drift
            w = np.sin(np.linspace(0, np.pi, n)).reshape(-1, 1)
            resid = resid * w
            method = f"{prefix}_shrink{str(shrink).replace('.', 'p')}"
            out[method] = (_resid_to_path(task, resid.reshape(-1), shrink=shrink), {
                "candidate_origin": "direct_goal_conditioned_action_policy",
                "source_method": method,
                "direct_generator_family": "goal_conditioned_action_policy",
                "direct_generator_sample": int(i),
                "n_source_candidates": int(group.get("n_train_rows", 0)),
                "is_direct_generation": 1,
                "is_experimental_v31": 1,
            })
        return out

    if variant == "e":
        group = artifacts.get("global_model", {})
        if not group:
            return {}
        pipe = group["model"]
        resid = np.zeros((n, 2), dtype=float)
        for j in range(n):
            tau = j / max(n - 1, 1)
            tf = [
                tau, 1.0 - tau,
                np.sin(np.pi * tau), np.cos(np.pi * tau),
                np.sin(2 * np.pi * tau), np.cos(2 * np.pi * tau),
                float(j == 0), float(j == n - 1),
            ]
            xrow = np.hstack([cond.ravel(), tf]).reshape(1, -1)
            resid[j] = pipe.predict(xrow)[0]
        resid[0] = 0.0
        resid[-1] = 0.0
        for i, shrink in enumerate([0.60, 0.75, 0.90, 1.00], start=1):
            method = f"{prefix}_shrink{str(shrink).replace('.', 'p')}"
            out[method] = (_resid_to_path(task, resid.reshape(-1), shrink=shrink), {
                "candidate_origin": "direct_implicit_continuous_trajectory_sr",
                "source_method": method,
                "direct_generator_family": "implicit_continuous_trajectory_sr",
                "direct_generator_sample": int(i),
                "n_source_candidates": int(group.get("n_train_rows", 0)),
                "is_direct_generation": 1,
                "is_experimental_v31": 1,
            })
        return out

    return {}


def make_v31_direct_topk_diagnostics(metrics: pd.DataFrame, ks=(1, 3, 5, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oracle-within-generated-set diagnostics for latent direct generators."""
    if metrics is None or metrics.empty or "method" not in metrics.columns or "task_uid" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    try:
        from .metrics import add_linear_baseline_metrics
        mm = add_linear_baseline_metrics(metrics.copy())
    except Exception:
        mm = metrics.copy()
    method = mm["method"].astype(str)
    methods = sorted([m for m in method.unique() if m.startswith("direct_sr_")])
    if not methods:
        return pd.DataFrame(), pd.DataFrame()
    rows, choices = [], []
    # Determine natural order by sample/shrink suffix when available.
    def _rank(m: str) -> int:
        mmatch = re.search(r"sample(\d+)", m)
        if mmatch:
            return int(mmatch.group(1))
        smatch = re.search(r"shrink", m)
        if smatch:
            order = ["0p60", "0p75", "0p90", "1p00"]
            for j, token in enumerate(order, start=1):
                if token in m:
                    return j
        return 999
    methods = sorted(methods, key=lambda x: (_rank(x), x))
    for k in ks:
        allowed = [m for m in methods if _rank(m) <= int(k)]
        if not allowed:
            continue
        out_name = f"direct_sr_top{k}_minADE_candidate_set"
        for uid, g in mm[mm["method"].astype(str).isin(allowed)].groupby("task_uid", sort=False):
            gg = g.copy()
            gg["_ade_sort"] = pd.to_numeric(gg["ADE"], errors="coerce")
            if gg["_ade_sort"].notna().sum() == 0:
                continue
            best = gg.sort_values(["_ade_sort", "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
            src = str(best.get("method", "unknown"))
            best["method"] = out_name
            best["source_method"] = src
            best["candidate_set_k"] = int(k)
            best["candidate_set_methods"] = ";".join(allowed)
            best["oracle_within_generated_set"] = 1
            best["not_deployable_top1"] = 1
            best["selector_version"] = "v31_direct_generator_candidate_set_diagnostic"
            rows.append(best.to_dict())
        choices.append({
            "paper_method": out_name,
            "topk_k": int(k),
            "ranked_methods": ";".join(allowed),
            "selection_reason": "oracle_within_v31_direct_generated_set",
        })
    return pd.DataFrame(rows), pd.DataFrame(choices)


# -----------------------------------------------------------------------------
# V32 expanded candidate-set union diagnostic
# -----------------------------------------------------------------------------
def _v32_rank_from_method(method: str) -> int:
    """Extract natural rank/sample order from a generated method name."""
    m = str(method)
    mm = re.search(r"rank(\d+)", m)
    if mm:
        return int(mm.group(1))
    mm = re.search(r"sample(\d+)", m)
    if mm:
        return int(mm.group(1))
    return 999


def make_v32_expanded_candidate_set_diagnostics(metrics: pd.DataFrame, pairs=((3, 3), (5, 5), (10, 10))) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union V30 diverse SR candidates with latentc latent direct candidates.

    The rows are oracle-within-generated-set diagnostics.  They evaluate whether
    the expanded probabilistic candidate set contains a closer trajectory, not
    whether the method can select that trajectory as a deployable Top-1 path.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns or "task_uid" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    try:
        from .metrics import add_linear_baseline_metrics
        mm = add_linear_baseline_metrics(metrics.copy())
    except Exception:
        mm = metrics.copy()

    method = mm["method"].astype(str)
    diverse_methods = sorted(
        [m for m in method.unique() if str(m).startswith("probabilistic_tg_diverse_sr_rank")],
        key=lambda x: (_v32_rank_from_method(x), str(x))
    )
    latent_methods = sorted(
        [m for m in method.unique() if str(m).startswith("direct_sr_conditional_latent_decoder_sample")],
        key=lambda x: (_v32_rank_from_method(x), str(x))
    )
    if not diverse_methods or not latent_methods:
        return pd.DataFrame(), pd.DataFrame()

    rows, choices = [], []
    for diverse_k, latent_k in pairs:
        allowed_diverse = [m for m in diverse_methods if _v32_rank_from_method(m) <= int(diverse_k)]
        allowed_latent = [m for m in latent_methods if _v32_rank_from_method(m) <= int(latent_k)]
        allowed = allowed_diverse + allowed_latent
        if not allowed:
            continue
        total_k = len(allowed)
        method_name = f"expanded_sr_top{total_k}_minADE_candidate_set"
        for uid, g in mm[mm["method"].astype(str).isin(allowed)].groupby("task_uid", sort=False):
            gg = g.copy()
            gg["_ade_sort"] = pd.to_numeric(gg["ADE"], errors="coerce")
            if gg["_ade_sort"].notna().sum() == 0:
                continue
            best = gg.sort_values(["_ade_sort", "method"], ascending=[True, True], kind="mergesort").iloc[0].copy()
            source = str(best.get("method", "unknown"))
            best["method"] = method_name
            best["source_method"] = source
            best["candidate_set_k"] = int(total_k)
            best["diverse_sr_k"] = int(len(allowed_diverse))
            best["latent_direct_k"] = int(len(allowed_latent))
            best["candidate_set_methods"] = ";".join(allowed)
            best["oracle_within_generated_set"] = 1
            best["not_deployable_top1"] = 1
            best["selector_version"] = "expanded_diverse_sr_plus_latent_direct_candidate_set"
            best["candidate_set_family"] = "expanded_probabilistic_union"
            rows.append(best.to_dict())
        choices.append({
            "paper_method": method_name,
            "candidate_set_k": int(total_k),
            "diverse_sr_k": int(len(allowed_diverse)),
            "latent_direct_k": int(len(allowed_latent)),
            "ranked_methods": ";".join(allowed),
            "selection_reason": "oracle_within_expanded_diverse_sr_plus_latent_direct_candidate_set",
        })
    return pd.DataFrame(rows), pd.DataFrame(choices)
