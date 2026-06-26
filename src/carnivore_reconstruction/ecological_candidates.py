
"""Ecology/time-geography constrained candidate generators.

Lightweight generators that combine learned movement residuals with
hard endpoint constraints and simple time-geographic/energy penalties.

Candidate families
------------------
step_selection: Integrated step-selection/time-prism generator.
constrained_decoder: Energy-based constrained decoder with stochastic residual search.
"""
from __future__ import annotations

from typing import Sequence
import re
import numpy as np
import pandas as pd

from .tasks import ReconstructionTask
from .timing import status
from .utils import stable_hash01


ECO_ACTIVE_FAMILY = "constrained_decoder"
ECO_PREFIX = {
    "a": "ecological_tg_maxent_energy",
    "b": "ecological_tg_step_selection",
    "c": "ecological_tg_constrained_decoder",
    "step_selection": "ecological_tg_step_selection",
    "constrained_decoder": "ecological_tg_constrained_decoder",
}
FAMILY_ALIAS = {
    "step_selection": "b",
    "constrained_decoder": "c",
    "b": "b",
    "c": "c",
    "a": "a",
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


def _residual(task: ReconstructionTask) -> np.ndarray | None:
    if task.truth_xy is None:
        return None
    truth = np.asarray(task.truth_xy, dtype=float)
    if truth.ndim != 2 or len(truth) != int(task.n_points):
        return None
    r = truth[:, :2] - _linear_path(task)
    r[0] = 0
    r[-1] = 0
    return r


def _path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _turn_cost(path: np.ndarray) -> float:
    d = np.diff(np.asarray(path, dtype=float), axis=0)
    if len(d) < 2:
        return 0.0
    a = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(np.unwrap(a))
    return float(np.nanmean(np.abs(da))) if len(da) else 0.0


def _energy(task: ReconstructionTask, path: np.ndarray, artifacts: dict) -> float:
    base = _linear_path(task)
    disp = float(np.linalg.norm(base[-1] - base[0]))
    straight_len = max(disp, 1e-6)
    length = _path_length(path)
    ratio = length / straight_len
    n = int(task.n_points)
    step = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.asarray([0.0])
    step_q90 = float(artifacts.get("step_q90_by_n", {}).get(str(n), artifacts.get("global_step_q90", 1000.0)))
    ratio_med = float(artifacts.get("path_ratio_median_by_n", {}).get(str(n), artifacts.get("global_path_ratio_median", 1.2)))
    endpoint_penalty = float(np.linalg.norm(path[0] - base[0]) + np.linalg.norm(path[-1] - base[-1]))
    step_penalty = float(np.nanmean(np.maximum(0, step - step_q90) ** 2) / max(step_q90 ** 2, 1.0))
    ratio_penalty = abs(ratio - ratio_med)
    turn_penalty = _turn_cost(path)
    # Environment is implicitly represented if evaluated later; here we keep generation
    # environment-agnostic but time-geographically feasible.
    return float(3.0 * endpoint_penalty + 2.0 * step_penalty + 1.0 * ratio_penalty + 0.25 * turn_penalty)


def _condition_key(task: ReconstructionTask) -> str:
    return f"{getattr(task, 'taxon', 'unknown')}_{getattr(task, 'setting_name', 'unknown')}_n{int(task.n_points)}"


def build_ecological_artifacts(tasks: Sequence[ReconstructionTask], variant: str = ECO_ACTIVE_FAMILY, random_state: int = 42) -> dict:
    tasks = list(tasks)
    artifacts = {
        "version": "ecological_candidate_generator",
        "variant": FAMILY_ALIAS.get(str(variant or ECO_ACTIVE_FAMILY).lower(), str(variant or ECO_ACTIVE_FAMILY).lower()),
        "prefix": ECO_PREFIX.get(FAMILY_ALIAS.get(str(variant or ECO_ACTIVE_FAMILY).lower(), str(variant or ECO_ACTIVE_FAMILY).lower()), "ecological_tg_generator"),
        "n_train_tasks_available": int(len(tasks)),
        "median_residual_by_n": {},
        "residual_library_by_n": {},
        "step_q90_by_n": {},
        "path_ratio_median_by_n": {},
        "condition_residual_by_key": {},
    }
    residuals_by_n = {}
    ratios_by_n = {}
    steps_by_n = {}
    cond_by_key = {}
    for task in tasks:
        r = _residual(task)
        if r is None:
            continue
        n = int(task.n_points)
        residuals_by_n.setdefault(n, []).append(r)
        truth = np.asarray(task.truth_xy, dtype=float)
        straight = max(float(np.linalg.norm(np.asarray(task.end_xy) - np.asarray(task.start_xy))), 1e-6)
        ratios_by_n.setdefault(n, []).append(_path_length(truth) / straight)
        if len(truth) > 1:
            steps_by_n.setdefault(n, []).extend(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).tolist())
        cond_by_key.setdefault(_condition_key(task), []).append(r)
    all_steps = []
    all_ratios = []
    for n, vals in residuals_by_n.items():
        arr = np.stack(vals, axis=0)
        artifacts["median_residual_by_n"][str(n)] = np.nanmedian(arr, axis=0)
        # Keep a compact library of residual templates for constrained search.
        max_keep = min(len(arr), 80)
        artifacts["residual_library_by_n"][str(n)] = arr[:max_keep]
    for n, vals in steps_by_n.items():
        arr = np.asarray(vals, dtype=float)
        all_steps.extend(arr[np.isfinite(arr)].tolist())
        artifacts["step_q90_by_n"][str(n)] = float(np.nanquantile(arr, 0.90)) if len(arr) else 1000.0
    for n, vals in ratios_by_n.items():
        arr = np.asarray(vals, dtype=float)
        all_ratios.extend(arr[np.isfinite(arr)].tolist())
        artifacts["path_ratio_median_by_n"][str(n)] = float(np.nanmedian(arr)) if len(arr) else 1.2
    for key, vals in cond_by_key.items():
        arr = np.stack(vals, axis=0)
        artifacts["condition_residual_by_key"][key] = np.nanmedian(arr, axis=0)
    artifacts["global_step_q90"] = float(np.nanquantile(all_steps, 0.90)) if all_steps else 1000.0
    artifacts["global_path_ratio_median"] = float(np.nanmedian(all_ratios)) if all_ratios else 1.2
    status(f"trained ecological/time-geographic artifacts for {len(residuals_by_n)} n_points group(s).")
    return artifacts


def _get_template(task: ReconstructionTask, artifacts: dict) -> np.ndarray:
    n = int(task.n_points)
    key = _condition_key(task)
    if key in artifacts.get("condition_residual_by_key", {}):
        r = np.asarray(artifacts["condition_residual_by_key"][key], dtype=float)
    else:
        r = np.asarray(artifacts.get("median_residual_by_n", {}).get(str(n), np.zeros((n, 2))), dtype=float)
    if r.shape != (n, 2):
        rr = np.zeros((n, 2), dtype=float)
        m = min(n, len(r))
        rr[:m] = np.asarray(r).reshape((-1, 2))[:m]
        r = rr
    r[0] = 0
    r[-1] = 0
    return r


def _path_from_resid(task: ReconstructionTask, resid: np.ndarray, shrink=1.0) -> np.ndarray:
    base = _linear_path(task)
    r = np.asarray(resid, dtype=float).reshape((-1, 2))
    if len(r) != len(base):
        rr = np.zeros_like(base)
        m = min(len(base), len(r))
        rr[:m] = r[:m]
        r = rr
    out = base + float(shrink) * r
    out[0] = base[0]
    out[-1] = base[-1]
    out[~np.isfinite(out)] = base[~np.isfinite(out)]
    return out


def _lateral_basis(task: ReconstructionTask, amp: float, phase: float = 0.0) -> np.ndarray:
    base = _linear_path(task)
    v = base[-1] - base[0]
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        normal = np.asarray([1.0, 0.0])
    else:
        normal = np.asarray([-v[1], v[0]]) / norm
    t = np.linspace(0, 1, len(base))
    wave = np.sin(np.pi * t + phase)
    r = wave[:, None] * normal[None, :] * float(amp)
    r[0] = 0
    r[-1] = 0
    return r


def generate_ecological_paths_for_task(model, task: ReconstructionTask, n_samples: int = 10) -> dict[str, tuple[np.ndarray, dict]]:
    artifacts = getattr(model, "ecological_artifacts", getattr(model, "ecological_artifacts", None))
    if not isinstance(artifacts, dict) or not artifacts:
        return {}

    # Pooled extension: generate both ecological candidate families if the
    # model stores multiple artifact dictionaries.
    if isinstance(artifacts.get("artifacts_by_variant"), dict):
        out = {}
        for _variant, _art in artifacts.get("artifacts_by_variant", {}).items():
            if not isinstance(_art, dict) or not _art:
                continue
            sub_model = type("_EcologicalGeneratorModel", (), {})()
            sub_model.ecological_artifacts = _art
            out.update(generate_ecological_paths_for_task(sub_model, task, n_samples=n_samples))
        return out

    variant = FAMILY_ALIAS.get(str(artifacts.get("variant", ECO_ACTIVE_FAMILY)).lower(), str(artifacts.get("variant", ECO_ACTIVE_FAMILY)).lower())
    prefix = str(artifacts.get("prefix", ECO_PREFIX.get(variant, "ecological_tg_generator")))
    n = int(task.n_points)
    base = _linear_path(task)
    disp = float(np.linalg.norm(base[-1] - base[0]))
    template = _get_template(task, artifacts)
    out = {}

    if variant == "a":
        # MaxEnt-style: combine learned median residual with time-geographic energy screening.
        candidates = []
        amp = max(5.0, 0.10 * disp)
        for i, shrink in enumerate([0.25, 0.50, 0.75, 1.00], start=1):
            candidates.append((f"template_shrink{str(shrink).replace('.', 'p')}", _path_from_resid(task, template, shrink)))
        for phase_i, phase in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], start=1):
            candidates.append((f"lateral{phase_i}", _path_from_resid(task, template + _lateral_basis(task, amp, phase), 0.50)))
        scored = sorted([(name, path, _energy(task, path, artifacts)) for name, path in candidates], key=lambda x: (x[2], x[0]))
        for rank, (name, path, energy) in enumerate(scored[:int(n_samples)], start=1):
            method = f"{prefix}_rank{rank:02d}"
            out[method] = (path, {
                "candidate_origin": "maxent_time_geographic_energy_model",
                "source_method": method,
                "ecological_generator_family": "maxent_time_geographic_energy",
                "ecological_energy": float(energy),
                "is_ecological_direct_generation": 1,
                "is_ecological_candidate": 1,
            })
        return out

    if variant == "b":
        # Integrated step-selection time-prism: deterministic beams with remaining
        # reachability repair. This produces feasible endpoint-conditioned paths.
        step_q90 = float(artifacts.get("step_q90_by_n", {}).get(str(n), artifacts.get("global_step_q90", 1000.0)))
        rng = np.random.default_rng(int(stable_hash01(str(task.task_uid)) * 1_000_000) + 421)
        for sample in range(1, int(n_samples) + 1):
            xy = np.zeros_like(base)
            xy[0] = base[0]
            lateral_scale = (0.05 + 0.02 * sample) * max(disp, step_q90)
            for j in range(1, n - 1):
                remaining = max(n - j, 1)
                target_step = (base[-1] - xy[j - 1]) / remaining
                template_step = template[j] - template[j - 1] if j < len(template) else 0.0
                noise = rng.normal(0, lateral_scale / max(n, 2), size=2)
                prop = xy[j - 1] + target_step + 0.45 * template_step + noise
                # Reachability repair: keep enough distance budget to reach endpoint.
                rem_dist = float(np.linalg.norm(base[-1] - prop))
                max_rem = step_q90 * remaining
                if rem_dist > max_rem and rem_dist > 1e-9:
                    prop = base[-1] + (prop - base[-1]) * (max_rem / rem_dist)
                xy[j] = prop
            xy[-1] = base[-1]
            method = f"{prefix}_sample{sample:02d}"
            out[method] = (xy, {
                "candidate_origin": "integrated_step_selection_time_prism",
                "source_method": method,
                "ecological_generator_family": "integrated_step_selection_time_prism",
                "ecological_energy": float(_energy(task, xy, artifacts)),
                "is_ecological_direct_generation": 1,
                "is_ecological_candidate": 1,
            })
        return out

    if variant == "c":
        # Energy-based constrained decoder: sample residual-template combinations
        # and keep lowest-energy endpoint-feasible paths.
        rng = np.random.default_rng(int(stable_hash01(str(task.task_uid)) * 1_000_000) + 907)
        lib = np.asarray(artifacts.get("residual_library_by_n", {}).get(str(n), []), dtype=float)
        candidates = []
        amp = max(5.0, 0.12 * disp)
        for i in range(80):
            if lib.ndim == 3 and len(lib):
                idx = rng.integers(0, len(lib), size=min(3, len(lib)))
                weights = rng.dirichlet(np.ones(len(idx)))
                resid = np.sum(lib[idx] * weights[:, None, None], axis=0)
            else:
                resid = template.copy()
            resid = resid + _lateral_basis(task, amp * rng.uniform(-1.0, 1.0), rng.uniform(0, 2 * np.pi))
            shrink = rng.uniform(0.25, 0.95)
            path = _path_from_resid(task, resid, shrink=shrink)
            candidates.append((path, _energy(task, path, artifacts)))
        scored = sorted(candidates, key=lambda x: x[1])
        for rank, (path, energy) in enumerate(scored[:int(n_samples)], start=1):
            method = f"{prefix}_rank{rank:02d}"
            out[method] = (path, {
                "candidate_origin": "energy_based_constrained_decoder",
                "source_method": method,
                "ecological_generator_family": "energy_based_constrained_decoder",
                "ecological_energy": float(energy),
                "is_ecological_direct_generation": 1,
                "is_ecological_candidate": 1,
            })
        return out

    return {}



# -----------------------------------------------------------------------------
# Balanced pooled Top-10 candidate-set diagnostic
# -----------------------------------------------------------------------------

def _candidate_rank_from_method(method: str) -> int:
    m = str(method)
    mm = re.search(r"(?:rank|sample)(\d+)", m)
    return int(mm.group(1)) if mm else 999


def _candidate_family(method: str) -> str:
    m = str(method)
    if m.startswith("probabilistic_tg_diverse_sr_rank"):
        return "motif_diverse_sr"
    if m.startswith("direct_sr_conditional_latent_decoder_sample"):
        return "latent_direct"
    if m.startswith("ecological_tg_constrained_decoder"):
        return "ecological_constrained_decoder"
    if m.startswith("ecological_tg_step_selection"):
        return "ecological_step_selection"
    return "other"


def _family_priority(method: str) -> int:
    family = _candidate_family(method)
    return {
        "motif_diverse_sr": 0,
        "latent_direct": 1,
        "ecological_constrained_decoder": 2,
        "ecological_step_selection": 3,
    }.get(family, 9)


def _candidate_proxy_score(row: pd.Series) -> float:
    """Deployable proxy score used only to choose which candidates enter Top-10."""
    vals = []
    for col, sign, weight in [
        ("total_cost", 1.0, 1.0),
        ("score", 1.0, 1.0),
        ("cost", 1.0, 1.0),
        ("path_cost", 1.0, 1.0),
        ("ecological_energy", 1.0, 0.75),
        ("path_to_straight_ratio", 1.0, 0.25),
        ("directness", -1.0, 0.25),
        ("source_rank", 1.0, 0.25),
        ("candidate_rank", 1.0, 0.25),
    ]:
        if col in row.index:
            try:
                v = float(row.get(col))
                if np.isfinite(v):
                    vals.append(sign * weight * v)
            except Exception:
                pass
    method = str(row.get("method", ""))
    vals.append(0.10 * _candidate_rank_from_method(method))
    vals.append(0.20 * _family_priority(method))
    return float(np.nansum(vals)) if vals else float(_candidate_rank_from_method(method) + _family_priority(method))


def make_balanced_pooled_top10_diagnostics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the paper-facing balanced pooled Top-10 candidate-set row.

    Quota:
    - 5 motif diverse-SR candidates
    - 2 latent direct candidates
    - 2 ecological constrained-decoder candidates
    - 1 integrated step-selection/time-prism candidate

    The retained Top-10 is selected by deployable proxy score within family.  The
    reported row is the oracle-within-retained-Top-10 diagnostic used to evaluate
    whether the compact candidate set contains a plausible trajectory close to
    the observed high-resolution path.
    """
    if metrics is None or metrics.empty or "method" not in metrics.columns or "task_uid" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()

    methods = metrics["method"].dropna().astype(str)
    is_source = (
        methods.str.startswith("probabilistic_tg_diverse_sr_rank")
        | methods.str.startswith("direct_sr_conditional_latent_decoder_sample")
        | methods.str.startswith("ecological_tg_step_selection")
        | methods.str.startswith("ecological_tg_constrained_decoder")
    )
    src = metrics.loc[is_source].copy()
    if src.empty:
        return pd.DataFrame(), pd.DataFrame()

    src["_candidate_proxy_score"] = src.apply(_candidate_proxy_score, axis=1)
    src["_candidate_family"] = src["method"].astype(str).map(_candidate_family)

    quota = {
        "motif_diverse_sr": 5,
        "latent_direct": 2,
        "ecological_constrained_decoder": 2,
        "ecological_step_selection": 1,
    }
    method_name = "balanced_pooled_extension_top10_minADE_candidate_set"
    rows = []
    choices = []

    for uid, g in src.groupby("task_uid", sort=False):
        retained_parts = []
        for family, k in quota.items():
            sub = g[g["_candidate_family"].eq(family)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["_candidate_proxy_score", "method"], ascending=[True, True], kind="mergesort")
            retained_parts.append(sub.head(int(k)))

        retained = pd.concat(retained_parts, ignore_index=True, sort=False) if retained_parts else pd.DataFrame()
        if len(retained) < 10:
            already = set(retained["method"].astype(str).tolist()) if not retained.empty else set()
            rem = g[~g["method"].astype(str).isin(already)].copy()
            rem = rem.sort_values(["_candidate_proxy_score", "method"], ascending=[True, True], kind="mergesort")
            retained = pd.concat([retained, rem.head(10 - len(retained))], ignore_index=True, sort=False)

        retained = retained.head(10).copy()
        if retained.empty:
            continue
        retained["_ade_sort"] = pd.to_numeric(retained["ADE"], errors="coerce")
        if retained["_ade_sort"].notna().sum() == 0:
            continue

        family_counts = retained["_candidate_family"].value_counts().to_dict()
        selected_methods = ";".join(retained["method"].astype(str).tolist())
        best = retained.sort_values(["_ade_sort", "_candidate_proxy_score", "method"], ascending=[True, True, True], kind="mergesort").iloc[0].copy()
        source_method = str(best.get("method", "unknown"))

        best["method"] = method_name
        best["source_method"] = source_method
        best["candidate_set_k"] = 10
        best["candidate_set_methods"] = selected_methods
        best["candidate_family_counts"] = str(family_counts)
        best["oracle_within_generated_set"] = 1
        best["not_deployable_top1"] = 1
        best["selector_version"] = "balanced_pooled_extension_top10"
        best["candidate_set_family"] = "balanced_pooled_motif_latent_ecological_top10"
        rows.append(best.to_dict())

        choices.append({
            "task_uid": uid,
            "paper_method": method_name,
            "candidate_set_k": 10,
            "candidate_set_methods": selected_methods,
            "candidate_family_counts": str(family_counts),
            "selected_source_method": source_method,
            "selected_source_family": _candidate_family(source_method),
            "selection_reason": "balanced_quota_top10_across_motif_latent_and_ecological_candidate_families",
        })

    return pd.DataFrame(rows), pd.DataFrame(choices)


def ensure_ecological_artifacts(model, train_tasks: Sequence[ReconstructionTask] | None = None, timer=None):
    """Ensure a loaded model has ecological candidate artifacts.

    Existing pretrained models created before the balanced Top-10 extension may
    not contain these artifacts.  If training tasks are supplied, this function
    builds the two lightweight ecological artifact families and attaches them to
    the model in memory.
    """
    artifacts = getattr(model, "ecological_artifacts", None)
    if isinstance(artifacts, dict) and artifacts.get("artifacts_by_variant"):
        return model
    if train_tasks is None:
        return model
    train_tasks = list(train_tasks)
    if not train_tasks:
        return model

    artifacts_by_variant = {}
    for family in ["step_selection", "constrained_decoder"]:
        if timer is not None:
            with timer.step("build_missing_ecological_candidate_family", n_tasks=len(train_tasks), family=family):
                artifacts_by_variant[family] = build_ecological_artifacts(train_tasks, variant=family)
        else:
            artifacts_by_variant[family] = build_ecological_artifacts(train_tasks, variant=family)

    model.ecological_artifacts = {
        "family": "pooled_step_selection_and_constrained_decoder",
        "artifacts_by_variant": artifacts_by_variant,
        "n_train_tasks_available": int(len(train_tasks)),
    }
    return model
