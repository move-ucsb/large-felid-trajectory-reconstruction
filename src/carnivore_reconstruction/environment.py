"""Raster and environmental-exposure utilities.

Raster covariates are optional. When available, they are sampled in two places:
(1) observed fixes during data loading, so task truth paths carry environmental
    exposure values; and
(2) reconstructed/baseline paths during evaluation, so environmental exposure
    errors can be computed.

The reconstruction selector remains deployable: it does not use hidden truth
rasters for ranking unless a future notebook explicitly adds a deployable
raster-aware scoring term.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Iterable

import numpy as np
import pandas as pd

from .timing import status

RASTER_EXTENSIONS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def _iter_raster_files(directory: str | Path | None, recursive: bool = True) -> list[Path]:
    """Return raster files under a directory, sorted and de-duplicated."""
    if directory is None:
        return []
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return []
    files: list[Path] = []
    for pat in RASTER_EXTENSIONS:
        files.extend(d.rglob(pat) if recursive else d.glob(pat))
    out: list[Path] = []
    seen: set[Path] = set()
    for p in sorted(files):
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def resolve_raster_paths(
    raster_paths: Mapping[str, str | Path] | None = None,
    raster_dir: str | Path | None = None,
    *,
    recursive: bool = True,
) -> dict[str, Path]:
    """Resolve named raster paths without duplicating the same file.

    Explicit ``raster_paths`` are treated as authoritative because they map
    filenames/aliases to the paper-facing environmental column names. Directory
    auto-discovery is only used for extra rasters not already listed. When the
    same tif is found through both mechanisms, it is kept once under the
    explicit name.
    """
    out: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    if raster_paths:
        for name, path in raster_paths.items():
            p = Path(path)
            if p.exists():
                rp = p.resolve()
                out[str(name)] = p
                seen_paths.add(rp)
    for p in _iter_raster_files(raster_dir, recursive=recursive):
        rp = p.resolve()
        if rp in seen_paths:
            continue
        out.setdefault(p.stem, p)
        seen_paths.add(rp)
    return out


def _transform_coords_for_dataset(
    coords: list[tuple[float, float]],
    dataset_crs,
    source_epsg: int | None,
    *,
    raster_epsg: int | None = None,
) -> list[tuple[float, float]]:
    """Transform source x/y coordinates to raster CRS only when necessary.

    The usual project setup stores trajectory x/y and rasters in the same UTM
    CRS by study system:

    * Santa Cruz puma / Olympic cougar / Olympic bobcat: EPSG:32610
    * Thailand tiger/leopard: EPSG:32647

    When ``source_epsg`` and ``raster_epsg`` match, coordinates are sampled
    directly without asking PROJ/rasterio to resolve EPSG metadata. This avoids
    failures from local PROJ database conflicts such as ``proj.db lacks
    DATABASE.LAYOUT.VERSION``.
    """
    if source_epsg is None:
        return coords

    # Explicit dataset/raster EPSG match: no transform needed, and no PROJ call.
    if raster_epsg is not None:
        try:
            if int(source_epsg) == int(raster_epsg):
                return coords
        except Exception:
            pass

    # If no explicit raster EPSG was provided, fall back to raster metadata.
    if dataset_crs is None:
        return coords

    try:
        import rasterio
        from rasterio.warp import transform
        src_crs = rasterio.crs.CRS.from_epsg(int(source_epsg))
        dst_crs = dataset_crs
        if src_crs == dst_crs:
            return coords
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        tx, ty = transform(src_crs, dst_crs, xs, ys)
        return list(zip(tx, ty))
    except Exception as exc:
        status(f"Could not transform coordinates for raster CRS; sampling with original x/y. Reason: {exc}")
        return coords

def sample_rasters_at_xy(
    df: pd.DataFrame,
    raster_paths: Mapping[str, str | Path],
    x_col: str = "x",
    y_col: str = "y",
    prefix: str = "",
    *,
    source_epsg: int | None = None,
    raster_epsg: int | None = None,
) -> pd.DataFrame:
    """Sample rasters at x/y coordinates and append columns to ``df``.

    If ``source_epsg`` and ``raster_epsg`` are supplied and they match, x/y are
    sampled directly. This is the recommended project setup because the tracking
    data and rasters are already projected to the same UTM zone. If they differ,
    the function attempts a CRS transform and falls back safely to original x/y
    if local PROJ metadata is broken. Missing rasters, invalid points, and
    out-of-bounds samples become NaN rather than failing the workflow.
    """
    out = df.copy()
    paths = resolve_raster_paths(raster_paths)
    if not paths:
        return out
    try:
        import rasterio
    except Exception:
        status("rasterio is not installed; skipping raster environmental sampling")
        return out

    raw_x = pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=float)
    raw_y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
    base_coords = list(zip(raw_x, raw_y))

    for name, path in paths.items():
        col = f"{prefix}{name}" if prefix else str(name)
        values = np.full(len(out), np.nan, dtype=float)
        try:
            with rasterio.open(path) as ds:
                coords = _transform_coords_for_dataset(base_coords, ds.crs, source_epsg, raster_epsg=raster_epsg)
                for i, val in enumerate(ds.sample(coords)):
                    if len(val):
                        v = float(val[0])
                        if ds.nodata is not None and np.isclose(v, ds.nodata):
                            v = np.nan
                        values[i] = v
        except Exception as exc:
            status(f"Could not sample raster {path.name}: {exc}")
            values[:] = np.nan
        out[col] = values
    return out


def environmental_columns(df: pd.DataFrame) -> list[str]:
    """Return likely environmental covariate columns from a standardized table."""
    exclude = {
        "dataset", "taxon", "animal_id", "animal_name", "animal_key", "time", "x", "y",
        "task_settings_json", "prev_x", "prev_y", "next_x", "next_y", "prev_time", "next_time",
        "step_m", "dt_min", "sex", "age_class", "habitat_id", "study_system",
        "species_common_name", "species_id", "species_group", "genus_group", "transfer_unit",
        "metadata_source", "sex_female", "sex_male", "age_adult", "age_subadult",
        "split", "task_uid", "point_order", "start_time", "end_time", "coarse_dt_min", "fine_dt_min",
        "n_points", "start_x", "start_y", "end_x", "end_y", "displacement_m", "base_step_m",
        "truth_path_length_m", "truth_directness", "has_prev", "has_next",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if str(c).startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def raster_layers_needing_sampling(
    df: pd.DataFrame,
    raster_paths: Mapping[str, str | Path] | None,
    *,
    min_existing_fraction: float = 0.50,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Split raster layers into missing layers and CSV-annotated layers.

    If an environmental column already exists in the input CSV with enough
    non-missing values, we keep it and do not resample the observed fixes. This
    prevents slow full-table raster sampling when values were already annotated
    upstream. Returned status labels are intended for progress logs.
    """
    paths = resolve_raster_paths(raster_paths)
    if not paths:
        return {}, {}
    missing: dict[str, Path] = {}
    existing: dict[str, str] = {}
    for name, path in paths.items():
        col = str(name)
        if col in df.columns:
            frac = float(pd.to_numeric(df[col], errors="coerce").notna().mean()) if len(df) else 0.0
            if frac >= min_existing_fraction:
                existing[col] = f"csv_existing_{frac:.1%}"
                continue
        missing[col] = path
    return missing, existing


def sample_path_environment(
    path_xy: np.ndarray,
    raster_paths: Mapping[str, str | Path] | None,
    *,
    source_epsg: int | None = None,
    raster_epsg: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample environmental rasters along one reconstructed path."""
    if raster_paths is None:
        return {}
    pts = pd.DataFrame({"x": np.asarray(path_xy)[:, 0], "y": np.asarray(path_xy)[:, 1]})
    sampled = sample_rasters_at_xy(pts, raster_paths, source_epsg=source_epsg, raster_epsg=raster_epsg)
    return {c: sampled[c].to_numpy(dtype=float) for c in sampled.columns if c not in {"x", "y"}}


def exposure_summary(values: np.ndarray) -> float:
    """Median exposure summary used for truth-vs-reconstruction diagnostics."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    return float(np.nanmedian(arr))


def environmental_error_dict(truth_env: Mapping[str, np.ndarray] | None, path_env: Mapping[str, np.ndarray] | None) -> dict[str, float]:
    """Return absolute normalized environmental exposure errors by variable.

    For each variable, the error is |median(pred) - median(truth)| divided by
    the truth IQR when possible. Raw absolute error is also returned.
    """
    if not truth_env or not path_env:
        return {}
    out: dict[str, float] = {}
    for name, truth_vals in truth_env.items():
        if name not in path_env:
            continue
        tv = np.asarray(truth_vals, dtype=float)
        pv = np.asarray(path_env[name], dtype=float)
        if tv.size == 0 or pv.size == 0 or np.all(~np.isfinite(tv)) or np.all(~np.isfinite(pv)):
            continue
        tmed = float(np.nanmedian(tv))
        pmed = float(np.nanmedian(pv))
        raw = abs(pmed - tmed)
        q75, q25 = np.nanpercentile(tv, [75, 25])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = float(np.nanstd(tv))
        norm = raw / scale if np.isfinite(scale) and scale > 1e-9 else np.nan
        safe = str(name).replace(" ", "_")
        out[f"env_error_{safe}"] = float(norm) if np.isfinite(norm) else np.nan
        out[f"env_abs_error_{safe}"] = float(raw) if np.isfinite(raw) else np.nan
    if out:
        norm_cols = [v for k, v in out.items() if k.startswith("env_error_") and np.isfinite(v)]
        out["env_error_mean"] = float(np.nanmean(norm_cols)) if norm_cols else np.nan
    return out
