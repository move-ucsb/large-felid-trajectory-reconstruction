"""Data loading and standardization for tracking tables."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import json

import numpy as np
import pandas as pd

from .config import DatasetSpec
from .timing import status
from .environment import sample_rasters_at_xy, resolve_raster_paths, raster_layers_needing_sampling


def _project_lonlat_to_xy(lon, lat, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("pyproj is required when lon/lat columns are used.") from exc
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)




def normalize_animal_id(value: object) -> str:
    """Normalize collar/animal identifiers while preserving meaningful strings.

    Several tracking files store numeric collar IDs as floats, e.g. ``31899.0``.
    Older notebooks treated these as ``31899``. Keeping the same behavior here
    makes per-animal settings and tiger/leopard relabeling robust.
    """
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _settings_to_json(settings: list[dict[str, Any]] | None) -> str:
    """Serialize task settings for storage on every standardized track row."""
    if not settings:
        return ""
    clean = []
    for item in settings:
        if not item:
            continue
        coarse = item.get("coarse_dt_min", item.get("coarse", None))
        fine = item.get("fine_dt_min", item.get("fine", None))
        if coarse is None or fine is None:
            continue
        clean.append({
            "coarse_dt_min": float(coarse),
            "fine_dt_min": float(fine),
            "setting_name": item.get("setting_name", item.get("name", "")),
        })
    return json.dumps(clean) if clean else ""


def normalize_time_column(values, dataset: str = "") -> pd.Series:
    """Parse timestamps into a single UTC-naive dtype.

    Different tracking files often mix timezone-aware strings, such as
    ``2020-01-01T12:00:00Z``, with timezone-naive strings, such as
    ``2020-01-01 12:00:00``. Pandas cannot sort or group a combined
    column containing both representations. This helper standardizes all
    parsed values to ``datetime64[ns]`` with no timezone after converting
    timezone-aware values to UTC. Naive values are treated as already being
    in the intended analysis time reference.
    """
    try:
        # pandas >= 2.0: ``format="mixed"`` handles columns that combine
        # strings with and without timezone suffixes.
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    except TypeError:  # pragma: no cover - for older pandas versions
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    # Convert to UTC clock time and drop timezone info so all datasets share
    # the same pandas dtype and can be sorted after concatenation.
    return parsed.dt.tz_convert(None)



def animal_key(value: object) -> str:
    """Normalized key for metadata joins and robust collar comparisons."""
    s = normalize_animal_id(value).upper().replace(" ", "")
    return s


def _norm_sex(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"f", "female", "fem", "♀"}:
        return "female"
    if s in {"m", "male", "♂"}:
        return "male"
    return "unknown"


def _norm_age(value: object) -> str:
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "unknown", "na", "n/a"}:
        return "unknown"
    if "sub" in s or "juvenile" in s or "juv" in s:
        return "subadult"
    if "adult" in s or s in {"a"}:
        return "adult"
    return s


def _sex_from_suffix(value: object) -> str:
    """Infer sex from IDs ending in F or M, e.g. 121F, B01M, B03F."""
    key = animal_key(value)
    if key.endswith("F"):
        return "female"
    if key.endswith("M"):
        return "male"
    return "unknown"


def _parse_thailand_code(value: object) -> tuple[str, str]:
    """Parse LIST_ALL_TIGER short labels.

    FL/ML are female/male leopard. F/M or FT/MT are female/male tiger.
    """
    key = animal_key(value)
    if not key:
        return "unknown", "unknown"
    if key.startswith("FL"):
        return "female", "leopard"
    if key.startswith("ML"):
        return "male", "leopard"
    if key.startswith("FT") or key == "F" or (key.startswith("F") and not key.startswith("FL")):
        return "female", "tiger"
    if key.startswith("MT") or key == "M" or (key.startswith("M") and not key.startswith("ML")):
        return "male", "tiger"
    if "LEOPARD" in key:
        if key.startswith("F"):
            return "female", "leopard"
        if key.startswith("M"):
            return "male", "leopard"
        return "unknown", "leopard"
    if "TIGER" in key:
        if key.startswith("F"):
            return "female", "tiger"
        if key.startswith("M"):
            return "male", "tiger"
        return "unknown", "tiger"
    return "unknown", "unknown"


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lookup.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _read_metadata_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _detect_overlap_col(meta: pd.DataFrame, target_keys: set[str], preferred: list[str] | None = None) -> str | None:
    preferred = preferred or []
    best_col = None
    best_score = (-1, False, False, -1)
    for col in meta.columns:
        keys = meta[col].map(animal_key)
        overlap = int(keys.isin(target_keys).sum())
        pref = str(col).lower() in {p.lower() for p in preferred}
        id_like = any(tok in str(col).lower() for tok in ["collar", "animal", "name", "id"])
        nonempty = int(keys.ne("").sum())
        score = (overlap, pref, id_like, nonempty)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score[0] > 0 else None


def _detect_thailand_code_col(meta: pd.DataFrame) -> str | None:
    best_col = None
    best_score = (-1, False)
    for col in meta.columns:
        parsed = meta[col].map(_parse_thailand_code)
        hits = int(parsed.map(lambda x: x[1] in {"tiger", "leopard"}).sum())
        name_hint = any(tok in str(col).lower() for tok in ["code", "short", "name", "label", "sex"])
        score = (hits, name_hint)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score[0] > 0 else None


def _metadata_source_append(existing: pd.Series, label: str, mask: pd.Series) -> pd.Series:
    out = existing.astype(str).copy()
    old = out.loc[mask].replace({"unknown": "", "": ""})
    out.loc[mask] = old.map(lambda x: label if not x else f"{x};{label}")
    return out


def apply_individual_metadata(tracks: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Attach species/sex/age metadata while keeping reconstruction features unchanged.

    Metadata rules implemented here:
    - Puma sex is parsed from ID suffix F/M.
    - Thailand LIST_ALL_TIGER labels FL/ML are leopard; F/M or FT/MT are tiger.
    - Olympic bobcat/cougar metadata CSV is used when available; suffix F/M is a fallback.
    """
    tr = tracks.copy()
    tr["animal_id"] = tr["animal_id"].map(normalize_animal_id)
    tr["animal_name"] = tr["animal_id"]
    tr["animal_key"] = tr["animal_id"].map(animal_key)

    if "sex" not in tr.columns:
        tr["sex"] = "unknown"
    if "age_class" not in tr.columns:
        tr["age_class"] = "unknown"
    tr["metadata_source"] = "default"

    habitat_map = {
        "SantaCruz_puma": "SantaCruz",
        "Thailand_tiger_leopard": "Thailand_WEFCOM",
        "Olympic_cougar": "OlympicPeninsula",
        "Olympic_bobcat": "OlympicPeninsula",
    }
    tr["habitat_id"] = tr["dataset"].map(habitat_map).fillna(tr["dataset"])
    tr["study_system"] = tr["habitat_id"]

    # Puma: suffix F/M.
    puma = tr["dataset"].eq("SantaCruz_puma") | tr["taxon"].eq("puma")
    tr.loc[puma, "taxon"] = "puma"
    suffix = tr.loc[puma, "animal_id"].map(_sex_from_suffix)
    mask = puma & suffix.eq("female")
    tr.loc[mask, "sex"] = "female"
    tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "puma_id_suffix", mask)
    mask = puma & suffix.eq("male")
    tr.loc[mask, "sex"] = "male"
    tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "puma_id_suffix", mask)

    # Dataset-level species labels.
    tr.loc[tr["dataset"].eq("Olympic_cougar"), "taxon"] = "cougar"
    tr.loc[tr["dataset"].eq("Olympic_bobcat"), "taxon"] = "bobcat"
    tl = tr["dataset"].eq("Thailand_tiger_leopard")
    tr.loc[tl & tr["taxon"].isna(), "taxon"] = "tiger"

    # Olympic suffix fallback before metadata table.
    olympic = tr["dataset"].isin(["Olympic_cougar", "Olympic_bobcat"])
    suffix = tr.loc[olympic, "animal_id"].map(_sex_from_suffix)
    mask = olympic & tr["sex"].eq("unknown") & suffix.eq("female")
    tr.loc[mask, "sex"] = "female"
    tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "id_suffix", mask)
    mask = olympic & tr["sex"].eq("unknown") & suffix.eq("male")
    tr.loc[mask, "sex"] = "male"
    tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "id_suffix", mask)

    # Optional metadata table. For Thailand this is LIST_ALL_TIGER; for Olympic it
    # is usually Bobcat_cougar_metadata_revised.csv.
    meta = _read_metadata_table(spec.metadata_path)
    if meta is not None and not meta.empty:
        target_keys = set(tr["animal_key"])
        id_col = spec.metadata_id_col or _detect_overlap_col(
            meta,
            target_keys,
            preferred=["idcollar", "IDcollar", "collar_id", "CollarID", "animal_id", "id", "ID", "Name", "name"],
        )
        if id_col is not None:
            sex_col = spec.metadata_sex_col or _first_existing_col(meta, ["sex", "Sex", "gender", "Gender"])
            age_col = spec.metadata_age_col or _first_existing_col(meta, ["age_class", "Age class", "Age class at first capture", "Age", "age"])
            species_col = spec.metadata_species_col or _first_existing_col(meta, ["Species", "species", "taxon", "Taxon"])
            code_col = spec.metadata_code_col or _detect_thailand_code_col(meta) if spec.dataset == "Thailand_tiger_leopard" else spec.metadata_code_col

            mm = pd.DataFrame({"animal_key": meta[id_col].map(animal_key)})
            mm["meta_sex"] = meta[sex_col].map(_norm_sex) if sex_col is not None else "unknown"
            mm["meta_age"] = meta[age_col].map(_norm_age) if age_col is not None else "unknown"
            mm["meta_taxon"] = "unknown"
            if species_col is not None:
                sp = meta[species_col].astype(str).str.lower()
                mm.loc[sp.str.contains("cougar|puma", na=False), "meta_taxon"] = "cougar"
                mm.loc[sp.str.contains("bobcat", na=False), "meta_taxon"] = "bobcat"
                mm.loc[sp.str.contains("leopard", na=False), "meta_taxon"] = "leopard"
                mm.loc[sp.str.contains("tiger", na=False), "meta_taxon"] = "tiger"
            if spec.dataset == "Thailand_tiger_leopard" and code_col is not None:
                parsed = meta[code_col].map(_parse_thailand_code)
                parsed_sex = parsed.map(lambda z: z[0])
                parsed_taxon = parsed.map(lambda z: z[1])
                mm.loc[parsed_taxon.isin(["tiger", "leopard"]), "meta_taxon"] = parsed_taxon[parsed_taxon.isin(["tiger", "leopard"])]
                mm.loc[mm["meta_sex"].eq("unknown") & parsed_sex.ne("unknown"), "meta_sex"] = parsed_sex[mm["meta_sex"].eq("unknown") & parsed_sex.ne("unknown")]

            tmp = tr.merge(mm.drop_duplicates("animal_key"), how="left", on="animal_key")
            joined = tmp["meta_sex"].notna() | tmp["meta_age"].notna() | tmp["meta_taxon"].notna()
            # Use metadata sex/age when informative.
            mask = joined & tmp["meta_sex"].notna() & tmp["meta_sex"].ne("unknown")
            tmp.loc[mask, "sex"] = tmp.loc[mask, "meta_sex"]
            tmp["metadata_source"] = _metadata_source_append(tmp["metadata_source"], Path(spec.metadata_path).name, mask)
            mask = joined & tmp["meta_age"].notna() & tmp["meta_age"].ne("unknown")
            tmp.loc[mask, "age_class"] = tmp.loc[mask, "meta_age"]
            tmp["metadata_source"] = _metadata_source_append(tmp["metadata_source"], Path(spec.metadata_path).name, mask)
            mask = joined & tmp["meta_taxon"].notna() & tmp["meta_taxon"].isin(["tiger", "leopard", "cougar", "bobcat", "puma"])
            tmp.loc[mask, "taxon"] = tmp.loc[mask, "meta_taxon"]
            tmp["metadata_source"] = _metadata_source_append(tmp["metadata_source"], Path(spec.metadata_path).name, mask)
            tr = tmp.drop(columns=[c for c in ["meta_sex", "meta_age", "meta_taxon"] if c in tmp.columns])
            status(f"  metadata joined for {spec.dataset}: {Path(spec.metadata_path).name}, id_col={id_col}")
        else:
            status(f"  metadata file found for {spec.dataset}, but no ID column overlapped tracks: {Path(spec.metadata_path).name}")

    # Thailand hard-coded fallback for known study animals, after metadata table.
    if spec.dataset == "Thailand_tiger_leopard":
        leopard_id_to_sex = {"31899": "female", "31898": "male", "37821": "male", "37822": "male", "37823": "male"}
        tiger_id_to_sex = {"131343": "female", "229011": "female", "229041": "female", "229012": "male", "229022": "male", "229032": "male"}
        for collar, sex in leopard_id_to_sex.items():
            mask = tr["animal_key"].eq(animal_key(collar))
            tr.loc[mask, "taxon"] = "leopard"
            tr.loc[mask & tr["sex"].eq("unknown"), "sex"] = sex
            tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "known_thailand_collars", mask)
        for collar, sex in tiger_id_to_sex.items():
            mask = tr["animal_key"].eq(animal_key(collar))
            tr.loc[mask, "taxon"] = "tiger"
            tr.loc[mask & tr["sex"].eq("unknown"), "sex"] = sex
            tr["metadata_source"] = _metadata_source_append(tr["metadata_source"], "known_thailand_collars", mask)
        # Remaining Thailand rows default to tiger unless explicitly leopard.
        tr.loc[tr["taxon"].ne("leopard"), "taxon"] = "tiger"

    # Transfer-analysis labels.
    # ``taxon`` keeps the user-facing common label found in each dataset
    # (puma/cougar/bobcat/tiger/leopard). ``species_id`` groups biological
    # species across common names and habitats; this is what supports transfer
    # scenarios such as puma vs. cougar as the same species in different
    # habitats. ``genus_group`` is kept for optional broader taxonomic summaries.
    species_id_map = {
        "puma": "puma_concolor",
        "cougar": "puma_concolor",
        "bobcat": "lynx_rufus",
        "tiger": "panthera_tigris",
        "leopard": "panthera_pardus",
    }
    common_name_map = {
        "puma": "puma",
        "cougar": "cougar",
        "bobcat": "bobcat",
        "tiger": "tiger",
        "leopard": "leopard",
    }
    genus_map = {
        "puma": "puma",
        "cougar": "puma",
        "bobcat": "lynx",
        "tiger": "panthera",
        "leopard": "panthera",
    }
    tr["species_common_name"] = tr["taxon"].map(common_name_map).fillna(tr["taxon"])
    tr["species_id"] = tr["taxon"].map(species_id_map).fillna(tr["taxon"])
    tr["species_group"] = tr["species_id"]
    tr["genus_group"] = tr["taxon"].map(genus_map).fillna(tr["taxon"])
    tr["transfer_unit"] = tr["species_id"].astype(str) + "__" + tr["habitat_id"].astype(str)
    tr["sex_female"] = (tr["sex"] == "female").astype(float)
    tr["sex_male"] = (tr["sex"] == "male").astype(float)
    tr["age_adult"] = (tr["age_class"] == "adult").astype(float)
    tr["age_subadult"] = (tr["age_class"] == "subadult").astype(float)
    return tr

def standardize_track_table(raw: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Convert one raw tracking table to the package's standard schema."""
    df = raw.copy()
    required = [spec.id_col, spec.time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} for dataset {spec.dataset}.")

    if spec.x_col and spec.y_col and spec.x_col in df.columns and spec.y_col in df.columns:
        x = pd.to_numeric(df[spec.x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[spec.y_col], errors="coerce").to_numpy(dtype=float)
    elif spec.lon_col and spec.lat_col and spec.lon_col in df.columns and spec.lat_col in df.columns and spec.epsg:
        lon = pd.to_numeric(df[spec.lon_col], errors="coerce")
        lat = pd.to_numeric(df[spec.lat_col], errors="coerce")
        x, y = _project_lonlat_to_xy(lon, lat, spec.epsg)
    else:
        raise KeyError(
            f"Dataset {spec.dataset} needs projected x/y columns or lon/lat+epsg. "
            f"Available columns: {list(df.columns)[:50]}"
        )

    animal_id = df[spec.id_col].map(normalize_animal_id)

    # Relabel taxon by animal ID when a single CSV stores multiple species
    # (e.g., Thailand tiger/leopard collars). Unmapped animals keep spec.taxon.
    taxon_map = {normalize_animal_id(k): v for k, v in (spec.taxon_by_animal_id or {}).items()}
    taxon = animal_id.map(lambda a: taxon_map.get(a, spec.taxon))

    # Store dataset/animal-specific task settings on the standardized table so
    # make_tasks_from_tracks() can remain independent of external config files.
    default_settings_json = _settings_to_json(spec.task_settings)
    animal_settings = {normalize_animal_id(k): v for k, v in (spec.animal_task_settings or {}).items()}
    task_settings_json = animal_id.map(
        lambda a: _settings_to_json(animal_settings.get(a, spec.task_settings)) or default_settings_json
    )

    out = pd.DataFrame({
        "dataset": spec.dataset,
        "taxon": taxon,
        "animal_id": animal_id,
        "time": normalize_time_column(df[spec.time_col], spec.dataset),
        "x": x,
        "y": y,
        "task_settings_json": task_settings_json,
    })
    env_cols = [c for c in spec.env_cols if c in df.columns]
    for c in env_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional raster covariates. Prefer environmental columns already annotated
    # in the CSV. Only missing layers are sampled onto observed fixes. This is
    # much faster for large already-annotated files and avoids overwriting
    # previously curated covariate values.
    raster_paths = resolve_raster_paths(getattr(spec, "raster_paths", None), getattr(spec, "raster_dir", None))
    if raster_paths:
        to_sample, already_present = raster_layers_needing_sampling(out, raster_paths)
        if already_present:
            names = ", ".join(sorted(already_present))
            status(f"  using {len(already_present)} CSV environmental column(s) for {spec.dataset}: {names}")
        if to_sample:
            names = ", ".join(sorted(to_sample))
            status(f"  sampling {len(to_sample)} missing environmental raster(s) for {spec.dataset}: {names}")
            out = sample_rasters_at_xy(
                out,
                to_sample,
                source_epsg=getattr(spec, "epsg", None),
                raster_epsg=getattr(spec, "raster_epsg", None) or getattr(spec, "epsg", None),
            )
        else:
            status(f"  all {len(raster_paths)} configured environmental raster layer(s) already present in CSV for {spec.dataset}; skipping observed-fix raster sampling")

    out = out.dropna(subset=["animal_id", "time", "x", "y"]).copy()
    out = apply_individual_metadata(out, spec)
    out = out.sort_values(["dataset", "taxon", "animal_id", "time"]).reset_index(drop=True)
    return add_step_columns(out)


def load_dataset(spec: DatasetSpec, verbose: bool = True) -> pd.DataFrame:
    """Load one dataset spec, including optional extra CSV files."""
    paths = [Path(spec.path)] + [Path(p) for p in spec.extra_paths]
    frames = []
    if verbose:
        status(f"Loading dataset {spec.dataset} ({spec.taxon}) from {len(paths)} file(s)")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if verbose:
            status(f"  reading {path.name}")
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_file"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    if verbose:
        status(f"  raw rows for {spec.dataset}: {len(raw):,}")
    out = standardize_track_table(raw, spec)
    if verbose:
        n_animals = out["animal_id"].nunique() if "animal_id" in out.columns else 0
        status(f"  standardized {spec.dataset}: {len(out):,} fixes, {n_animals:,} animals")
        if "task_settings_json" in out.columns:
            n_configured = int((out["task_settings_json"].astype(str).str.len() > 0).sum())
            if n_configured:
                status(f"  task settings configured on {n_configured:,} fixes")
        if {"sex", "age_class"}.issubset(out.columns):
            indiv = out.drop_duplicates(["dataset", "taxon", "animal_id"])
            sex_counts = indiv["sex"].value_counts(dropna=False).to_dict()
            age_counts = indiv["age_class"].value_counts(dropna=False).to_dict()
            status(f"  metadata individuals by sex: {sex_counts}")
            status(f"  metadata individuals by age: {age_counts}")
    return out


def load_datasets(specs: Iterable[DatasetSpec], verbose: bool = True) -> pd.DataFrame:
    """Load and concatenate multiple high-resolution tracking datasets."""
    specs = list(specs)
    if verbose:
        status(f"Loading {len(specs)} dataset specification(s)")
    parts = [load_dataset(spec, verbose=verbose) for spec in specs]
    if not parts:
        raise ValueError("No dataset specs were provided.")
    tracks = pd.concat(parts, ignore_index=True, sort=False)
    tracks = add_step_columns(tracks)
    if verbose:
        status(f"Combined tracks: {len(tracks):,} fixes, {tracks['animal_id'].nunique():,} animals")
    return tracks

def add_step_columns(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add previous/next movement columns used by descriptors."""
    df = tracks.copy()
    # Defensive normalization: callers may pass already-loaded data or combine
    # tables outside ``load_datasets``. This prevents tz-aware/tz-naive mixtures
    # from breaking the sort operation.
    df["time"] = normalize_time_column(df["time"], "combined_tracks")
    df["dataset"] = df["dataset"].astype(str)
    df["taxon"] = df["taxon"].astype(str)
    df["animal_id"] = df["animal_id"].astype(str)
    df = df.sort_values(["dataset", "taxon", "animal_id", "time"]).reset_index(drop=True)
    group_cols = ["dataset", "taxon", "animal_id"]
    for col in ["x", "y"]:
        df[f"prev_{col}"] = df.groupby(group_cols)[col].shift(1)
        df[f"next_{col}"] = df.groupby(group_cols)[col].shift(-1)
    df["prev_time"] = df.groupby(group_cols)["time"].shift(1)
    df["next_time"] = df.groupby(group_cols)["time"].shift(-1)
    dx = df["x"] - df["prev_x"]
    dy = df["y"] - df["prev_y"]
    df["step_m"] = np.sqrt(dx * dx + dy * dy)
    df["dt_min"] = (df["time"] - df["prev_time"]).dt.total_seconds() / 60.0
    return df


def infer_fine_interval_min(group: pd.DataFrame) -> float:
    """Infer the typical fix interval from a sorted animal track."""
    dt = group["time"].sort_values().diff().dt.total_seconds().dropna() / 60.0
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.empty:
        return np.nan
    return float(np.nanmedian(dt))
