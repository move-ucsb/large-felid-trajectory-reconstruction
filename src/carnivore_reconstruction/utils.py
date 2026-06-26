"""Small utilities shared across modules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def stable_hash01(value: Any) -> float:
    """Map any value to a deterministic float in [0, 1)."""
    s = str(value).encode("utf-8")
    h = hashlib.md5(s).hexdigest()[:12]
    return int(h, 16) / float(16**12)


def make_uid(*parts: Any) -> str:
    """Create a stable compact identifier from semantic parts."""
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    safe = "_".join(str(p).replace(" ", "-").replace("/", "-") for p in parts[:4])
    return f"{safe}_{digest}"


def finite_or(value: float, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a table with parquet when available, otherwise CSV fallback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return pd.read_csv(alt)
            raise
    return pd.read_csv(path)
