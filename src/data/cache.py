"""Local parquet cache for data pipeline artifacts."""

from pathlib import Path
from datetime import datetime

import pandas as pd


def _cache_path(data_dir: str, namespace: str, name: str) -> Path:
    p = Path(data_dir) / "processed" / namespace
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def _meta_path(data_dir: str, namespace: str, name: str) -> Path:
    p = Path(data_dir) / "processed" / namespace
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.meta"


def save(df: pd.DataFrame, data_dir: str, namespace: str, name: str) -> Path:
    path = _cache_path(data_dir, namespace, name)
    df.to_parquet(path)
    meta = _meta_path(data_dir, namespace, name)
    meta.write_text(datetime.utcnow().isoformat())
    return path


def load(data_dir: str, namespace: str, name: str) -> pd.DataFrame | None:
    path = _cache_path(data_dir, namespace, name)
    if path.exists():
        return pd.read_parquet(path)
    return None


def last_updated(data_dir: str, namespace: str, name: str) -> datetime | None:
    meta = _meta_path(data_dir, namespace, name)
    if meta.exists():
        return datetime.fromisoformat(meta.read_text().strip())
    return None


def is_stale(data_dir: str, namespace: str, name: str, max_age_hours: int = 24) -> bool:
    ts = last_updated(data_dir, namespace, name)
    if ts is None:
        return True
    age = datetime.utcnow() - ts
    return age.total_seconds() > max_age_hours * 3600
