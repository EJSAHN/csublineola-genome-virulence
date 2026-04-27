from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported table format: {file_path.suffix}")


def normalise_identifier(value: object) -> str:
    return str(value).strip()


def build_run_metadata(
    package_name: str,
    package_version: str,
    input_hashes: dict[str, str],
    extra_values: Iterable[tuple[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"field": "package_name", "value": package_name},
        {"field": "package_version", "value": package_version},
        {"field": "executed_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"field": "python_version", "value": platform.python_version()},
    ]
    for label, hash_value in input_hashes.items():
        rows.append({"field": f"{label}_sha256", "value": hash_value})
    for key, value in extra_values:
        rows.append({"field": key, "value": value})
    return pd.DataFrame(rows)
