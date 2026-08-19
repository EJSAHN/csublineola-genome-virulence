from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HAPMAP_META_COLUMNS = [
    "rs", "alleles", "chrom", "pos", "strand", "assembly", "center",
    "protLSID", "assayLSID", "panelLSID", "QCcode",
]


@dataclass(frozen=True)
class HapMapData:
    markers: pd.DataFrame
    sample_ids: list[str]
    dosage: np.ndarray
    recorded_missing: np.ndarray
    calculated_missing: np.ndarray
    maf: np.ndarray


@dataclass(frozen=True)
class VirulenceData:
    frame: pd.DataFrame
    isolate_ids: list[str]
    host_names: list[str]
    response: np.ndarray
    pathotypes: np.ndarray


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def read_hapmap(path: str | Path) -> HapMapData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HapMap file not found: {path}")

    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    missing_columns = [c for c in HAPMAP_META_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"HapMap file is missing metadata columns: {missing_columns}")
    if "No. of NN" not in frame.columns:
        raise ValueError("HapMap file must contain the 'No. of NN' column.")

    # Source HapMap contains one non-marker summary row at the end.
    frame = frame.loc[frame["rs"].notna() & frame["pos"].notna()].copy().reset_index(drop=True)
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(int)
    frame["No. of NN"] = pd.to_numeric(frame["No. of NN"], errors="raise").astype(int)

    start = frame.columns.get_loc("QCcode") + 1
    stop = frame.columns.get_loc("No. of NN")
    sample_ids = list(frame.columns[start:stop])
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample columns were found in the HapMap file.")
    if not sample_ids:
        raise ValueError("No sample genotype columns were detected.")

    n_markers = len(frame)
    n_samples = len(sample_ids)
    dosage = np.full((n_samples, n_markers), np.nan, dtype=np.float32)
    calculated_missing = np.zeros(n_markers, dtype=np.int16)
    maf = np.full(n_markers, np.nan, dtype=np.float64)
    genotype_matrix = frame[sample_ids].to_numpy(dtype=object)

    for marker_index, (allele_text, calls) in enumerate(
        zip(frame["alleles"].astype(str), genotype_matrix, strict=True)
    ):
        alleles = allele_text.upper().split("/")
        if len(alleles) != 2 or any(len(a) != 1 for a in alleles):
            calculated_missing[marker_index] = n_samples
            continue
        ref, alt = alleles
        observed_dosages: list[int] = []
        for sample_index, raw_call in enumerate(calls):
            call = _clean_text(raw_call).upper()
            if len(call) != 2 or "N" in call or any(base not in (ref, alt) for base in call):
                calculated_missing[marker_index] += 1
                continue
            alt_dosage = int(call[0] == alt) + int(call[1] == alt)
            dosage[sample_index, marker_index] = alt_dosage
            observed_dosages.append(alt_dosage)
        if observed_dosages:
            alternate_frequency = float(np.sum(observed_dosages)) / (2.0 * len(observed_dosages))
            maf[marker_index] = min(alternate_frequency, 1.0 - alternate_frequency)

    recorded_missing = frame["No. of NN"].to_numpy(dtype=np.int16)
    if not np.array_equal(recorded_missing, calculated_missing):
        mismatch = np.flatnonzero(recorded_missing != calculated_missing)
        examples = frame.loc[mismatch[:5], "rs"].tolist()
        raise ValueError(
            "Recorded and calculated missing-call counts differ for "
            f"{len(mismatch)} markers; examples: {examples}"
        )

    frame["calculated_missing_calls"] = calculated_missing
    frame["missing_fraction"] = calculated_missing / n_samples
    frame["calculated_maf"] = maf

    return HapMapData(
        markers=frame,
        sample_ids=sample_ids,
        dosage=dosage,
        recorded_missing=recorded_missing,
        calculated_missing=calculated_missing,
        maf=maf,
    )


def read_virulence(path: str | Path) -> VirulenceData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Virulence file not found: {path}")
    frame = pd.read_csv(path)
    required = {"isolate_id", "pathotype"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Virulence file must contain columns {sorted(required)}")
    host_names = [c for c in frame.columns if c not in required]
    if not host_names:
        raise ValueError("No host differential columns were found.")
    normalized = frame[host_names].apply(lambda column: column.astype(str).str.strip().str.upper().map({"R": 0, "S": 1}))
    if normalized.isna().any().any():
        bad = sorted(set(frame[host_names].stack().astype(str)) - {"R", "S", "r", "s"})
        raise ValueError(f"Virulence matrix contains unsupported values: {bad}")
    response = normalized.astype(np.int8).to_numpy()
    return VirulenceData(
        frame=frame,
        isolate_ids=frame["isolate_id"].astype(str).tolist(),
        host_names=host_names,
        response=response,
        pathotypes=frame["pathotype"].to_numpy(),
    )


def read_isolate_metadata(path: str | Path, expected_ids: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, dtype={"isolate_id": str, "origin": str, "site": str})
    expected = list(expected_ids)
    if frame["isolate_id"].duplicated().any():
        raise ValueError("Isolate metadata contains duplicate isolate IDs.")
    missing = sorted(set(expected) - set(frame["isolate_id"]))
    extra = sorted(set(frame["isolate_id"]) - set(expected))
    if missing or extra:
        raise ValueError(f"Isolate metadata mismatch. Missing={missing}; extra={extra}")
    return frame.set_index("isolate_id").loc[expected].reset_index()



def read_validation_panel_400(path: str | Path) -> dict[str, list[str]]:
    """Read a 400-marker validation panel as isolate -> genotype calls.

    HapMap is the publication-facing format. A legacy genotype-call text layout is
    accepted for provenance checks but is not required for routine analysis.
    """
    import re

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"400-marker validation panel not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith(">"):
        records: dict[str, list[str]] = {}
        for match in re.finditer(r">([^\r\n\t]+)[\r\n\t]+(.*?)(?=>|\Z)", text, flags=re.S):
            isolate = match.group(1).strip()
            calls = [
                value.strip().upper()
                for value in re.split(r"[\t\r\n]+", match.group(2))
                if value.strip()
            ]
            if len(calls) != 400:
                raise ValueError(
                    f"Validation-panel record {isolate} has {len(calls)} calls; expected 400."
                )
            if isolate in records:
                raise ValueError(f"Duplicate validation-panel isolate record: {isolate}")
            records[isolate] = calls
        if len(records) != 140:
            raise ValueError(
                f"Validation panel contains {len(records)} isolates; expected 140."
            )
        return records

    panel = read_hapmap(path)
    if len(panel.markers) != 400:
        raise ValueError(
            f"Validation HapMap contains {len(panel.markers)} markers; expected 400."
        )
    return {
        sample: panel.markers[sample].astype(str).str.strip().str.upper().tolist()
        for sample in panel.sample_ids
    }


def resolve_input_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
