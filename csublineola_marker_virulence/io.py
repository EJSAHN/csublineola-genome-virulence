from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio import SeqIO


CANONICAL_BASES = frozenset({"A", "C", "G", "T"})


def read_alignment(path: str | Path) -> tuple[list[str], list[str]]:
    """Read an equal-length FASTA alignment.

    Returns isolate identifiers and uppercase aligned sequences in input order.
    """
    alignment_path = Path(path)
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    records = list(SeqIO.parse(str(alignment_path), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records were found in: {alignment_path}")

    isolate_ids = [record.id.strip() for record in records]
    if any(not isolate_id for isolate_id in isolate_ids):
        raise ValueError("Every FASTA record must have a non-empty identifier.")
    if len(set(isolate_ids)) != len(isolate_ids):
        duplicates = sorted({x for x in isolate_ids if isolate_ids.count(x) > 1})
        raise ValueError(f"Duplicate FASTA identifiers: {', '.join(duplicates)}")

    sequences = [str(record.seq).upper() for record in records]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("All FASTA sequences must have the same aligned length.")
    return isolate_ids, sequences


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _find_column(columns: Iterable[str], requested: str, alternatives: tuple[str, ...]) -> str:
    mapping = {str(column).strip().lower(): str(column) for column in columns}
    candidates = (requested,) + alternatives
    for candidate in candidates:
        found = mapping.get(candidate.strip().lower())
        if found is not None:
            return found
    raise ValueError(
        f"Required column '{requested}' was not found. Available columns: "
        + ", ".join(map(str, columns))
    )


def read_virulence_table(
    path: str | Path,
    *,
    isolate_column: str = "isolate_id",
    pathotype_column: str = "pathotype",
) -> tuple[pd.DataFrame, str, str | None, list[str]]:
    """Read and validate a binary R/S host-differential response table."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Virulence table not found: {table_path}")

    table = _read_table(table_path)
    table.columns = [str(column).strip() for column in table.columns]
    isolate_col = _find_column(
        table.columns,
        isolate_column,
        ("isolate", "isolate id", "isolate_id", "strain", "taxa"),
    )

    pathotype_col: str | None = None
    for candidate in (pathotype_column, "pathotype", "pathotype label", "race"):
        matches = [column for column in table.columns if column.lower() == candidate.lower()]
        if matches:
            pathotype_col = matches[0]
            break

    table[isolate_col] = table[isolate_col].astype(str).str.strip()
    if table[isolate_col].eq("").any():
        raise ValueError("The virulence table contains empty isolate identifiers.")
    if table[isolate_col].duplicated().any():
        duplicates = sorted(table.loc[table[isolate_col].duplicated(False), isolate_col].unique())
        raise ValueError(f"Duplicate isolate identifiers in virulence table: {', '.join(duplicates)}")

    excluded = {isolate_col}
    if pathotype_col is not None:
        excluded.add(pathotype_col)
    host_columns = [column for column in table.columns if column not in excluded]
    if not host_columns:
        raise ValueError("No host-differential columns were found in the virulence table.")

    allowed = {"R", "S", "", "NA", "N/A", "NAN", "."}
    for column in host_columns:
        normalized = table[column].where(table[column].notna(), "").astype(str).str.strip().str.upper()
        invalid = sorted(set(normalized) - allowed)
        if invalid:
            raise ValueError(
                f"Column '{column}' contains values other than R, S, or missing: {invalid}"
            )
        normalized = normalized.replace({"NA": "", "N/A": "", "NAN": "", ".": ""})
        table[column] = normalized

    if pathotype_col is not None:
        table[pathotype_col] = table[pathotype_col].where(table[pathotype_col].notna(), "").astype(str).str.strip()

    return table, isolate_col, pathotype_col, host_columns
