from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


CANONICAL_BASES = np.array(list("ACGT"), dtype="<U1")


@dataclass(frozen=True)
class AlignmentData:
    isolate_ids: list[str]
    sequences: np.ndarray
    alignment_length: int


def load_alignment(file_path: Path) -> AlignmentData:
    records = list(SeqIO.parse(file_path, "fasta"))
    if not records:
        raise ValueError("The alignment file does not contain any FASTA records.")

    isolate_ids = [record.id.split()[0].strip() for record in records]
    sequence_strings = [str(record.seq).upper() for record in records]
    sequence_lengths = {len(sequence) for sequence in sequence_strings}
    if len(sequence_lengths) != 1:
        raise ValueError("The alignment file must contain sequences of equal length.")

    alignment_length = sequence_lengths.pop()
    sequence_array = np.array([list(sequence) for sequence in sequence_strings], dtype="<U1")
    return AlignmentData(
        isolate_ids=isolate_ids,
        sequences=sequence_array,
        alignment_length=alignment_length,
    )


def _canonical_mask(sequence_array: np.ndarray) -> np.ndarray:
    return np.isin(sequence_array, CANONICAL_BASES)


def _gap_mask(sequence_array: np.ndarray) -> np.ndarray:
    return sequence_array == "-"


def build_alignment_summary(alignment_data: AlignmentData) -> pd.DataFrame:
    sequence_array = alignment_data.sequences
    canonical_mask = _canonical_mask(sequence_array)
    gap_mask = _gap_mask(sequence_array)
    ambiguous_mask = ~canonical_mask & ~gap_mask

    base_counts = np.stack([(sequence_array == base).sum(axis=0) for base in CANONICAL_BASES], axis=1)
    canonical_states_observed = (base_counts > 0).sum(axis=1)
    variable_canonical = canonical_states_observed > 1
    parsimony_informative = (base_counts >= 2).sum(axis=1) >= 2

    summary_rows = [
        {"metric": "isolate_count", "value": len(alignment_data.isolate_ids)},
        {"metric": "alignment_length", "value": alignment_data.alignment_length},
        {"metric": "complete_case_sites", "value": int(np.all(canonical_mask, axis=0).sum())},
        {"metric": "sites_with_gaps", "value": int(np.any(gap_mask, axis=0).sum())},
        {"metric": "sites_with_ambiguous_calls", "value": int(np.any(ambiguous_mask, axis=0).sum())},
        {"metric": "variable_canonical_sites", "value": int(variable_canonical.sum())},
        {"metric": "parsimony_informative_sites", "value": int(parsimony_informative.sum())},
    ]
    return pd.DataFrame(summary_rows)


def build_isolate_summary(alignment_data: AlignmentData) -> pd.DataFrame:
    sequence_array = alignment_data.sequences
    canonical_mask = _canonical_mask(sequence_array)
    gap_mask = _gap_mask(sequence_array)
    ambiguous_mask = ~canonical_mask & ~gap_mask

    canonical_counts = canonical_mask.sum(axis=1)
    gc_counts = (sequence_array == "G").sum(axis=1) + (sequence_array == "C").sum(axis=1)

    dataframe = pd.DataFrame(
        {
            "isolate_id": alignment_data.isolate_ids,
            "alignment_length": alignment_data.alignment_length,
            "canonical_bases": canonical_counts,
            "gap_characters": gap_mask.sum(axis=1),
            "ambiguous_characters": ambiguous_mask.sum(axis=1),
            "missing_fraction": np.divide(
                gap_mask.sum(axis=1) + ambiguous_mask.sum(axis=1),
                alignment_data.alignment_length,
                out=np.zeros(len(alignment_data.isolate_ids), dtype=float),
                where=alignment_data.alignment_length > 0,
            ),
            "gc_fraction_among_canonical": np.divide(
                gc_counts,
                canonical_counts,
                out=np.full(len(alignment_data.isolate_ids), np.nan, dtype=float),
                where=canonical_counts > 0,
            ),
        }
    )
    return dataframe.sort_values("isolate_id").reset_index(drop=True)


def build_site_summary(alignment_data: AlignmentData) -> pd.DataFrame:
    sequence_array = alignment_data.sequences
    canonical_mask = _canonical_mask(sequence_array)
    gap_mask = _gap_mask(sequence_array)
    ambiguous_mask = ~canonical_mask & ~gap_mask

    base_counts = {base: (sequence_array == base).sum(axis=0) for base in CANONICAL_BASES}
    base_count_matrix = np.stack([base_counts[base] for base in CANONICAL_BASES], axis=1)
    comparable_isolates = canonical_mask.sum(axis=0)
    gap_count = gap_mask.sum(axis=0)
    ambiguous_count = ambiguous_mask.sum(axis=0)

    canonical_states_observed = (base_count_matrix > 0).sum(axis=1)
    variable_canonical = canonical_states_observed > 1
    parsimony_informative = (base_count_matrix >= 2).sum(axis=1) >= 2

    major_index = base_count_matrix.argmax(axis=1)
    major_base = CANONICAL_BASES[major_index]
    major_base = np.where(comparable_isolates > 0, major_base, "")

    sorted_counts = np.sort(base_count_matrix, axis=1)
    minor_count = sorted_counts[:, -2]
    minor_index = np.argsort(base_count_matrix, axis=1)[:, -2]
    minor_base = CANONICAL_BASES[minor_index]
    minor_base = np.where(minor_count > 0, minor_base, "")

    dataframe = pd.DataFrame(
        {
            "site_number": np.arange(1, alignment_data.alignment_length + 1),
            "comparable_isolates": comparable_isolates,
            "A_count": base_counts["A"],
            "C_count": base_counts["C"],
            "G_count": base_counts["G"],
            "T_count": base_counts["T"],
            "gap_count": gap_count,
            "ambiguous_count": ambiguous_count,
            "canonical_states_observed": canonical_states_observed,
            "variable_canonical": variable_canonical,
            "parsimony_informative": parsimony_informative,
            "major_base": major_base,
            "minor_base": minor_base,
            "gc_fraction_among_comparable": np.divide(
                base_counts["G"] + base_counts["C"],
                comparable_isolates,
                out=np.full(alignment_data.alignment_length, np.nan, dtype=float),
                where=comparable_isolates > 0,
            ),
        }
    )
    return dataframe


def build_pairwise_distance_outputs(alignment_data: AlignmentData) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_array = alignment_data.sequences
    canonical_mask = _canonical_mask(sequence_array)
    isolate_ids = alignment_data.isolate_ids
    isolate_count = len(isolate_ids)

    distance_matrix = np.full((isolate_count, isolate_count), np.nan, dtype=float)
    comparable_matrix = np.zeros((isolate_count, isolate_count), dtype=int)
    mismatch_matrix = np.zeros((isolate_count, isolate_count), dtype=int)

    for index in range(isolate_count):
        distance_matrix[index, index] = 0.0
        comparable_matrix[index, index] = int(canonical_mask[index].sum())

    for left_index in range(isolate_count):
        for right_index in range(left_index + 1, isolate_count):
            comparable = canonical_mask[left_index] & canonical_mask[right_index]
            comparable_sites = int(comparable.sum())
            mismatches = int(np.count_nonzero(sequence_array[left_index, comparable] != sequence_array[right_index, comparable]))
            distance = np.nan if comparable_sites == 0 else mismatches / comparable_sites

            comparable_matrix[left_index, right_index] = comparable_sites
            comparable_matrix[right_index, left_index] = comparable_sites
            mismatch_matrix[left_index, right_index] = mismatches
            mismatch_matrix[right_index, left_index] = mismatches
            distance_matrix[left_index, right_index] = distance
            distance_matrix[right_index, left_index] = distance

    matrix_dataframe = pd.DataFrame(distance_matrix, index=isolate_ids, columns=isolate_ids)
    matrix_dataframe.index.name = "isolate_id"

    long_rows: list[dict[str, object]] = []
    for left_index, left_isolate in enumerate(isolate_ids):
        for right_index in range(left_index + 1, isolate_count):
            long_rows.append(
                {
                    "isolate_id_left": left_isolate,
                    "isolate_id_right": isolate_ids[right_index],
                    "comparable_sites": comparable_matrix[left_index, right_index],
                    "mismatches": mismatch_matrix[left_index, right_index],
                    "genetic_distance": distance_matrix[left_index, right_index],
                }
            )
    long_dataframe = pd.DataFrame(long_rows)
    return matrix_dataframe, long_dataframe
