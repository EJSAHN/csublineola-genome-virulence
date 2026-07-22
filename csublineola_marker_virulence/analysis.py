from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .io import CANONICAL_BASES


def alignment_summaries(
    isolate_ids: list[str],
    sequences: list[str],
    *,
    pathotype_map: dict[str, str] | None = None,
    virulence_counts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    array = np.asarray([list(sequence) for sequence in sequences], dtype="U1")
    isolate_count, position_count = array.shape
    pathotype_map = pathotype_map or {}
    count_lookup = {}
    if virulence_counts is not None and not virulence_counts.empty:
        count_lookup = virulence_counts.set_index("Isolate ID").to_dict("index")

    isolate_rows: list[dict[str, object]] = []
    for row_index, isolate_id in enumerate(isolate_ids):
        row = array[row_index]
        canonical_mask = np.isin(row, list(CANONICAL_BASES))
        canonical = row[canonical_mask]
        gap_count = int(np.sum(row == "-"))
        ambiguous_count = int(position_count - canonical_mask.sum() - gap_count)
        gc_fraction = (
            float(np.mean(np.isin(canonical, ["G", "C"]))) if canonical.size else np.nan
        )
        vir_counts = count_lookup.get(isolate_id, {})
        isolate_rows.append(
            {
                "Isolate ID": isolate_id,
                "Marker alignment positions": position_count,
                "Unambiguous nucleotide calls": int(canonical_mask.sum()),
                "Gap calls": gap_count,
                "Ambiguous calls": ambiguous_count,
                "Missing-call fraction": float(1.0 - canonical_mask.mean()),
                "GC fraction": gc_fraction,
                "Pathotype": pathotype_map.get(isolate_id, ""),
                "Susceptible differentials": vir_counts.get("Susceptible differentials", np.nan),
                "Resistant differentials": vir_counts.get("Resistant differentials", np.nan),
            }
        )

    position_rows: list[dict[str, object]] = []
    variable_count = 0
    informative_count = 0
    complete_count = 0
    for position_index in range(position_count):
        column = array[:, position_index]
        counts = {base: int(np.sum(column == base)) for base in "ACGT"}
        gap_count = int(np.sum(column == "-"))
        canonical_count = sum(counts.values())
        ambiguous_count = int(isolate_count - canonical_count - gap_count)
        observed = [base for base, count in counts.items() if count > 0]
        variable = len(observed) > 1
        informative = sum(count >= 2 for count in counts.values()) >= 2
        complete = canonical_count == isolate_count
        variable_count += int(variable)
        informative_count += int(informative)
        complete_count += int(complete)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        major = ranked[0][0] if ranked and ranked[0][1] > 0 else ""
        minor = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else ""
        minor_count = ranked[1][1] if minor else 0
        position_rows.append(
            {
                "Marker alignment position": position_index + 1,
                "Unambiguous calls": canonical_count,
                "A": counts["A"],
                "C": counts["C"],
                "G": counts["G"],
                "T": counts["T"],
                "Gaps": gap_count,
                "Ambiguous calls": ambiguous_count,
                "Observed nucleotide states": len(observed),
                "Variable position": variable,
                "Parsimony-informative position": informative,
                "Major nucleotide": major,
                "Minor nucleotide": minor,
                "Call rate": canonical_count / isolate_count,
                "Minor nucleotide count": minor_count,
                "Minor nucleotide frequency": minor_count / canonical_count if canonical_count else np.nan,
                "Missing fraction": 1.0 - canonical_count / isolate_count,
            }
        )

    summary = pd.DataFrame(
        [
            ["Marker dataset", "Isolates represented in the processed alignment", isolate_count],
            ["Marker dataset", "Marker alignment positions", position_count],
            ["Marker dataset", "Positions with complete calls", complete_count],
            ["Marker dataset", "Variable marker positions", variable_count],
            ["Marker dataset", "Parsimony-informative marker positions", informative_count],
        ],
        columns=["Category", "Metric", "Value"],
    )
    return summary, pd.DataFrame(isolate_rows), pd.DataFrame(position_rows)


def marker_distance_pairs(isolate_ids: list[str], sequences: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_index, right_index in combinations(range(len(isolate_ids)), 2):
        left = sequences[left_index]
        right = sequences[right_index]
        comparable = 0
        mismatches = 0
        for left_base, right_base in zip(left, right):
            if left_base in CANONICAL_BASES and right_base in CANONICAL_BASES:
                comparable += 1
                mismatches += int(left_base != right_base)
        distance = mismatches / comparable if comparable else np.nan
        rows.append(
            {
                "Isolate 1": isolate_ids[left_index],
                "Isolate 2": isolate_ids[right_index],
                "Comparable positions": comparable,
                "Mismatches": mismatches,
                "Marker distance": distance,
            }
        )
    return pd.DataFrame(rows)


def virulence_outputs(
    table: pd.DataFrame,
    isolate_column: str,
    pathotype_column: str | None,
    host_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = table[[isolate_column] + ([pathotype_column] if pathotype_column else []) + host_columns].copy()
    rename = {isolate_column: "Isolate ID"}
    if pathotype_column:
        rename[pathotype_column] = "Pathotype"
    profiles = profiles.rename(columns=rename)
    if "Pathotype" not in profiles.columns:
        profiles.insert(1, "Pathotype", "")

    count_rows = []
    for _, row in profiles.iterrows():
        calls = row[host_columns]
        susceptible = int((calls == "S").sum())
        resistant = int((calls == "R").sum())
        count_rows.append(
            {
                "Isolate ID": row["Isolate ID"],
                "Susceptible differentials": susceptible,
                "Resistant differentials": resistant,
            }
        )
    counts = pd.DataFrame(count_rows)

    pair_rows: list[dict[str, object]] = []
    for left_index, right_index in combinations(range(len(profiles)), 2):
        left = profiles.iloc[left_index]
        right = profiles.iloc[right_index]
        comparable = 0
        mismatches = 0
        for host in host_columns:
            left_call = left[host]
            right_call = right[host]
            if left_call in {"R", "S"} and right_call in {"R", "S"}:
                comparable += 1
                mismatches += int(left_call != right_call)
        pair_rows.append(
            {
                "Isolate 1": left["Isolate ID"],
                "Isolate 2": right["Isolate ID"],
                "Comparable host differentials": comparable,
                "Mismatched host responses": mismatches,
                "Virulence distance": mismatches / comparable if comparable else np.nan,
            }
        )
    pairs = pd.DataFrame(pair_rows)

    assignment_rows = []
    if profiles["Pathotype"].astype(str).str.strip().ne("").any():
        for pathotype, group in profiles.groupby("Pathotype", sort=False):
            assignment_rows.append(
                {
                    "Pathotype": pathotype,
                    "Number of isolates": len(group),
                    "Isolate IDs": ", ".join(group["Isolate ID"].astype(str)),
                }
            )
    assignments = pd.DataFrame(
        assignment_rows,
        columns=["Pathotype", "Number of isolates", "Isolate IDs"],
    )
    return profiles, counts, pairs, assignments


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def merge_marker_virulence_pairs(
    marker_pairs: pd.DataFrame,
    virulence_pairs: pd.DataFrame,
    pathotype_map: dict[str, str],
) -> pd.DataFrame:
    marker = marker_pairs.copy()
    virulence = virulence_pairs.copy()
    marker["_pair_key"] = [pair_key(a, b) for a, b in zip(marker["Isolate 1"], marker["Isolate 2"])]
    virulence["_pair_key"] = [pair_key(a, b) for a, b in zip(virulence["Isolate 1"], virulence["Isolate 2"])]
    merged = marker.merge(
        virulence[["_pair_key", "Comparable host differentials", "Mismatched host responses", "Virulence distance"]],
        on="_pair_key",
        how="inner",
    ).drop(columns="_pair_key")
    merged["Pathotype 1"] = merged["Isolate 1"].map(pathotype_map).fillna("")
    merged["Pathotype 2"] = merged["Isolate 2"].map(pathotype_map).fillna("")
    merged["Same pathotype"] = np.where(
        (merged["Pathotype 1"] != "") & (merged["Pathotype 1"] == merged["Pathotype 2"]),
        "Yes",
        "No",
    )
    return merged


def square_distance_matrix(
    isolate_order: list[str], pair_table: pd.DataFrame, value_column: str
) -> np.ndarray:
    index = {isolate_id: position for position, isolate_id in enumerate(isolate_order)}
    matrix = np.full((len(isolate_order), len(isolate_order)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    for _, row in pair_table.iterrows():
        left = row["Isolate 1"]
        right = row["Isolate 2"]
        if left in index and right in index:
            i = index[left]
            j = index[right]
            value = float(row[value_column])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    x = left[valid]
    y = right[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    x = pd.Series(left[valid]).rank(method="average").to_numpy(float)
    y = pd.Series(right[valid]).rank(method="average").to_numpy(float)
    return _pearson(x, y)


def matrix_correlation_test(
    marker_matrix: np.ndarray,
    virulence_matrix: np.ndarray,
    *,
    permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if marker_matrix.shape != virulence_matrix.shape:
        raise ValueError("Marker and virulence distance matrices must have the same shape.")
    lower = np.tril_indices(marker_matrix.shape[0], k=-1)
    marker_vector = marker_matrix[lower]
    virulence_vector = virulence_matrix[lower]
    observed = {
        "Pearson": _pearson(marker_vector, virulence_vector),
        "Spearman": _spearman(marker_vector, virulence_vector),
    }
    rng = np.random.default_rng(random_seed)
    null = {"Pearson": np.empty(permutations), "Spearman": np.empty(permutations)}
    for permutation_index in range(permutations):
        order = rng.permutation(marker_matrix.shape[0])
        permuted = virulence_matrix[np.ix_(order, order)][lower]
        null["Pearson"][permutation_index] = _pearson(marker_vector, permuted)
        null["Spearman"][permutation_index] = _spearman(marker_vector, permuted)

    rows = []
    for method in ("Pearson", "Spearman"):
        values = null[method]
        obs = observed[method]
        p_value = (np.sum(np.abs(values) >= abs(obs)) + 1) / (len(values) + 1)
        rows.append(
            {
                "Method": method,
                "Observed correlation": obs,
                "Permutation P value": p_value,
                "Null mean": float(np.mean(values)),
                "Null standard deviation": float(np.std(values, ddof=1)),
                "Null 2.5% quantile": float(np.quantile(values, 0.025)),
                "Null 97.5% quantile": float(np.quantile(values, 0.975)),
                "Permutations": permutations,
            }
        )
    return pd.DataFrame(rows)


def pathotype_separation_test(
    marker_matrix: np.ndarray,
    isolate_order: list[str],
    pathotype_map: dict[str, str],
    *,
    permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    labels = np.asarray([pathotype_map.get(isolate_id, "") for isolate_id in isolate_order], dtype=object)
    lower = np.tril_indices(len(isolate_order), k=-1)
    distances = marker_matrix[lower]
    same = labels[lower[0]] == labels[lower[1]]
    valid_labels = (labels[lower[0]] != "") & (labels[lower[1]] != "")
    same &= valid_labels
    between = (~same) & valid_labels
    within_values = distances[same]
    between_values = distances[between]
    within_mean = float(np.mean(within_values)) if len(within_values) else np.nan
    between_mean = float(np.mean(between_values)) if len(between_values) else np.nan
    observed = between_mean - within_mean

    rng = np.random.default_rng(random_seed)
    null = np.empty(permutations)
    for permutation_index in range(permutations):
        permuted = rng.permutation(labels)
        perm_same = (permuted[lower[0]] == permuted[lower[1]]) & valid_labels
        perm_between = (~perm_same) & valid_labels
        if not np.any(perm_same) or not np.any(perm_between):
            null[permutation_index] = np.nan
        else:
            null[permutation_index] = float(
                np.mean(distances[perm_between]) - np.mean(distances[perm_same])
            )
    finite = null[np.isfinite(null)]
    p_two = (np.sum(np.abs(finite) >= abs(observed)) + 1) / (len(finite) + 1)
    p_greater = (np.sum(finite >= observed) + 1) / (len(finite) + 1)
    counts = pd.Series(labels[labels != ""]).value_counts()
    return pd.DataFrame(
        [
            {
                "Label type": "Pathotype",
                "Evaluated isolates": len(isolate_order),
                "Unique pathotypes": int(len(counts)),
                "Singleton pathotypes": int((counts == 1).sum()),
                "Within-pathotype pairs": int(len(within_values)),
                "Between-pathotype pairs": int(len(between_values)),
                "Mean within-pathotype marker distance": within_mean,
                "Mean between-pathotype marker distance": between_mean,
                "Between-minus-within marker distance": observed,
                "Two-sided permutation P value": p_two,
                "Greater-tail permutation P value": p_greater,
                "Null mean": float(np.mean(finite)),
                "Null standard deviation": float(np.std(finite, ddof=1)),
                "Null 2.5% quantile": float(np.quantile(finite, 0.025)),
                "Null 97.5% quantile": float(np.quantile(finite, 0.975)),
                "Permutations": permutations,
            }
        ]
    )
