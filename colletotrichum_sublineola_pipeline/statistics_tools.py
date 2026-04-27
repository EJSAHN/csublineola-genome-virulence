
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def lower_triangle_values(square_matrix: np.ndarray) -> np.ndarray:
    if square_matrix.shape[0] != square_matrix.shape[1]:
        raise ValueError("The input matrix must be square.")
    indices = np.tril_indices(square_matrix.shape[0], k=-1)
    return square_matrix[indices].astype(float)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 2:
        return float("nan")
    x = left_values[valid]
    y = right_values[valid]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denominator == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 2:
        return float("nan")
    left_ranks = pd.Series(left_values[valid]).rank(method="average").to_numpy(dtype=float)
    right_ranks = pd.Series(right_values[valid]).rank(method="average").to_numpy(dtype=float)
    return pearson_correlation(left_ranks, right_ranks)


def benjamini_hochberg(p_values: Iterable[float | int | None]) -> np.ndarray:
    p_array = np.asarray(list(p_values), dtype=float)
    result = np.full(p_array.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p_array)
    if not finite_mask.any():
        return result
    finite_values = p_array[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    adjusted = np.empty(count, dtype=float)
    cumulative = 1.0
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        value = ranked[reverse_index] * count / rank
        cumulative = min(cumulative, value)
        adjusted[reverse_index] = cumulative
    output = np.empty(count, dtype=float)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    result[finite_mask] = output
    return result


def permutation_pvalue(observed: float, null_distribution: np.ndarray, *, two_sided: bool = True) -> float:
    if np.isnan(observed):
        return float("nan")
    valid = np.asarray(null_distribution, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    if two_sided:
        extreme = np.abs(valid) >= abs(observed)
    else:
        extreme = valid >= observed
    return float((extreme.sum() + 1.0) / (valid.size + 1.0))


def mantel_permutation_test(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    *,
    permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    left = np.asarray(left_matrix, dtype=float)
    right = np.asarray(right_matrix, dtype=float)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("Both distance matrices must be square and have the same shape.")
    if left.shape[0] < 3:
        return pd.DataFrame(
            [
                {
                    "method": "pearson",
                    "observed_correlation": np.nan,
                    "permutation_pvalue_two_sided": np.nan,
                    "permutation_null_mean": np.nan,
                    "permutation_null_std": np.nan,
                    "permutation_null_q025": np.nan,
                    "permutation_null_q975": np.nan,
                    "permutations": permutations,
                },
                {
                    "method": "spearman",
                    "observed_correlation": np.nan,
                    "permutation_pvalue_two_sided": np.nan,
                    "permutation_null_mean": np.nan,
                    "permutation_null_std": np.nan,
                    "permutation_null_q025": np.nan,
                    "permutation_null_q975": np.nan,
                    "permutations": permutations,
                },
            ]
        )

    lower_indices = np.tril_indices(left.shape[0], k=-1)
    left_vector = left[lower_indices]
    right_vector = right[lower_indices]
    observed_pearson = pearson_correlation(left_vector, right_vector)
    observed_spearman = spearman_correlation(left_vector, right_vector)

    generator = np.random.default_rng(random_seed)
    pearson_null = np.empty(permutations, dtype=float)
    spearman_null = np.empty(permutations, dtype=float)
    for permutation_index in range(permutations):
        order = generator.permutation(left.shape[0])
        permuted_vector = right[np.ix_(order, order)][lower_indices]
        pearson_null[permutation_index] = pearson_correlation(left_vector, permuted_vector)
        spearman_null[permutation_index] = spearman_correlation(left_vector, permuted_vector)

    rows = []
    for method_name, observed_value, null_distribution in [
        ("pearson", observed_pearson, pearson_null),
        ("spearman", observed_spearman, spearman_null),
    ]:
        rows.append(
            {
                "method": method_name,
                "observed_correlation": observed_value,
                "permutation_pvalue_two_sided": permutation_pvalue(observed_value, null_distribution, two_sided=True),
                "permutation_null_mean": float(np.nanmean(null_distribution)),
                "permutation_null_std": float(np.nanstd(null_distribution, ddof=1)) if permutations > 1 else np.nan,
                "permutation_null_q025": float(np.nanquantile(null_distribution, 0.025)),
                "permutation_null_q975": float(np.nanquantile(null_distribution, 0.975)),
                "permutations": permutations,
            }
        )
    return pd.DataFrame(rows)


def label_separation_permutation_test(
    distance_matrix: np.ndarray,
    labels: Iterable[object],
    *,
    permutations: int,
    random_seed: int,
    label_name: str,
) -> pd.DataFrame:
    matrix = np.asarray(distance_matrix, dtype=float)
    labels_array = np.asarray([str(value) if value is not None and not pd.isna(value) else "" for value in labels], dtype=object)
    valid_mask = labels_array != ""
    matrix = matrix[np.ix_(valid_mask, valid_mask)]
    labels_array = labels_array[valid_mask]
    if matrix.shape[0] < 3:
        return pd.DataFrame(
            [
                {
                    "label_type": label_name,
                    "evaluated_isolates": matrix.shape[0],
                    "unique_labels": int(pd.Series(labels_array).nunique()),
                    "singleton_labels": int((pd.Series(labels_array).value_counts() == 1).sum()) if labels_array.size else 0,
                    "within_pair_count": 0,
                    "between_pair_count": 0,
                    "within_mean_genetic_distance": np.nan,
                    "between_mean_genetic_distance": np.nan,
                    "between_minus_within": np.nan,
                    "permutation_pvalue_two_sided": np.nan,
                    "permutation_pvalue_greater": np.nan,
                    "permutation_null_mean": np.nan,
                    "permutation_null_std": np.nan,
                    "permutation_null_q025": np.nan,
                    "permutation_null_q975": np.nan,
                    "permutations": permutations,
                }
            ]
        )
    lower_indices = np.tril_indices(matrix.shape[0], k=-1)
    vector = matrix[lower_indices]
    if vector.size == 0:
        within_count = 0
        between_count = 0
        within_mean = np.nan
        between_mean = np.nan
        observed = np.nan
        null_distribution = np.empty(permutations, dtype=float)
        null_distribution.fill(np.nan)
    else:
        same_mask = labels_array[lower_indices[0]] == labels_array[lower_indices[1]]
        within_values = vector[same_mask]
        between_values = vector[~same_mask]
        within_count = int(within_values.size)
        between_count = int(between_values.size)
        within_mean = float(np.nanmean(within_values)) if within_values.size else np.nan
        between_mean = float(np.nanmean(between_values)) if between_values.size else np.nan
        observed = between_mean - within_mean if within_values.size and between_values.size else np.nan

        generator = np.random.default_rng(random_seed)
        null_distribution = np.empty(permutations, dtype=float)
        for permutation_index in range(permutations):
            order = generator.permutation(labels_array.size)
            permuted_labels = labels_array[order]
            perm_same_mask = permuted_labels[lower_indices[0]] == permuted_labels[lower_indices[1]]
            perm_within = vector[perm_same_mask]
            perm_between = vector[~perm_same_mask]
            if perm_within.size == 0 or perm_between.size == 0:
                null_distribution[permutation_index] = np.nan
            else:
                null_distribution[permutation_index] = float(np.nanmean(perm_between) - np.nanmean(perm_within))

    value_counts = pd.Series(labels_array).value_counts() if labels_array.size else pd.Series(dtype=int)
    row = {
        "label_type": label_name,
        "evaluated_isolates": int(matrix.shape[0]),
        "unique_labels": int(value_counts.size),
        "singleton_labels": int((value_counts == 1).sum()) if not value_counts.empty else 0,
        "within_pair_count": within_count,
        "between_pair_count": between_count,
        "within_mean_genetic_distance": within_mean,
        "between_mean_genetic_distance": between_mean,
        "between_minus_within": observed,
        "permutation_pvalue_two_sided": permutation_pvalue(observed, null_distribution, two_sided=True),
        "permutation_pvalue_greater": permutation_pvalue(observed, null_distribution, two_sided=False),
        "permutation_null_mean": float(np.nanmean(null_distribution)) if np.isfinite(null_distribution).any() else np.nan,
        "permutation_null_std": float(np.nanstd(null_distribution, ddof=1)) if np.isfinite(null_distribution).sum() > 1 else np.nan,
        "permutation_null_q025": float(np.nanquantile(null_distribution[np.isfinite(null_distribution)], 0.025)) if np.isfinite(null_distribution).any() else np.nan,
        "permutation_null_q975": float(np.nanquantile(null_distribution[np.isfinite(null_distribution)], 0.975)) if np.isfinite(null_distribution).any() else np.nan,
        "permutations": permutations,
    }
    return pd.DataFrame([row])


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if total == 0:
        return float("nan")
    minimum = max(0, col1 - row2)
    maximum = min(col1, row1)

    def log_combination(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def table_probability(top_left: int) -> float:
        return math.exp(
            log_combination(row1, top_left)
            + log_combination(row2, col1 - top_left)
            - log_combination(total, col1)
        )

    observed_probability = table_probability(a)
    p_value = 0.0
    for candidate in range(minimum, maximum + 1):
        probability = table_probability(candidate)
        if probability <= observed_probability + 1e-12:
            p_value += probability
    return float(min(p_value, 1.0))
