from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from .distance import upper_triangle


@dataclass(frozen=True)
class CorrelationPermutationResult:
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    pearson_null: np.ndarray
    spearman_null: np.ndarray


@dataclass(frozen=True)
class PathotypeSeparationResult:
    within_n: int
    between_n: int
    within_mean: float
    between_mean: float
    difference: float
    p_value: float
    null: np.ndarray


def _two_sided_permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))


def matrix_correlation_permutation(
    marker_distance: np.ndarray,
    virulence_distance: np.ndarray,
    reps: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> CorrelationPermutationResult:
    marker_distance = np.asarray(marker_distance, dtype=float)
    virulence_distance = np.asarray(virulence_distance, dtype=float)
    if marker_distance.shape != virulence_distance.shape:
        raise ValueError("Distance matrices must have the same shape.")
    triangle = np.triu_indices_from(marker_distance, k=1)
    marker_vector = marker_distance[triangle]
    virulence_vector = virulence_distance[triangle]
    pearson_r = float(stats.pearsonr(marker_vector, virulence_vector).statistic)
    spearman_rho = float(stats.spearmanr(marker_vector, virulence_vector).statistic)
    pearson_null = np.empty(reps, dtype=float)
    spearman_null = np.empty(reps, dtype=float)
    rng = np.random.default_rng(seed)
    n = marker_distance.shape[0]

    group_indices: list[np.ndarray] | None = None
    if groups is not None:
        group_array = np.asarray(groups)
        if len(group_array) != n:
            raise ValueError("groups length must equal matrix size.")
        group_indices = [np.flatnonzero(group_array == value) for value in np.unique(group_array)]

    for replicate in range(reps):
        if group_indices is None:
            permutation = rng.permutation(n)
        else:
            permutation = np.arange(n)
            for indices in group_indices:
                permutation[indices] = rng.permutation(indices)
        permuted = virulence_distance[np.ix_(permutation, permutation)][triangle]
        pearson_null[replicate] = np.corrcoef(marker_vector, permuted)[0, 1]
        spearman_null[replicate] = stats.spearmanr(marker_vector, permuted).statistic

    return CorrelationPermutationResult(
        pearson_r=pearson_r,
        pearson_p=_two_sided_permutation_p(pearson_r, pearson_null),
        spearman_rho=spearman_rho,
        spearman_p=_two_sided_permutation_p(spearman_rho, spearman_null),
        pearson_null=pearson_null,
        spearman_null=spearman_null,
    )


def pathotype_separation_permutation(
    marker_distance: np.ndarray,
    pathotypes: Sequence[object],
    reps: int,
    seed: int,
) -> PathotypeSeparationResult:
    marker_distance = np.asarray(marker_distance, dtype=float)
    labels = np.asarray(pathotypes)
    triangle = np.triu_indices_from(marker_distance, k=1)
    values = marker_distance[triangle]
    same = labels[triangle[0]] == labels[triangle[1]]
    if same.sum() == 0:
        raise ValueError("No within-pathotype pairs are available.")
    within_mean = float(values[same].mean())
    between_mean = float(values[~same].mean())
    difference = between_mean - within_mean
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for replicate in range(reps):
        permuted_labels = rng.permutation(labels)
        permuted_same = permuted_labels[triangle[0]] == permuted_labels[triangle[1]]
        null[replicate] = values[~permuted_same].mean() - values[permuted_same].mean()
    return PathotypeSeparationResult(
        within_n=int(same.sum()),
        between_n=int((~same).sum()),
        within_mean=within_mean,
        between_mean=between_mean,
        difference=difference,
        p_value=_two_sided_permutation_p(difference, null),
        null=null,
    )


def permanova_pseudo_f(distance: np.ndarray, groups: Sequence[str]) -> float:
    """One-way PERMANOVA pseudo-F computed from squared distances."""
    distance = np.asarray(distance, dtype=float)
    groups_array = np.asarray(groups)
    n = len(groups_array)
    unique = np.unique(groups_array)
    if len(unique) < 2:
        raise ValueError("PERMANOVA requires at least two groups.")
    total_ss = float(np.sum(upper_triangle(distance) ** 2) / n)
    within_ss = 0.0
    for group in unique:
        indices = np.flatnonzero(groups_array == group)
        if len(indices) < 2:
            continue
        within = distance[np.ix_(indices, indices)]
        within_ss += float(np.sum(upper_triangle(within) ** 2) / len(indices))
    between_ss = total_ss - within_ss
    return float((between_ss / (len(unique) - 1)) / (within_ss / (n - len(unique))))


def permanova_permutation(
    distance: np.ndarray,
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    groups_array = np.asarray(groups)
    observed = permanova_pseudo_f(distance, groups_array)
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for replicate in range(reps):
        null[replicate] = permanova_pseudo_f(distance, rng.permutation(groups_array))
    p_value = float((1 + np.sum(null >= observed)) / (reps + 1))
    return observed, p_value, null


def marker_panel_metrics(distance: np.ndarray, selected_indices: Sequence[int]) -> np.ndarray:
    selected = np.sort(np.asarray(selected_indices, dtype=int))
    unselected = np.setdiff1d(np.arange(distance.shape[0]), selected)
    within = distance[np.ix_(selected, selected)]
    within_vector = upper_triangle(within)
    nearest = distance[np.ix_(unselected, selected)].min(axis=1)
    return np.array(
        [
            within_vector.mean(),
            np.median(within_vector),
            nearest.mean(),
            np.quantile(nearest, 0.95),
            nearest.max(),
            np.median(nearest),
        ],
        dtype=float,
    )


def panel_randomization_audit(
    distance: np.ndarray,
    selected_indices: Sequence[int],
    reps: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=float)
    selected = np.asarray(selected_indices, dtype=int)
    observed = marker_panel_metrics(distance, selected)
    null = np.empty((reps, len(observed)), dtype=float)
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    k = len(selected)

    if groups is None:
        for replicate in range(reps):
            null[replicate] = marker_panel_metrics(distance, rng.choice(n, k, replace=False))
    else:
        group_array = np.asarray(groups)
        unique = np.unique(group_array)
        counts = {group: int(np.sum(group_array[selected] == group)) for group in unique}
        pools = {group: np.flatnonzero(group_array == group) for group in unique}
        for group in unique:
            if counts[group] > len(pools[group]):
                raise ValueError(f"Cannot sample {counts[group]} isolates from group {group}.")
        for replicate in range(reps):
            random_selected = np.concatenate(
                [rng.choice(pools[group], counts[group], replace=False) for group in unique if counts[group] > 0]
            )
            null[replicate] = marker_panel_metrics(distance, random_selected)

    percentile = np.mean(null <= observed, axis=0)
    directional_p = np.empty(len(observed), dtype=float)
    # Higher is preferable for within-panel diversity; lower is preferable for coverage distances.
    directional_p[:2] = (1 + np.sum(null[:, :2] >= observed[:2], axis=0)) / (reps + 1)
    directional_p[2:] = (1 + np.sum(null[:, 2:] <= observed[2:], axis=0)) / (reps + 1)
    return observed, null, percentile, directional_p


def _macro_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = y_true.shape[0]
    polymorphic = np.flatnonzero((y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < n))
    per_host = np.empty(len(polymorphic), dtype=float)
    for index, host in enumerate(polymorphic):
        observed = y_true[:, host]
        predicted = y_pred[:, host]
        specificity = np.mean(predicted[observed == 0] == 0)
        sensitivity = np.mean(predicted[observed == 1] == 1)
        per_host[index] = 0.5 * (specificity + sensitivity)
    return float(per_host.mean()), per_host, polymorphic


def knn_predictions(distance: np.ndarray, response: np.ndarray, k: int) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    response = np.asarray(response, dtype=np.int8)
    n = distance.shape[0]
    if k < 1 or k >= n or k % 2 == 0:
        raise ValueError("k must be an odd integer between 1 and n-1.")
    neighbor_order = np.argsort(distance + np.eye(n) * 1e9, axis=1, kind="stable")
    return (response[neighbor_order[:, :k]].mean(axis=1) > 0.5).astype(np.int8)


def knn_summary(distance: np.ndarray, response: np.ndarray, k_values: Sequence[int]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for k in k_values:
        prediction = knn_predictions(distance, response, k)
        balanced, per_host, polymorphic = _macro_balanced_accuracy(response, prediction)
        summaries.append(
            {
                "k": int(k),
                "micro_accuracy": float(np.mean(prediction == response)),
                "macro_balanced_accuracy": balanced,
                "exact_profile_accuracy": float(np.mean(np.all(prediction == response, axis=1))),
                "per_host_balanced_accuracy": per_host,
                "polymorphic_host_indices": polymorphic,
            }
        )
    return summaries


def knn_permutation_test(
    distance: np.ndarray,
    response: np.ndarray,
    k: int,
    reps: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> tuple[float, float, np.ndarray]:
    response = np.asarray(response, dtype=np.int8)
    observed_prediction = knn_predictions(distance, response, k)
    observed, _, _ = _macro_balanced_accuracy(response, observed_prediction)
    n = response.shape[0]
    neighbor_order = np.argsort(np.asarray(distance) + np.eye(n) * 1e9, axis=1, kind="stable")
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)

    group_indices: list[np.ndarray] | None = None
    if groups is not None:
        group_array = np.asarray(groups)
        group_indices = [np.flatnonzero(group_array == group) for group in np.unique(group_array)]

    for replicate in range(reps):
        if group_indices is None:
            permuted_response = response[rng.permutation(n)]
        else:
            permutation = np.arange(n)
            for indices in group_indices:
                permutation[indices] = rng.permutation(indices)
            permuted_response = response[permutation]
        prediction = (permuted_response[neighbor_order[:, :k]].mean(axis=1) > 0.5).astype(np.int8)
        null[replicate] = _macro_balanced_accuracy(permuted_response, prediction)[0]

    p_value = float((1 + np.sum(null >= observed)) / (reps + 1))
    return observed, p_value, null


def majority_baseline(response: np.ndarray) -> dict[str, float]:
    response = np.asarray(response, dtype=np.int8)
    n = response.shape[0]
    column_sums = response.sum(axis=0)
    prediction = np.zeros_like(response)
    for isolate in range(n):
        prediction[isolate] = (((column_sums - response[isolate]) / (n - 1)) > 0.5).astype(np.int8)
    balanced, _, _ = _macro_balanced_accuracy(response, prediction)
    return {
        "micro_accuracy": float(np.mean(prediction == response)),
        "macro_balanced_accuracy": balanced,
        "exact_profile_accuracy": float(np.mean(np.all(prediction == response, axis=1))),
    }


def permanova_summary(
    distance: np.ndarray,
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """One-way PERMANOVA with pseudo-F, R², and unrestricted label permutations."""
    distance = np.asarray(distance, dtype=float)
    group_array = np.asarray(groups)
    n = len(group_array)
    levels = np.unique(group_array)
    if len(levels) < 2:
        raise ValueError("PERMANOVA requires at least two groups")

    def components(labels: np.ndarray) -> tuple[float, float, float, float]:
        total_ss = float(np.sum(upper_triangle(distance) ** 2) / n)
        within_ss = 0.0
        for level in np.unique(labels):
            indices = np.flatnonzero(labels == level)
            if len(indices) < 2:
                continue
            within = distance[np.ix_(indices, indices)]
            within_ss += float(np.sum(upper_triangle(within) ** 2) / len(indices))
        between_ss = total_ss - within_ss
        pseudo_f = float((between_ss / (len(levels) - 1)) / (within_ss / (n - len(levels))))
        r2 = float(between_ss / total_ss) if total_ss > 0 else float("nan")
        return pseudo_f, r2, between_ss, within_ss

    observed_f, observed_r2, ss_between, ss_within = components(group_array)
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for replicate in range(reps):
        null[replicate] = components(rng.permutation(group_array))[0]
    p_value = float((1 + np.sum(null >= observed_f)) / (reps + 1))
    return (
        {
            "pseudo_F": observed_f,
            "R2": observed_r2,
            "SS_between": ss_between,
            "SS_within": ss_within,
            "p_value": p_value,
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "permutations": int(reps),
        },
        null,
    )


def _pcoa_real_imaginary_coordinates(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return real and imaginary PCoA coordinates for a dissimilarity matrix."""
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > 1e-12
    negative = eigenvalues < -1e-12
    real = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive]) if np.any(positive) else np.zeros((n, 0))
    imaginary = eigenvectors[:, negative] * np.sqrt(-eigenvalues[negative]) if np.any(negative) else np.zeros((n, 0))
    return real, imaginary


def permdisp_summary(
    distance: np.ndarray,
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Centroid-based PERMDISP with Anderson's correction for negative axes.

    Group labels are permuted, and group centroids are recomputed in the fixed
    principal-coordinate space for every null draw.
    """
    distance = np.asarray(distance, dtype=float)
    group_array = np.asarray(groups)
    levels = np.unique(group_array)
    n = len(group_array)
    if len(levels) < 2:
        raise ValueError("PERMDISP requires at least two groups")
    real, imaginary = _pcoa_real_imaginary_coordinates(distance)

    def distances_to_centroids(labels: np.ndarray) -> np.ndarray:
        result = np.empty(n, dtype=float)
        for level in np.unique(labels):
            indices = np.flatnonzero(labels == level)
            real_delta = real[indices] - real[indices].mean(axis=0) if real.shape[1] else np.zeros((len(indices), 0))
            imag_delta = imaginary[indices] - imaginary[indices].mean(axis=0) if imaginary.shape[1] else np.zeros((len(indices), 0))
            real_sq = np.sum(real_delta ** 2, axis=1)
            imag_sq = np.sum(imag_delta ** 2, axis=1)
            # Equation 3 of Anderson (2006); negative values are assigned zero,
            # matching PERMDISP2/vegan behavior for non-Euclidean distances.
            result[indices] = np.sqrt(np.maximum(real_sq - imag_sq, 0.0))
        return result

    def anova_f(labels: np.ndarray) -> tuple[float, np.ndarray]:
        values = distances_to_centroids(labels)
        grand = float(values.mean())
        ss_between = 0.0
        ss_within = 0.0
        for level in np.unique(labels):
            group_values = values[labels == level]
            mean = float(group_values.mean())
            ss_between += len(group_values) * (mean - grand) ** 2
            ss_within += float(np.sum((group_values - mean) ** 2))
        numerator = ss_between / (len(levels) - 1)
        denominator = ss_within / (n - len(levels))
        return float(numerator / denominator), values

    observed_f, observed_distances = anova_f(group_array)
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for replicate in range(reps):
        null[replicate] = anova_f(rng.permutation(group_array))[0]
    p_value = float((1 + np.sum(null >= observed_f)) / (reps + 1))
    return (
        {
            "F": observed_f,
            "p_value": p_value,
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "groups": int(len(levels)),
            "samples": int(n),
            "permutations": int(reps),
        },
        null,
        observed_distances,
    )


def origin_distance_contrast_permutation(
    distance: np.ndarray,
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Compare mean between-origin and within-origin pairwise distances."""
    distance = np.asarray(distance, dtype=float)
    group_array = np.asarray(groups)
    triangle = np.triu_indices(len(group_array), k=1)
    values = distance[triangle]
    same = group_array[triangle[0]] == group_array[triangle[1]]
    within_mean = float(values[same].mean())
    between_mean = float(values[~same].mean())
    observed = between_mean - within_mean
    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for replicate in range(reps):
        permuted = rng.permutation(group_array)
        permuted_same = permuted[triangle[0]] == permuted[triangle[1]]
        null[replicate] = float(values[~permuted_same].mean() - values[permuted_same].mean())
    p_value = float((1 + np.sum(null >= observed)) / (reps + 1))
    return (
        {
            "within_origin_mean": within_mean,
            "between_origin_mean": between_mean,
            "between_minus_within": observed,
            "within_pair_count": int(same.sum()),
            "between_pair_count": int((~same).sum()),
            "p_value": p_value,
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "permutations": int(reps),
        },
        null,
    )
