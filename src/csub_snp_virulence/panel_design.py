from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd


def panel_metrics(distance: np.ndarray, selected_indices: Sequence[int]) -> dict[str, float]:
    """Summarize diversity within a panel and coverage of unselected isolates."""
    distance = np.asarray(distance, dtype=float)
    selected = np.asarray(sorted(set(int(i) for i in selected_indices)), dtype=int)
    if len(selected) < 2:
        raise ValueError("At least two selected isolates are required.")
    all_indices = np.arange(distance.shape[0])
    unselected = np.setdiff1d(all_indices, selected)
    tri = np.triu_indices(len(selected), k=1)
    within = distance[np.ix_(selected, selected)][tri]
    result = {
        "panel_size": int(len(selected)),
        "mean_within_distance": float(within.mean()),
        "median_within_distance": float(np.median(within)),
    }
    if len(unselected) == 0:
        result.update(
            mean_unselected_nearest_distance=0.0,
            median_unselected_nearest_distance=0.0,
            p95_unselected_nearest_distance=0.0,
            max_unselected_nearest_distance=0.0,
        )
    else:
        nearest = distance[np.ix_(unselected, selected)].min(axis=1)
        result.update(
            mean_unselected_nearest_distance=float(nearest.mean()),
            median_unselected_nearest_distance=float(np.median(nearest)),
            p95_unselected_nearest_distance=float(np.quantile(nearest, 0.95)),
            max_unselected_nearest_distance=float(nearest.max()),
        )
    return result


def _candidate_priority(
    distance: np.ndarray,
    selected: list[int],
    candidates: np.ndarray,
    sample_ids: Sequence[str],
) -> int:
    """Farthest-first choice with deterministic coverage-oriented tie breaking."""
    nearest = distance[np.ix_(candidates, np.asarray(selected, dtype=int))].min(axis=1)
    best_value = nearest.max()
    tied = candidates[np.isclose(nearest, best_value, rtol=0.0, atol=1e-12)]
    if len(tied) == 1:
        return int(tied[0])
    # Prefer the candidate with the greater mean distance to the current panel,
    # then use isolate ID for deterministic resolution of any remaining tie.
    mean_to_panel = distance[np.ix_(tied, np.asarray(selected, dtype=int))].mean(axis=1)
    best_mean = mean_to_panel.max()
    tied2 = tied[np.isclose(mean_to_panel, best_mean, rtol=0.0, atol=1e-12)]
    return int(sorted(tied2, key=lambda i: str(sample_ids[int(i)]))[0])


def farthest_first_panel(
    distance: np.ndarray,
    sample_ids: Sequence[str],
    panel_size: int,
    *,
    start_index: int,
    groups: Sequence[str] | None = None,
    quotas: dict[str, int] | None = None,
) -> list[int]:
    """Construct one deterministic farthest-first panel.

    When quotas are supplied, no group can exceed its requested final count.
    """
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    if panel_size < 2 or panel_size > n:
        raise ValueError("panel_size must be between 2 and the number of isolates")
    selected = [int(start_index)]
    all_indices = np.arange(n)
    group_array = np.asarray(groups) if groups is not None else None
    remaining_quota: Counter[str] | None = None
    if quotas is not None:
        if group_array is None:
            raise ValueError("groups are required when quotas are supplied")
        if sum(quotas.values()) != panel_size:
            raise ValueError("quota counts must sum to panel_size")
        remaining_quota = Counter({str(k): int(v) for k, v in quotas.items()})
        start_group = str(group_array[start_index])
        if remaining_quota[start_group] <= 0:
            raise ValueError("start isolate belongs to a group with zero quota")
        remaining_quota[start_group] -= 1

    while len(selected) < panel_size:
        candidates = np.setdiff1d(all_indices, np.asarray(selected, dtype=int))
        if remaining_quota is not None:
            candidates = np.asarray(
                [i for i in candidates if remaining_quota[str(group_array[i])] > 0],
                dtype=int,
            )
        if len(candidates) == 0:
            raise RuntimeError("No eligible candidate remains before the panel is complete")
        choice = _candidate_priority(distance, selected, candidates, sample_ids)
        selected.append(choice)
        if remaining_quota is not None:
            remaining_quota[str(group_array[choice])] -= 1
    return selected


def _coverage_score(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    """Lexicographic score: minimize worst/tail/mean gaps, then maximize diversity."""
    return (
        metrics["max_unselected_nearest_distance"],
        metrics["p95_unselected_nearest_distance"],
        metrics["mean_unselected_nearest_distance"],
        -metrics["mean_within_distance"],
        -metrics["median_within_distance"],
    )


def optimize_farthest_first_panel(
    distance: np.ndarray,
    sample_ids: Sequence[str],
    panel_size: int,
    *,
    groups: Sequence[str] | None = None,
    quotas: dict[str, int] | None = None,
) -> tuple[list[int], dict[str, float]]:
    """Try every eligible starting isolate and retain the best coverage panel."""
    group_array = np.asarray(groups) if groups is not None else None
    starts = list(range(distance.shape[0]))
    if quotas is not None:
        starts = [i for i in starts if quotas.get(str(group_array[i]), 0) > 0]
    best_panel: list[int] | None = None
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    for start in starts:
        panel = farthest_first_panel(
            distance,
            sample_ids,
            panel_size,
            start_index=start,
            groups=groups,
            quotas=quotas,
        )
        metrics = panel_metrics(distance, panel)
        score = _coverage_score(metrics)
        if best_score is None or score < best_score:
            best_score = score
            best_panel = panel
            best_metrics = metrics
        elif score == best_score and best_panel is not None:
            current_ids = tuple(sorted(str(sample_ids[i]) for i in panel))
            best_ids = tuple(sorted(str(sample_ids[i]) for i in best_panel))
            if current_ids < best_ids:
                best_panel = panel
                best_metrics = metrics
    if best_panel is None or best_metrics is None:
        raise RuntimeError("Panel optimization failed")
    return best_panel, best_metrics


def augment_panel_farthest_first(
    distance: np.ndarray,
    sample_ids: Sequence[str],
    initial_indices: Sequence[int],
    final_size: int,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """Sequentially augment an existing panel and record every design step."""
    selected = list(dict.fromkeys(int(i) for i in initial_indices))
    if final_size < len(selected) or final_size > distance.shape[0]:
        raise ValueError("final_size must be between the initial panel size and n")
    metric_rows = [{"addition_number": 0, **panel_metrics(distance, selected)}]
    additions: list[dict[str, object]] = []
    all_indices = np.arange(distance.shape[0])
    while len(selected) < final_size:
        candidates = np.setdiff1d(all_indices, np.asarray(selected, dtype=int))
        nearest_before = distance[np.ix_(candidates, np.asarray(selected, dtype=int))].min(axis=1)
        choice = _candidate_priority(distance, selected, candidates, sample_ids)
        choice_position = int(np.flatnonzero(candidates == choice)[0])
        additions.append(
            {
                "addition_number": len(selected) - len(initial_indices) + 1,
                "panel_size_after_addition": len(selected) + 1,
                "isolate_id": str(sample_ids[choice]),
                "distance_to_panel_before_addition": float(nearest_before[choice_position]),
            }
        )
        selected.append(choice)
        metric_rows.append(
            {
                "addition_number": len(selected) - len(initial_indices),
                **panel_metrics(distance, selected),
            }
        )
    return selected, pd.DataFrame(additions), pd.DataFrame(metric_rows)


def panel_membership_table(
    sample_ids: Sequence[str],
    metadata: pd.DataFrame,
    selected_indices: Sequence[int],
    scenario: str,
) -> pd.DataFrame:
    selected = list(selected_indices)
    rows = []
    indexed = metadata.set_index("isolate_id")
    for rank, index in enumerate(selected, start=1):
        isolate = str(sample_ids[index])
        meta = indexed.loc[isolate]
        rows.append(
            {
                "scenario": scenario,
                "selection_order": rank,
                "isolate_id": isolate,
                "origin": meta["origin"],
                "site": meta["site"],
                "collection_year": meta["collection_year"],
            }
        )
    return pd.DataFrame(rows)
