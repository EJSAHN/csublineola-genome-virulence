
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .alignment_processing import AlignmentData
from .statistics_tools import benjamini_hochberg, fisher_exact_two_sided, label_separation_permutation_test


CANONICAL_BASES = np.array(list("ACGT"), dtype="<U1")


def _virulence_columns(virulence_profiles: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in virulence_profiles.columns:
        if column in {"isolate_id", "pathotype"}:
            continue
        series = virulence_profiles[column].dropna().astype(str).str.upper().str.strip()
        if not series.empty and set(series.unique()).issubset({"R", "S"}):
            columns.append(column)
    return columns


def _profile_signature_map(virulence_profiles: pd.DataFrame) -> dict[str, str]:
    virulence_columns = _virulence_columns(virulence_profiles)
    signatures = virulence_profiles[virulence_columns].apply(lambda row: "|".join(row.astype(str)), axis=1)
    return dict(zip(virulence_profiles["isolate_id"], signatures))


def build_analysis_cohorts(
    alignment_isolates: list[str],
    virulence_profiles: pd.DataFrame,
) -> pd.DataFrame:
    virulence_isolates = virulence_profiles["isolate_id"].astype(str).tolist()
    overlap = sorted(set(alignment_isolates).intersection(virulence_isolates))
    alignment_only = sorted(set(alignment_isolates).difference(overlap))
    virulence_only = sorted(set(virulence_isolates).difference(overlap))
    rows = [
        {"cohort": "alignment_total", "isolate_count": len(alignment_isolates)},
        {"cohort": "virulence_total", "isolate_count": len(virulence_isolates)},
        {"cohort": "overlap_total", "isolate_count": len(overlap)},
        {"cohort": "alignment_only", "isolate_count": len(alignment_only)},
        {"cohort": "virulence_only", "isolate_count": len(virulence_only)},
        {
            "cohort": "pathotype_count",
            "isolate_count": int(virulence_profiles["pathotype"].nunique()) if "pathotype" in virulence_profiles.columns else np.nan,
        },
    ]
    return pd.DataFrame(rows)


def build_sample_qc(
    isolate_summary: pd.DataFrame,
    virulence_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    dataframe = isolate_summary.copy()
    if virulence_profiles is not None and not virulence_profiles.empty:
        profile_map = _profile_signature_map(virulence_profiles)
        dataframe["has_virulence_profile"] = dataframe["isolate_id"].isin(virulence_profiles["isolate_id"])
        dataframe["profile_signature"] = dataframe["isolate_id"].map(profile_map)
        if "pathotype" in virulence_profiles.columns:
            dataframe["pathotype"] = dataframe["isolate_id"].map(
                virulence_profiles.set_index("isolate_id")["pathotype"].to_dict()
            )
            dataframe["pathotype_isolate_count"] = dataframe["pathotype"].map(
                virulence_profiles.groupby("pathotype", dropna=False)["isolate_id"].size().to_dict()
            )
    else:
        dataframe["has_virulence_profile"] = False
        dataframe["profile_signature"] = pd.NA
    dataframe["canonical_fraction"] = 1.0 - dataframe["missing_fraction"]
    dataframe["missing_fraction_rank_desc"] = dataframe["missing_fraction"].rank(method="min", ascending=False)
    dataframe["ambiguous_rank_desc"] = dataframe["ambiguous_characters"].rank(method="min", ascending=False)
    dataframe["gap_rank_desc"] = dataframe["gap_characters"].rank(method="min", ascending=False)
    dataframe["gc_fraction_rank_asc"] = dataframe["gc_fraction_among_canonical"].rank(method="min", ascending=True)
    return dataframe.sort_values(
        ["missing_fraction", "ambiguous_characters", "gap_characters", "isolate_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_site_qc(site_summary: pd.DataFrame) -> pd.DataFrame:
    dataframe = site_summary.copy()
    total_isolates = dataframe["comparable_isolates"] + dataframe["gap_count"] + dataframe["ambiguous_count"]
    dataframe["total_isolates"] = total_isolates
    dataframe["call_rate"] = np.divide(
        dataframe["comparable_isolates"],
        total_isolates,
        out=np.zeros(len(dataframe), dtype=float),
        where=total_isolates > 0,
    )
    base_matrix = dataframe[["A_count", "C_count", "G_count", "T_count"]].to_numpy(dtype=float)
    sorted_counts = np.sort(base_matrix, axis=1)
    minor_counts = sorted_counts[:, -2]
    dataframe["minor_allele_count"] = minor_counts
    dataframe["minor_allele_frequency"] = np.divide(
        minor_counts,
        dataframe["comparable_isolates"],
        out=np.full(len(dataframe), np.nan, dtype=float),
        where=dataframe["comparable_isolates"].to_numpy(dtype=float) > 0,
    )
    dataframe["site_type"] = np.where(
        dataframe["parsimony_informative"],
        "parsimony_informative",
        np.where(dataframe["variable_canonical"], "variable", "invariant"),
    )
    dataframe["missing_fraction"] = np.divide(
        dataframe["gap_count"] + dataframe["ambiguous_count"],
        total_isolates,
        out=np.zeros(len(dataframe), dtype=float),
        where=total_isolates > 0,
    )
    return dataframe.sort_values(
        ["site_type", "minor_allele_frequency", "site_number"], ascending=[True, False, True]
    ).reset_index(drop=True)


def build_binary_distance_summary(distance_relationship: pd.DataFrame, column_name: str, label: str) -> pd.DataFrame:
    if distance_relationship.empty or column_name not in distance_relationship.columns:
        return pd.DataFrame(
            columns=[
                label,
                "pair_count",
                "mean_genetic_distance",
                "median_genetic_distance",
                "min_genetic_distance",
                "max_genetic_distance",
                "mean_virulence_distance",
                "median_virulence_distance",
                "min_virulence_distance",
                "max_virulence_distance",
            ]
        )
    grouped = (
        distance_relationship.groupby(column_name, dropna=False)
        .agg(
            pair_count=("genetic_distance", "size"),
            mean_genetic_distance=("genetic_distance", "mean"),
            median_genetic_distance=("genetic_distance", "median"),
            min_genetic_distance=("genetic_distance", "min"),
            max_genetic_distance=("genetic_distance", "max"),
            mean_virulence_distance=("virulence_distance", "mean"),
            median_virulence_distance=("virulence_distance", "median"),
            min_virulence_distance=("virulence_distance", "min"),
            max_virulence_distance=("virulence_distance", "max"),
        )
        .reset_index()
        .rename(columns={column_name: label})
    )
    return grouped


def build_pair_class_summary(distance_relationship: pd.DataFrame) -> pd.DataFrame:
    if distance_relationship.empty:
        return pd.DataFrame(
            columns=[
                "same_pathotype",
                "same_virulence_profile",
                "pair_count",
                "mean_genetic_distance",
                "median_genetic_distance",
                "min_genetic_distance",
                "max_genetic_distance",
                "mean_virulence_distance",
                "median_virulence_distance",
                "min_virulence_distance",
                "max_virulence_distance",
            ]
        )

    grouped = (
        distance_relationship.groupby(["same_pathotype", "same_virulence_profile"], dropna=False)
        .agg(
            pair_count=("genetic_distance", "size"),
            mean_genetic_distance=("genetic_distance", "mean"),
            median_genetic_distance=("genetic_distance", "median"),
            min_genetic_distance=("genetic_distance", "min"),
            max_genetic_distance=("genetic_distance", "max"),
            mean_virulence_distance=("virulence_distance", "mean"),
            median_virulence_distance=("virulence_distance", "median"),
            min_virulence_distance=("virulence_distance", "min"),
            max_virulence_distance=("virulence_distance", "max"),
        )
        .reset_index()
    )
    return grouped


def build_pathotype_pair_summary(distance_relationship: pd.DataFrame) -> pd.DataFrame:
    if distance_relationship.empty or {"pathotype_left", "pathotype_right"}.difference(distance_relationship.columns):
        return pd.DataFrame(
            columns=[
                "pathotype_a",
                "pathotype_b",
                "same_pathotype_pair",
                "pair_count",
                "mean_genetic_distance",
                "median_genetic_distance",
                "mean_virulence_distance",
                "median_virulence_distance",
            ]
        )

    dataframe = distance_relationship.copy()
    pathotype_a = []
    pathotype_b = []
    for left, right in zip(dataframe["pathotype_left"], dataframe["pathotype_right"]):
        left_value = "" if pd.isna(left) else str(left)
        right_value = "" if pd.isna(right) else str(right)
        ordered = sorted([left_value, right_value])
        pathotype_a.append(ordered[0])
        pathotype_b.append(ordered[1])
    dataframe["pathotype_a"] = pathotype_a
    dataframe["pathotype_b"] = pathotype_b
    dataframe["same_pathotype_pair"] = dataframe["pathotype_a"].eq(dataframe["pathotype_b"])

    grouped = (
        dataframe.groupby(["pathotype_a", "pathotype_b", "same_pathotype_pair"], dropna=False)
        .agg(
            pair_count=("genetic_distance", "size"),
            mean_genetic_distance=("genetic_distance", "mean"),
            median_genetic_distance=("genetic_distance", "median"),
            mean_virulence_distance=("virulence_distance", "mean"),
            median_virulence_distance=("virulence_distance", "median"),
        )
        .reset_index()
        .sort_values(
            ["same_pathotype_pair", "pair_count", "pathotype_a", "pathotype_b"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    return grouped


def build_nearest_neighbor_summary(
    genetic_distance_pairs: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
    virulence_distance_pairs: pd.DataFrame,
) -> pd.DataFrame:
    if genetic_distance_pairs.empty:
        return pd.DataFrame(
            columns=[
                "isolate_id",
                "pathotype",
                "nearest_neighbor_isolate",
                "nearest_neighbor_pathotype",
                "nearest_genetic_distance",
                "virulence_distance_to_nearest_neighbor",
                "same_pathotype",
                "same_virulence_profile",
            ]
        )

    forward = genetic_distance_pairs.rename(
        columns={
            "isolate_id_left": "isolate_id",
            "isolate_id_right": "neighbor_isolate",
        }
    )
    reverse = genetic_distance_pairs.rename(
        columns={
            "isolate_id_right": "isolate_id",
            "isolate_id_left": "neighbor_isolate",
        }
    )
    bidirectional = pd.concat([forward, reverse], ignore_index=True)

    virulence_forward = virulence_distance_pairs.rename(
        columns={
            "isolate_id_left": "isolate_id",
            "isolate_id_right": "neighbor_isolate",
            "virulence_distance": "virulence_distance_to_neighbor",
        }
    )
    virulence_reverse = virulence_distance_pairs.rename(
        columns={
            "isolate_id_right": "isolate_id",
            "isolate_id_left": "neighbor_isolate",
            "virulence_distance": "virulence_distance_to_neighbor",
        }
    )
    virulence_bidirectional = pd.concat([virulence_forward, virulence_reverse], ignore_index=True)

    metadata = virulence_profiles[["isolate_id"]].copy()
    metadata["pathotype"] = virulence_profiles["pathotype"] if "pathotype" in virulence_profiles.columns else pd.NA
    profile_map = _profile_signature_map(virulence_profiles)
    metadata["profile_signature"] = metadata["isolate_id"].map(profile_map)

    bidirectional = bidirectional.merge(metadata, on="isolate_id", how="left")
    bidirectional = bidirectional.merge(
        metadata.rename(
            columns={
                "isolate_id": "neighbor_isolate",
                "pathotype": "neighbor_pathotype",
                "profile_signature": "neighbor_profile_signature",
            }
        ),
        on="neighbor_isolate",
        how="left",
    )
    bidirectional = bidirectional.merge(
        virulence_bidirectional[["isolate_id", "neighbor_isolate", "virulence_distance_to_neighbor"]],
        on=["isolate_id", "neighbor_isolate"],
        how="left",
    )
    bidirectional["same_pathotype"] = bidirectional["pathotype"].eq(bidirectional["neighbor_pathotype"])
    bidirectional["same_virulence_profile"] = bidirectional["profile_signature"].eq(bidirectional["neighbor_profile_signature"])

    ranked = bidirectional.sort_values(
        ["isolate_id", "genetic_distance", "neighbor_isolate"], ascending=[True, True, True]
    )
    nearest = ranked.groupby("isolate_id", as_index=False).first()
    nearest = nearest.rename(
        columns={
            "neighbor_isolate": "nearest_neighbor_isolate",
            "neighbor_pathotype": "nearest_neighbor_pathotype",
            "genetic_distance": "nearest_genetic_distance",
            "virulence_distance_to_neighbor": "virulence_distance_to_nearest_neighbor",
        }
    )
    return nearest[
        [
            "isolate_id",
            "pathotype",
            "nearest_neighbor_isolate",
            "nearest_neighbor_pathotype",
            "nearest_genetic_distance",
            "virulence_distance_to_nearest_neighbor",
            "same_pathotype",
            "same_virulence_profile",
        ]
    ].sort_values("isolate_id").reset_index(drop=True)


def build_host_separation_summary(
    distance_relationship: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if distance_relationship.empty or virulence_profiles.empty:
        return pd.DataFrame(
            columns=[
                "host_differential",
                "comparable_pairs",
                "same_response_pairs",
                "different_response_pairs",
                "mean_genetic_distance_same_response",
                "mean_genetic_distance_different_response",
                "distance_difference_different_minus_same",
            ]
        )

    host_columns = _virulence_columns(virulence_profiles)
    response_map = virulence_profiles.set_index("isolate_id")[host_columns].astype(str)
    rows: list[dict[str, object]] = []
    for host_column in host_columns:
        pair_frame = distance_relationship[["isolate_id_left", "isolate_id_right", "genetic_distance"]].copy()
        left_response = pair_frame["isolate_id_left"].map(response_map[host_column])
        right_response = pair_frame["isolate_id_right"].map(response_map[host_column])
        valid = left_response.isin(["R", "S"]) & right_response.isin(["R", "S"])
        pair_frame = pair_frame.loc[valid].copy()
        if pair_frame.empty:
            rows.append(
                {
                    "host_differential": host_column,
                    "comparable_pairs": 0,
                    "same_response_pairs": 0,
                    "different_response_pairs": 0,
                    "mean_genetic_distance_same_response": np.nan,
                    "mean_genetic_distance_different_response": np.nan,
                    "distance_difference_different_minus_same": np.nan,
                }
            )
            continue
        pair_frame["same_response"] = left_response.loc[valid].to_numpy() == right_response.loc[valid].to_numpy()
        same_distances = pair_frame.loc[pair_frame["same_response"], "genetic_distance"]
        different_distances = pair_frame.loc[~pair_frame["same_response"], "genetic_distance"]
        rows.append(
            {
                "host_differential": host_column,
                "comparable_pairs": int(len(pair_frame)),
                "same_response_pairs": int(pair_frame["same_response"].sum()),
                "different_response_pairs": int((~pair_frame["same_response"]).sum()),
                "mean_genetic_distance_same_response": same_distances.mean() if not same_distances.empty else np.nan,
                "mean_genetic_distance_different_response": different_distances.mean() if not different_distances.empty else np.nan,
                "distance_difference_different_minus_same": (
                    different_distances.mean() - same_distances.mean()
                    if not same_distances.empty and not different_distances.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "distance_difference_different_minus_same", ascending=False, na_position="last"
    ).reset_index(drop=True)


def build_site_association_by_host(
    alignment_data: AlignmentData,
    virulence_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if virulence_profiles.empty:
        return pd.DataFrame(
            columns=[
                "host_differential",
                "site_number",
                "comparable_isolates",
                "susceptible_isolates",
                "resistant_isolates",
                "canonical_states_observed",
                "susceptible_major_base",
                "resistant_major_base",
                "susceptible_major_fraction",
                "resistant_major_fraction",
                "allele_divergence",
                "rank_within_host",
            ]
        )

    host_columns = _virulence_columns(virulence_profiles)
    isolate_index = {isolate_id: index for index, isolate_id in enumerate(alignment_data.isolate_ids)}
    sequences = alignment_data.sequences

    rows: list[dict[str, object]] = []
    profile_frame = virulence_profiles.set_index("isolate_id")
    for host_column in host_columns:
        host_rows = profile_frame[host_column].dropna().astype(str).str.upper().str.strip()
        overlap_isolates = [
            isolate_id for isolate_id in host_rows.index
            if isolate_id in isolate_index and host_rows.loc[isolate_id] in {"R", "S"}
        ]
        if not overlap_isolates:
            continue
        indices = np.array([isolate_index[isolate_id] for isolate_id in overlap_isolates], dtype=int)
        responses = host_rows.loc[overlap_isolates].to_numpy(dtype="<U1")
        subset_sequences = sequences[indices, :]
        valid_mask = np.isin(subset_sequences, CANONICAL_BASES)
        resistant_mask = responses == "R"
        susceptible_mask = responses == "S"

        host_records: list[dict[str, object]] = []
        for site_index in range(subset_sequences.shape[1]):
            site_calls = subset_sequences[:, site_index]
            site_valid = valid_mask[:, site_index]
            resistant_valid = site_valid & resistant_mask
            susceptible_valid = site_valid & susceptible_mask

            resistant_count = int(resistant_valid.sum())
            susceptible_count = int(susceptible_valid.sum())
            comparable_isolates = resistant_count + susceptible_count
            if comparable_isolates == 0:
                continue

            resistant_counts = np.array([(site_calls[resistant_valid] == base).sum() for base in CANONICAL_BASES], dtype=float)
            susceptible_counts = np.array([(site_calls[susceptible_valid] == base).sum() for base in CANONICAL_BASES], dtype=float)
            total_counts = resistant_counts + susceptible_counts
            states_observed = int((total_counts > 0).sum())

            resistant_frequencies = resistant_counts / resistant_count if resistant_count > 0 else np.zeros(len(CANONICAL_BASES), dtype=float)
            susceptible_frequencies = susceptible_counts / susceptible_count if susceptible_count > 0 else np.zeros(len(CANONICAL_BASES), dtype=float)
            allele_divergence = 0.5 * float(np.abs(resistant_frequencies - susceptible_frequencies).sum())

            resistant_major_index = int(resistant_counts.argmax()) if resistant_count > 0 else 0
            susceptible_major_index = int(susceptible_counts.argmax()) if susceptible_count > 0 else 0

            resistant_major_fraction = float(resistant_frequencies[resistant_major_index]) if resistant_count > 0 else np.nan
            susceptible_major_fraction = float(susceptible_frequencies[susceptible_major_index]) if susceptible_count > 0 else np.nan

            host_records.append(
                {
                    "host_differential": host_column,
                    "site_number": site_index + 1,
                    "comparable_isolates": comparable_isolates,
                    "susceptible_isolates": susceptible_count,
                    "resistant_isolates": resistant_count,
                    "canonical_states_observed": states_observed,
                    "susceptible_major_base": str(CANONICAL_BASES[susceptible_major_index]) if susceptible_count > 0 else "",
                    "resistant_major_base": str(CANONICAL_BASES[resistant_major_index]) if resistant_count > 0 else "",
                    "susceptible_major_fraction": susceptible_major_fraction,
                    "resistant_major_fraction": resistant_major_fraction,
                    "allele_divergence": allele_divergence,
                }
            )

        host_frame = pd.DataFrame(host_records)
        if host_frame.empty:
            continue
        host_frame = host_frame.sort_values(
            ["allele_divergence", "canonical_states_observed", "comparable_isolates", "site_number"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        host_frame["rank_within_host"] = np.arange(1, len(host_frame) + 1)
        rows.append(host_frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_top_site_associations(site_association_by_host: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if site_association_by_host.empty:
        return pd.DataFrame(columns=site_association_by_host.columns)
    return (
        site_association_by_host.loc[site_association_by_host["rank_within_host"] <= top_n]
        .sort_values(["host_differential", "rank_within_host", "site_number"])
        .reset_index(drop=True)
    )


def build_pathotype_support_summary(virulence_profiles: pd.DataFrame) -> pd.DataFrame:
    if virulence_profiles.empty or "pathotype" not in virulence_profiles.columns:
        return pd.DataFrame(columns=["pathotype", "isolate_count", "singleton_flag", "isolate_ids"])
    summary = (
        virulence_profiles.groupby("pathotype", dropna=False)
        .agg(
            isolate_count=("isolate_id", "size"),
            isolate_ids=("isolate_id", lambda values: ", ".join(sorted(values))),
        )
        .reset_index()
        .sort_values(["isolate_count", "pathotype"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary["singleton_flag"] = summary["isolate_count"] == 1
    summary["repeated_flag"] = summary["isolate_count"] > 1
    return summary


def build_host_separation_tests(
    genetic_distance_matrix: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
    *,
    permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if virulence_profiles.empty:
        return pd.DataFrame(
            columns=[
                "host_differential",
                "evaluated_isolates",
                "same_response_pairs",
                "different_response_pairs",
                "mean_genetic_distance_same_response",
                "mean_genetic_distance_different_response",
                "between_minus_within",
                "permutation_pvalue_two_sided",
                "permutation_pvalue_greater",
                "permutation_null_mean",
                "permutation_null_std",
                "host_fdr_bh",
            ]
        )

    matrix = genetic_distance_matrix.copy()
    if "isolate_id" in matrix.columns:
        matrix = matrix.set_index("isolate_id")
    matrix = matrix.astype(float)

    host_columns = _virulence_columns(virulence_profiles)
    profile_frame = virulence_profiles.set_index("isolate_id")
    rows: list[dict[str, object]] = []

    for host_index, host_column in enumerate(host_columns):
        responses = profile_frame[host_column].dropna().astype(str).str.upper().str.strip()
        overlap = [isolate_id for isolate_id in responses.index if isolate_id in matrix.index and responses.loc[isolate_id] in {"R", "S"}]
        if len(overlap) < 3:
            rows.append(
                {
                    "host_differential": host_column,
                    "evaluated_isolates": len(overlap),
                    "same_response_pairs": 0,
                    "different_response_pairs": 0,
                    "mean_genetic_distance_same_response": np.nan,
                    "mean_genetic_distance_different_response": np.nan,
                    "between_minus_within": np.nan,
                    "permutation_pvalue_two_sided": np.nan,
                    "permutation_pvalue_greater": np.nan,
                    "permutation_null_mean": np.nan,
                    "permutation_null_std": np.nan,
                }
            )
            continue

        submatrix = matrix.loc[overlap, overlap].to_numpy(dtype=float)
        labels = responses.loc[overlap].to_numpy(dtype=object)
        result = label_separation_permutation_test(
            submatrix,
            labels,
            permutations=permutations,
            random_seed=random_seed + host_index,
            label_name=host_column,
        ).iloc[0].to_dict()
        rows.append(
            {
                "host_differential": host_column,
                "evaluated_isolates": int(result["evaluated_isolates"]),
                "same_response_pairs": int(result["within_pair_count"]),
                "different_response_pairs": int(result["between_pair_count"]),
                "mean_genetic_distance_same_response": result["within_mean_genetic_distance"],
                "mean_genetic_distance_different_response": result["between_mean_genetic_distance"],
                "between_minus_within": result["between_minus_within"],
                "permutation_pvalue_two_sided": result["permutation_pvalue_two_sided"],
                "permutation_pvalue_greater": result["permutation_pvalue_greater"],
                "permutation_null_mean": result["permutation_null_mean"],
                "permutation_null_std": result["permutation_null_std"],
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["host_fdr_bh"] = benjamini_hochberg(dataframe["permutation_pvalue_two_sided"].tolist())
    return dataframe.sort_values(
        ["host_fdr_bh", "permutation_pvalue_two_sided", "between_minus_within", "host_differential"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_site_association_by_host_exact(
    alignment_data: AlignmentData,
    virulence_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if virulence_profiles.empty:
        return pd.DataFrame(
            columns=[
                "host_differential",
                "site_number",
                "evaluated_isolates",
                "susceptible_isolates",
                "resistant_isolates",
                "major_allele",
                "minor_allele",
                "susceptible_minor_count",
                "susceptible_major_count",
                "resistant_minor_count",
                "resistant_major_count",
                "susceptible_minor_frequency",
                "resistant_minor_frequency",
                "minor_frequency_difference_susceptible_minus_resistant",
                "odds_ratio_haldane",
                "fisher_pvalue_two_sided",
                "host_fdr_bh",
                "global_fdr_bh",
            ]
        )

    isolate_index = {isolate_id: index for index, isolate_id in enumerate(alignment_data.isolate_ids)}
    sequences = alignment_data.sequences
    profile_frame = virulence_profiles.set_index("isolate_id")
    host_columns = _virulence_columns(virulence_profiles)

    result_frames: list[pd.DataFrame] = []
    for host_column in host_columns:
        host_responses = profile_frame[host_column].dropna().astype(str).str.upper().str.strip()
        overlap = [
            isolate_id for isolate_id in host_responses.index
            if isolate_id in isolate_index and host_responses.loc[isolate_id] in {"R", "S"}
        ]
        if len(overlap) < 4:
            continue
        indices = np.array([isolate_index[isolate_id] for isolate_id in overlap], dtype=int)
        responses = host_responses.loc[overlap].to_numpy(dtype="<U1")
        subset_sequences = sequences[indices, :]
        valid_mask = np.isin(subset_sequences, CANONICAL_BASES)
        host_rows: list[dict[str, object]] = []
        for site_index in range(subset_sequences.shape[1]):
            site_calls = subset_sequences[:, site_index]
            valid = valid_mask[:, site_index]
            site_calls = site_calls[valid]
            site_responses = responses[valid]
            if site_calls.size < 4:
                continue
            allele_counter = Counter(site_calls.tolist())
            if len(allele_counter) != 2:
                continue
            major_allele, minor_allele = [item[0] for item in allele_counter.most_common(2)]
            susceptible_minor = int(np.sum((site_responses == "S") & (site_calls == minor_allele)))
            susceptible_major = int(np.sum((site_responses == "S") & (site_calls == major_allele)))
            resistant_minor = int(np.sum((site_responses == "R") & (site_calls == minor_allele)))
            resistant_major = int(np.sum((site_responses == "R") & (site_calls == major_allele)))

            susceptible_total = susceptible_minor + susceptible_major
            resistant_total = resistant_minor + resistant_major
            if susceptible_total == 0 or resistant_total == 0:
                continue

            susceptible_minor_frequency = susceptible_minor / susceptible_total
            resistant_minor_frequency = resistant_minor / resistant_total
            frequency_difference = susceptible_minor_frequency - resistant_minor_frequency
            odds_ratio = ((susceptible_minor + 0.5) * (resistant_major + 0.5)) / ((susceptible_major + 0.5) * (resistant_minor + 0.5))
            p_value = fisher_exact_two_sided(
                susceptible_minor,
                susceptible_major,
                resistant_minor,
                resistant_major,
            )
            host_rows.append(
                {
                    "host_differential": host_column,
                    "site_number": site_index + 1,
                    "evaluated_isolates": int(site_calls.size),
                    "susceptible_isolates": susceptible_total,
                    "resistant_isolates": resistant_total,
                    "major_allele": major_allele,
                    "minor_allele": minor_allele,
                    "susceptible_minor_count": susceptible_minor,
                    "susceptible_major_count": susceptible_major,
                    "resistant_minor_count": resistant_minor,
                    "resistant_major_count": resistant_major,
                    "susceptible_minor_frequency": susceptible_minor_frequency,
                    "resistant_minor_frequency": resistant_minor_frequency,
                    "minor_frequency_difference_susceptible_minus_resistant": frequency_difference,
                    "odds_ratio_haldane": odds_ratio,
                    "fisher_pvalue_two_sided": p_value,
                }
            )
        if not host_rows:
            continue
        host_frame = pd.DataFrame(host_rows)
        host_frame["host_fdr_bh"] = benjamini_hochberg(host_frame["fisher_pvalue_two_sided"].tolist())
        result_frames.append(host_frame)

    if not result_frames:
        return pd.DataFrame()

    dataframe = pd.concat(result_frames, ignore_index=True)
    dataframe["global_fdr_bh"] = benjamini_hochberg(dataframe["fisher_pvalue_two_sided"].tolist())
    return dataframe.sort_values(
        ["global_fdr_bh", "host_fdr_bh", "fisher_pvalue_two_sided", "host_differential", "site_number"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_top_site_associations_exact(site_association_by_host: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if site_association_by_host.empty:
        return pd.DataFrame(columns=site_association_by_host.columns)
    host_frames = []
    for host, host_frame in site_association_by_host.groupby("host_differential", dropna=False):
        trimmed = host_frame.sort_values(
            ["host_fdr_bh", "fisher_pvalue_two_sided", "site_number"],
            ascending=[True, True, True],
            na_position="last",
        ).head(top_n).copy()
        trimmed["rank_within_host"] = np.arange(1, len(trimmed) + 1)
        host_frames.append(trimmed)
    return pd.concat(host_frames, ignore_index=True).sort_values(
        ["host_differential", "rank_within_host", "site_number"]
    ).reset_index(drop=True)


def build_prediction_outputs(
    genetic_distance_pairs: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if genetic_distance_pairs.empty or virulence_profiles.empty:
        empty_summary = pd.DataFrame(
            columns=["k_neighbors", "evaluated_isolates", "pathotype_accuracy", "mean_host_accuracy", "exact_profile_accuracy"]
        )
        empty_detail = pd.DataFrame(
            columns=["k_neighbors", "isolate_id", "true_pathotype", "predicted_pathotype", "pathotype_correct", "host_accuracy", "exact_profile_match"]
        )
        return empty_summary, empty_detail

    host_columns = _virulence_columns(virulence_profiles)
    metadata = virulence_profiles[["isolate_id"]].copy()
    if "pathotype" in virulence_profiles.columns:
        metadata["pathotype"] = virulence_profiles["pathotype"]
    else:
        metadata["pathotype"] = pd.NA
    metadata["profile_signature"] = virulence_profiles[host_columns].apply(lambda row: "|".join(row.astype(str)), axis=1)
    profile_map = metadata.set_index("isolate_id").to_dict(orient="index")
    host_map = virulence_profiles.set_index("isolate_id")[host_columns].astype(str).to_dict(orient="index")

    forward = genetic_distance_pairs.rename(columns={"isolate_id_left": "isolate_id", "isolate_id_right": "neighbor_isolate"})
    reverse = genetic_distance_pairs.rename(columns={"isolate_id_right": "isolate_id", "isolate_id_left": "neighbor_isolate"})
    bidirectional = pd.concat([forward, reverse], ignore_index=True)
    bidirectional = bidirectional[bidirectional["isolate_id"].isin(metadata["isolate_id"]) & bidirectional["neighbor_isolate"].isin(metadata["isolate_id"])]
    bidirectional = bidirectional.sort_values(["isolate_id", "genetic_distance", "neighbor_isolate"], ascending=[True, True, True])

    detail_rows: list[dict[str, object]] = []
    for k_value in [1, 3, 5]:
        isolate_rows = []
        for isolate_id, frame in bidirectional.groupby("isolate_id", sort=True):
            neighbors = frame.head(min(k_value, len(frame))).copy()
            if neighbors.empty:
                continue
            neighbor_ids = neighbors["neighbor_isolate"].tolist()

            pathotype_votes = [str(profile_map[neighbor_id]["pathotype"]) for neighbor_id in neighbor_ids if str(profile_map[neighbor_id]["pathotype"]) != ""]
            predicted_pathotype = ""
            if pathotype_votes:
                counted = pd.Series(pathotype_votes).value_counts()
                predicted_pathotype = counted.index.sort_values().tolist()[0] if counted.iloc[0] == counted.max() and (counted == counted.iloc[0]).sum() > 1 else counted.index[0]
            true_pathotype = str(profile_map[isolate_id]["pathotype"]) if str(profile_map[isolate_id]["pathotype"]) != "" else ""

            host_predictions = {}
            correct_count = 0
            evaluated_hosts = 0
            exact_profile_match = True
            for host_column in host_columns:
                votes = [host_map[neighbor_id][host_column] for neighbor_id in neighbor_ids if host_map[neighbor_id][host_column] in {"R", "S"}]
                if not votes:
                    host_predictions[host_column] = ""
                    exact_profile_match = False
                    continue
                vote_counts = pd.Series(votes).value_counts()
                top_count = vote_counts.iloc[0]
                top_labels = sorted(vote_counts[vote_counts == top_count].index.tolist())
                predicted = top_labels[0]
                truth = host_map[isolate_id][host_column]
                evaluated_hosts += 1
                if predicted == truth:
                    correct_count += 1
                else:
                    exact_profile_match = False
                host_predictions[host_column] = predicted

            host_accuracy = (correct_count / evaluated_hosts) if evaluated_hosts > 0 else np.nan
            pathotype_correct = predicted_pathotype == true_pathotype if predicted_pathotype and true_pathotype else False
            isolate_rows.append(
                {
                    "k_neighbors": k_value,
                    "isolate_id": isolate_id,
                    "true_pathotype": true_pathotype,
                    "predicted_pathotype": predicted_pathotype,
                    "pathotype_correct": pathotype_correct,
                    "host_accuracy": host_accuracy,
                    "exact_profile_match": bool(exact_profile_match),
                }
            )
        detail_frame = pd.DataFrame(isolate_rows)
        detail_rows.append(detail_frame)

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if detail.empty:
        return (
            pd.DataFrame(columns=["k_neighbors", "evaluated_isolates", "pathotype_accuracy", "mean_host_accuracy", "exact_profile_accuracy"]),
            detail,
        )
    summary = (
        detail.groupby("k_neighbors", dropna=False)
        .agg(
            evaluated_isolates=("isolate_id", "size"),
            pathotype_accuracy=("pathotype_correct", "mean"),
            mean_host_accuracy=("host_accuracy", "mean"),
            exact_profile_accuracy=("exact_profile_match", "mean"),
        )
        .reset_index()
        .sort_values("k_neighbors")
        .reset_index(drop=True)
    )
    return summary, detail
