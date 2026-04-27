
from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics_tools import label_separation_permutation_test, pearson_correlation, spearman_correlation


def _virulence_columns(virulence_profiles: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in virulence_profiles.columns:
        if column in {"isolate_id", "pathotype"}:
            continue
        series = virulence_profiles[column].dropna().astype(str).str.upper().str.strip()
        if not series.empty and set(series.unique()).issubset({"R", "S"}):
            columns.append(column)
    return columns


def build_profile_complexity_summary(virulence_profiles: pd.DataFrame) -> pd.DataFrame:
    if virulence_profiles.empty:
        return pd.DataFrame(columns=["metric", "value"])
    host_columns = _virulence_columns(virulence_profiles)
    signatures = virulence_profiles[host_columns].astype(str).agg("|".join, axis=1)
    susceptible_counts = virulence_profiles[host_columns].eq("S").sum(axis=1)
    pathotype_counts = virulence_profiles.groupby("pathotype", dropna=False)["isolate_id"].size()
    signature_counts = signatures.value_counts(dropna=False)
    rows = [
        {"metric": "virulence_isolates", "value": int(len(virulence_profiles))},
        {"metric": "host_differentials", "value": int(len(host_columns))},
        {"metric": "unique_pathotypes", "value": int(virulence_profiles["pathotype"].nunique()) if "pathotype" in virulence_profiles.columns else np.nan},
        {"metric": "singleton_pathotypes", "value": int((pathotype_counts == 1).sum()) if not pathotype_counts.empty else 0},
        {"metric": "repeated_pathotypes", "value": int((pathotype_counts > 1).sum()) if not pathotype_counts.empty else 0},
        {"metric": "pathotype_singleton_fraction", "value": float((pathotype_counts == 1).mean()) if not pathotype_counts.empty else np.nan},
        {"metric": "unique_exact_profiles", "value": int(signature_counts.size)},
        {"metric": "singleton_exact_profiles", "value": int((signature_counts == 1).sum()) if not signature_counts.empty else 0},
        {"metric": "repeated_exact_profiles", "value": int((signature_counts > 1).sum()) if not signature_counts.empty else 0},
        {"metric": "exact_profile_singleton_fraction", "value": float((signature_counts == 1).mean()) if not signature_counts.empty else np.nan},
        {"metric": "mean_susceptible_hosts_per_isolate", "value": float(susceptible_counts.mean())},
        {"metric": "median_susceptible_hosts_per_isolate", "value": float(susceptible_counts.median())},
        {"metric": "min_susceptible_hosts_per_isolate", "value": int(susceptible_counts.min())},
        {"metric": "max_susceptible_hosts_per_isolate", "value": int(susceptible_counts.max())},
        {"metric": "virulence_burden_std", "value": float(susceptible_counts.std(ddof=1)) if len(susceptible_counts) > 1 else np.nan},
    ]
    return pd.DataFrame(rows)


def build_host_signal_overview(
    virulence_profiles: pd.DataFrame,
    host_separation_tests: pd.DataFrame,
    site_association_by_host_exact: pd.DataFrame,
) -> pd.DataFrame:
    if virulence_profiles.empty:
        return pd.DataFrame()
    host_columns = _virulence_columns(virulence_profiles)
    rows: list[dict[str, object]] = []
    host_separation_lookup = {}
    if not host_separation_tests.empty and "host_differential" in host_separation_tests.columns:
        host_separation_lookup = host_separation_tests.set_index("host_differential").to_dict(orient="index")
    top_site_lookup: dict[str, dict[str, object]] = {}
    if not site_association_by_host_exact.empty and "host_differential" in site_association_by_host_exact.columns:
        first_rows = (
            site_association_by_host_exact.sort_values(
                ["host_differential", "host_fdr_bh", "fisher_pvalue_two_sided", "site_number"],
                ascending=[True, True, True, True],
                na_position="last",
            )
            .groupby("host_differential", dropna=False)
            .head(1)
        )
        top_site_lookup = first_rows.set_index("host_differential").to_dict(orient="index")
    for host_column in host_columns:
        responses = virulence_profiles[host_column].astype(str).str.upper().str.strip()
        host_stats = host_separation_lookup.get(host_column, {})
        top_site = top_site_lookup.get(host_column, {})
        rows.append(
            {
                "host_differential": host_column,
                "evaluated_isolates": int(responses.isin(["R", "S"]).sum()),
                "susceptible_isolates": int((responses == "S").sum()),
                "resistant_isolates": int((responses == "R").sum()),
                "susceptible_fraction": float((responses == "S").mean()),
                "between_minus_within": host_stats.get("between_minus_within", np.nan),
                "permutation_pvalue_two_sided": host_stats.get("permutation_pvalue_two_sided", np.nan),
                "host_fdr_bh": host_stats.get("host_fdr_bh", np.nan),
                "top_site_number": top_site.get("site_number", np.nan),
                "top_site_raw_pvalue": top_site.get("fisher_pvalue_two_sided", np.nan),
                "top_site_host_fdr": top_site.get("host_fdr_bh", np.nan),
                "top_site_global_fdr": top_site.get("global_fdr_bh", np.nan),
                "top_site_minor_frequency_difference": top_site.get("minor_frequency_difference_susceptible_minus_resistant", np.nan),
                "top_site_odds_ratio_haldane": top_site.get("odds_ratio_haldane", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["host_fdr_bh", "permutation_pvalue_two_sided", "top_site_raw_pvalue", "host_differential"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_top_host_site_table(site_association_by_host_exact: pd.DataFrame) -> pd.DataFrame:
    if site_association_by_host_exact.empty:
        return pd.DataFrame(columns=site_association_by_host_exact.columns)
    dataframe = (
        site_association_by_host_exact.sort_values(
            ["host_differential", "host_fdr_bh", "fisher_pvalue_two_sided", "site_number"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        .groupby("host_differential", dropna=False)
        .head(1)
        .copy()
    )
    return dataframe.reset_index(drop=True)


def _pair_vectors(genetic_matrix: pd.DataFrame, virulence_distance_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    aligned = sorted(set(genetic_matrix.index).intersection(virulence_distance_matrix.index))
    left = genetic_matrix.loc[aligned, aligned].to_numpy(dtype=float)
    right = virulence_distance_matrix.loc[aligned, aligned].to_numpy(dtype=float)
    lower = np.tril_indices(len(aligned), k=-1)
    return left[lower], right[lower]


def build_leave_one_out_influence(
    genetic_matrix: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
    virulence_distance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(genetic_matrix.index).intersection(virulence_profiles["isolate_id"]))
    if len(overlap) < 4:
        return pd.DataFrame()
    profiles = virulence_profiles[virulence_profiles["isolate_id"].isin(overlap)].copy().set_index("isolate_id")
    genetic_sub = genetic_matrix.loc[overlap, overlap].astype(float)
    virulence_sub = virulence_distance_matrix.loc[overlap, overlap].astype(float)
    full_genetic_vector, full_virulence_vector = _pair_vectors(genetic_sub, virulence_sub)
    full_pearson = pearson_correlation(full_genetic_vector, full_virulence_vector)
    full_spearman = spearman_correlation(full_genetic_vector, full_virulence_vector)
    full_pathotype = label_separation_permutation_test(
        genetic_sub.to_numpy(dtype=float),
        profiles.loc[overlap, "pathotype"].tolist(),
        permutations=1,
        random_seed=1,
        label_name="pathotype",
    ).iloc[0]["between_minus_within"]
    signatures = profiles[_virulence_columns(profiles.reset_index())].astype(str).agg("|".join, axis=1)
    full_profile = label_separation_permutation_test(
        genetic_sub.to_numpy(dtype=float),
        signatures.loc[overlap].tolist(),
        permutations=1,
        random_seed=1,
        label_name="virulence_profile",
    ).iloc[0]["between_minus_within"]

    rows: list[dict[str, object]] = []
    for isolate_id in overlap:
        remaining = [value for value in overlap if value != isolate_id]
        g_matrix = genetic_sub.loc[remaining, remaining]
        v_matrix = virulence_sub.loc[remaining, remaining]
        g_vector, v_vector = _pair_vectors(g_matrix, v_matrix)
        pearson_value = pearson_correlation(g_vector, v_vector)
        spearman_value = spearman_correlation(g_vector, v_vector)
        pathotype_value = label_separation_permutation_test(
            g_matrix.to_numpy(dtype=float),
            profiles.loc[remaining, "pathotype"].tolist(),
            permutations=1,
            random_seed=1,
            label_name="pathotype",
        ).iloc[0]["between_minus_within"]
        profile_value = label_separation_permutation_test(
            g_matrix.to_numpy(dtype=float),
            signatures.loc[remaining].tolist(),
            permutations=1,
            random_seed=1,
            label_name="virulence_profile",
        ).iloc[0]["between_minus_within"]
        rows.append(
            {
                "omitted_isolate": isolate_id,
                "remaining_isolates": len(remaining),
                "pearson_correlation": pearson_value,
                "spearman_correlation": spearman_value,
                "pathotype_between_minus_within": pathotype_value,
                "profile_between_minus_within": profile_value,
                "mean_genetic_distance": float(np.nanmean(g_vector)) if g_vector.size else np.nan,
                "mean_virulence_distance": float(np.nanmean(v_vector)) if v_vector.size else np.nan,
                "delta_pearson_from_full": pearson_value - full_pearson if pd.notna(pearson_value) and pd.notna(full_pearson) else np.nan,
                "delta_spearman_from_full": spearman_value - full_spearman if pd.notna(spearman_value) and pd.notna(full_spearman) else np.nan,
                "delta_pathotype_from_full": pathotype_value - full_pathotype if pd.notna(pathotype_value) and pd.notna(full_pathotype) else np.nan,
                "delta_profile_from_full": profile_value - full_profile if pd.notna(profile_value) and pd.notna(full_profile) else np.nan,
                "omitted_pathotype": profiles.loc[isolate_id, "pathotype"],
            }
        )
    dataframe = pd.DataFrame(rows)
    dataframe["influence_score_abs"] = (
        dataframe[["delta_pearson_from_full", "delta_spearman_from_full", "delta_pathotype_from_full", "delta_profile_from_full"]]
        .abs()
        .sum(axis=1)
    )
    return dataframe.sort_values(
        ["influence_score_abs", "omitted_isolate"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_primary_summary(
    alignment_summary: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
    mantel_summary: pd.DataFrame,
    pathotype_permutation: pd.DataFrame,
    profile_permutation: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    host_separation_tests: pd.DataFrame,
    site_association_by_host_exact: pd.DataFrame,
    leave_one_out_influence: pd.DataFrame,
) -> pd.DataFrame:
    def _mantel_value(method: str, column: str) -> float:
        subset = mantel_summary[mantel_summary["method"] == method]
        if subset.empty:
            return np.nan
        return subset.iloc[0][column]

    best_host = pd.DataFrame()
    if not host_separation_tests.empty:
        best_host = host_separation_tests.sort_values(
            ["host_fdr_bh", "permutation_pvalue_two_sided", "host_differential"],
            ascending=[True, True, True],
            na_position="last",
        ).head(1)
    best_site = pd.DataFrame()
    if not site_association_by_host_exact.empty:
        best_site = site_association_by_host_exact.sort_values(
            ["global_fdr_bh", "host_fdr_bh", "fisher_pvalue_two_sided", "host_differential", "site_number"],
            ascending=[True, True, True, True, True],
            na_position="last",
        ).head(1)
    k5 = prediction_summary[prediction_summary["k_neighbors"] == 5]
    k3 = prediction_summary[prediction_summary["k_neighbors"] == 3]
    leave_one_out_top = leave_one_out_influence.head(1)

    pathotype_counts = virulence_profiles.groupby("pathotype", dropna=False)["isolate_id"].size() if not virulence_profiles.empty else pd.Series(dtype=int)
    host_columns = _virulence_columns(virulence_profiles)
    signatures = virulence_profiles[host_columns].astype(str).agg("|".join, axis=1) if host_columns else pd.Series(dtype=str)
    rows = [
        {"field": "alignment_isolates", "value": float(alignment_summary.set_index("metric").loc["isolate_count", "value"]) if "isolate_count" in alignment_summary["metric"].values else np.nan},
        {"field": "alignment_length", "value": float(alignment_summary.set_index("metric").loc["alignment_length", "value"]) if "alignment_length" in alignment_summary["metric"].values else np.nan},
        {"field": "virulence_isolates", "value": int(len(virulence_profiles))},
        {"field": "host_differentials", "value": int(len(host_columns))},
        {"field": "unique_pathotypes", "value": int(virulence_profiles["pathotype"].nunique()) if "pathotype" in virulence_profiles.columns else np.nan},
        {"field": "singleton_pathotypes", "value": int((pathotype_counts == 1).sum()) if not pathotype_counts.empty else 0},
        {"field": "repeated_pathotypes", "value": int((pathotype_counts > 1).sum()) if not pathotype_counts.empty else 0},
        {"field": "unique_exact_profiles", "value": int(signatures.nunique()) if not signatures.empty else 0},
        {"field": "mantel_pearson", "value": _mantel_value("pearson", "observed_correlation")},
        {"field": "mantel_pearson_pvalue", "value": _mantel_value("pearson", "permutation_pvalue_two_sided")},
        {"field": "mantel_spearman", "value": _mantel_value("spearman", "observed_correlation")},
        {"field": "mantel_spearman_pvalue", "value": _mantel_value("spearman", "permutation_pvalue_two_sided")},
        {"field": "pathotype_between_minus_within", "value": pathotype_permutation.iloc[0]["between_minus_within"] if not pathotype_permutation.empty else np.nan},
        {"field": "pathotype_permutation_pvalue", "value": pathotype_permutation.iloc[0]["permutation_pvalue_two_sided"] if not pathotype_permutation.empty else np.nan},
        {"field": "profile_between_minus_within", "value": profile_permutation.iloc[0]["between_minus_within"] if not profile_permutation.empty else np.nan},
        {"field": "profile_permutation_pvalue", "value": profile_permutation.iloc[0]["permutation_pvalue_two_sided"] if not profile_permutation.empty else np.nan},
        {"field": "k3_mean_host_accuracy", "value": k3.iloc[0]["mean_host_accuracy"] if not k3.empty else np.nan},
        {"field": "k5_mean_host_accuracy", "value": k5.iloc[0]["mean_host_accuracy"] if not k5.empty else np.nan},
        {"field": "k5_exact_profile_accuracy", "value": k5.iloc[0]["exact_profile_accuracy"] if not k5.empty else np.nan},
        {"field": "hosts_with_host_fdr_lt_0_10", "value": int((host_separation_tests["host_fdr_bh"] < 0.10).sum()) if not host_separation_tests.empty else 0},
        {"field": "hosts_with_raw_p_lt_0_10", "value": int((host_separation_tests["permutation_pvalue_two_sided"] < 0.10).sum()) if not host_separation_tests.empty else 0},
        {"field": "top_host_by_pvalue", "value": best_host.iloc[0]["host_differential"] if not best_host.empty else ""},
        {"field": "top_host_raw_pvalue", "value": best_host.iloc[0]["permutation_pvalue_two_sided"] if not best_host.empty else np.nan},
        {"field": "top_host_fdr", "value": best_host.iloc[0]["host_fdr_bh"] if not best_host.empty else np.nan},
        {"field": "top_site_host", "value": best_site.iloc[0]["host_differential"] if not best_site.empty else ""},
        {"field": "top_site_number", "value": best_site.iloc[0]["site_number"] if not best_site.empty else np.nan},
        {"field": "top_site_raw_pvalue", "value": best_site.iloc[0]["fisher_pvalue_two_sided"] if not best_site.empty else np.nan},
        {"field": "top_site_global_fdr", "value": best_site.iloc[0]["global_fdr_bh"] if not best_site.empty else np.nan},
        {"field": "most_influential_leave_one_out_isolate", "value": leave_one_out_top.iloc[0]["omitted_isolate"] if not leave_one_out_top.empty else ""},
        {"field": "most_influential_leave_one_out_score", "value": leave_one_out_top.iloc[0]["influence_score_abs"] if not leave_one_out_top.empty else np.nan},
    ]

    interpretation = "broad_genome_wide_distance_does_not_recover_fine_grained_pathotype_labels"
    if rows[8]["value"] is not np.nan and pd.notna(rows[8]["value"]) and pd.notna(rows[9]["value"]):
        if abs(rows[8]["value"]) >= 0.2 and rows[9]["value"] < 0.05:
            interpretation = "genome_wide_distance_tracks_virulence_distance"
    rows.append({"field": "primary_interpretation", "value": interpretation})
    return pd.DataFrame(rows)
