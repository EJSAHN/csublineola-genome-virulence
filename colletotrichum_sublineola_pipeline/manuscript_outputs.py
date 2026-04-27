
from __future__ import annotations

import numpy as np
import pandas as pd


def _lookup_value(dataframe: pd.DataFrame, key_column: str, key_value: str, value_column: str) -> float:
    subset = dataframe[dataframe[key_column] == key_value]
    if subset.empty:
        return np.nan
    return subset.iloc[0][value_column]


def build_dataset_table(
    alignment_summary: pd.DataFrame,
    sample_qc: pd.DataFrame,
    site_qc: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
    pathotype_summary: pd.DataFrame,
) -> pd.DataFrame:
    host_columns = [
        column for column in virulence_profiles.columns
        if column not in {"isolate_id", "pathotype"}
        and set(virulence_profiles[column].dropna().astype(str).str.upper().str.strip()).issubset({"R", "S"})
    ]
    pathotype_counts = pathotype_summary.set_index("pathotype")["isolate_count"] if not pathotype_summary.empty else pd.Series(dtype=float)
    rows = [
        {"metric": "alignment_isolates", "value": _lookup_value(alignment_summary, "metric", "isolate_count", "value")},
        {"metric": "alignment_length", "value": _lookup_value(alignment_summary, "metric", "alignment_length", "value")},
        {"metric": "variable_sites", "value": _lookup_value(alignment_summary, "metric", "variable_canonical_sites", "value")},
        {"metric": "parsimony_informative_sites", "value": _lookup_value(alignment_summary, "metric", "parsimony_informative_sites", "value")},
        {"metric": "virulence_panel_isolates", "value": int(len(virulence_profiles))},
        {"metric": "host_differentials", "value": int(len(host_columns))},
        {"metric": "unique_pathotypes", "value": int(virulence_profiles["pathotype"].nunique()) if "pathotype" in virulence_profiles.columns else np.nan},
        {"metric": "singleton_pathotypes", "value": int((pathotype_counts == 1).sum()) if not pathotype_counts.empty else 0},
        {"metric": "repeated_pathotypes", "value": int((pathotype_counts > 1).sum()) if not pathotype_counts.empty else 0},
        {"metric": "mean_missing_fraction", "value": float(sample_qc["missing_fraction"].mean()) if "missing_fraction" in sample_qc.columns else np.nan},
        {"metric": "median_missing_fraction", "value": float(sample_qc["missing_fraction"].median()) if "missing_fraction" in sample_qc.columns else np.nan},
        {"metric": "mean_site_missing_rate", "value": float(site_qc["missing_rate"].mean()) if "missing_rate" in site_qc.columns else np.nan},
        {"metric": "median_site_missing_rate", "value": float(site_qc["missing_rate"].median()) if "missing_rate" in site_qc.columns else np.nan},
    ]
    return pd.DataFrame(rows)


def build_host_results(
    host_signal_overview: pd.DataFrame,
    top_host_site_table: pd.DataFrame,
) -> pd.DataFrame:
    if host_signal_overview.empty:
        return host_signal_overview
    dataframe = host_signal_overview.copy()
    if not top_host_site_table.empty:
        keep_columns = [
            "host_differential",
            "site_number",
            "fisher_pvalue_two_sided",
            "host_fdr_bh",
            "global_fdr_bh",
            "minor_frequency_difference_susceptible_minus_resistant",
            "odds_ratio_haldane",
        ]
        available = [column for column in keep_columns if column in top_host_site_table.columns]
        renamed = top_host_site_table[available].rename(
            columns={
                "site_number": "representative_site_number",
                "fisher_pvalue_two_sided": "representative_site_raw_pvalue",
                "host_fdr_bh": "representative_site_host_fdr",
                "global_fdr_bh": "representative_site_global_fdr",
                "minor_frequency_difference_susceptible_minus_resistant": "representative_site_minor_frequency_difference",
                "odds_ratio_haldane": "representative_site_odds_ratio_haldane",
            }
        )
        dataframe = dataframe.merge(renamed, on="host_differential", how="left")
    def _grade(row: pd.Series) -> str:
        if pd.notna(row.get("host_fdr_bh")) and row.get("host_fdr_bh") < 0.10:
            return "host_level_signal"
        if pd.notna(row.get("permutation_pvalue_two_sided")) and row.get("permutation_pvalue_two_sided") < 0.10:
            return "suggestive_host_signal"
        if pd.notna(row.get("representative_site_raw_pvalue")) and row.get("representative_site_raw_pvalue") < 0.05:
            return "site_level_only"
        return "no_clear_signal"
    dataframe["evidence_grade"] = dataframe.apply(_grade, axis=1)
    sort_columns = [
        "host_fdr_bh",
        "permutation_pvalue_two_sided",
        "representative_site_raw_pvalue",
        "host_differential",
    ]
    return dataframe.sort_values(sort_columns, ascending=[True, True, True, True], na_position="last").reset_index(drop=True)


def build_pathotype_results(
    pathotype_summary: pd.DataFrame,
    pathotype_distance_summary: pd.DataFrame,
    profile_distance_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not pathotype_summary.empty:
        for _, row in pathotype_summary.sort_values(["isolate_count", "pathotype"], ascending=[False, True]).iterrows():
            rows.append(
                {
                    "category": "pathotype_count",
                    "label": row.get("pathotype", ""),
                    "value": row.get("isolate_count", np.nan),
                    "secondary_value": row.get("susceptible_hosts_mean", np.nan),
                    "notes": "isolate_count_and_mean_susceptible_hosts",
                }
            )
    if not pathotype_distance_summary.empty:
        for _, row in pathotype_distance_summary.iterrows():
            rows.append(
                {
                    "category": "pathotype_distance_summary",
                    "label": row.get("comparison_flag", row.get("label", "pathotype")),
                    "value": row.get("mean_genetic_distance", np.nan),
                    "secondary_value": row.get("pair_count", np.nan),
                    "notes": "mean_genetic_distance_and_pair_count",
                }
            )
    if not profile_distance_summary.empty:
        for _, row in profile_distance_summary.iterrows():
            rows.append(
                {
                    "category": "profile_distance_summary",
                    "label": row.get("comparison_flag", row.get("label", "profile")),
                    "value": row.get("mean_genetic_distance", np.nan),
                    "secondary_value": row.get("pair_count", np.nan),
                    "notes": "mean_genetic_distance_and_pair_count",
                }
            )
    if not prediction_summary.empty:
        for _, row in prediction_summary.sort_values("k_neighbors").iterrows():
            rows.append(
                {
                    "category": "prediction_summary",
                    "label": f"k_{int(row.get('k_neighbors', np.nan))}",
                    "value": row.get("mean_host_accuracy", np.nan),
                    "secondary_value": row.get("exact_profile_accuracy", np.nan),
                    "notes": "mean_host_accuracy_and_exact_profile_accuracy",
                }
            )
    return pd.DataFrame(rows)


def build_interpretation_audit(
    primary_summary: pd.DataFrame,
    host_results: pd.DataFrame,
    top_host_site_table: pd.DataFrame,
    prediction_summary: pd.DataFrame,
) -> pd.DataFrame:
    lookup = primary_summary.set_index("field")["value"].to_dict() if not primary_summary.empty else {}
    k5 = prediction_summary[prediction_summary["k_neighbors"] == 5]
    k5_host_accuracy = k5.iloc[0]["mean_host_accuracy"] if not k5.empty else np.nan
    rows = []
    mantel_p = lookup.get("mantel_pearson_pvalue", np.nan)
    rows.append(
        {
            "claim_id": "C1",
            "claim": "Genome-wide distance recovers virulence distance.",
            "status": "supported" if pd.notna(mantel_p) and mantel_p < 0.05 else "not_supported",
            "evidence": f"mantel_pearson_pvalue={mantel_p}",
            "recommended_wording": "Avoid strong coupling language when Mantel support is absent." if not (pd.notna(mantel_p) and mantel_p < 0.05) else "Coupling language is defensible.",
        }
    )
    repeated = lookup.get("repeated_pathotypes", np.nan)
    rows.append(
        {
            "claim_id": "C2",
            "claim": "Pathotype labels provide repeated phenotype classes for within-class comparisons.",
            "status": "supported" if pd.notna(repeated) and repeated > 0 else "not_supported",
            "evidence": f"repeated_pathotypes={repeated}",
            "recommended_wording": "Singleton-heavy pathotype labels should be discussed as fine-grained labels rather than robust classes." if not (pd.notna(repeated) and repeated > 0) else "Repeated pathotype comparisons are available.",
        }
    )
    any_host_signal = False
    if not host_results.empty:
        any_host_signal = bool((host_results["evidence_grade"].isin(["host_level_signal", "suggestive_host_signal"])).any())
    rows.append(
        {
            "claim_id": "C3",
            "claim": "Some host differentials retain non-random structure after genome-wide decoupling is acknowledged.",
            "status": "supported" if any_host_signal else "weak_support",
            "evidence": f"hosts_with_signal={int(any_host_signal)}",
            "recommended_wording": "Host-specific structure can be described as non-random but not genome-wide predictive." if any_host_signal else "Host-specific claims should remain cautious.",
        }
    )
    best_global_fdr = np.nan
    if not top_host_site_table.empty and "global_fdr_bh" in top_host_site_table.columns:
        best_global_fdr = top_host_site_table["global_fdr_bh"].min()
    rows.append(
        {
            "claim_id": "C4",
            "claim": "Specific individual sites meet study-wide significance after multiple testing control.",
            "status": "supported" if pd.notna(best_global_fdr) and best_global_fdr < 0.10 else "not_supported",
            "evidence": f"best_global_fdr={best_global_fdr}",
            "recommended_wording": "Report site-level signals as candidates unless study-wide control is met." if not (pd.notna(best_global_fdr) and best_global_fdr < 0.10) else "Study-wide candidate site language is defensible.",
        }
    )
    rows.append(
        {
            "claim_id": "C5",
            "claim": "This dataset supports an architecture-focused manuscript rather than a high-confidence marker-discovery manuscript.",
            "status": "supported" if pd.notna(mantel_p) and mantel_p >= 0.05 else "consider_alternative_framing",
            "evidence": f"mantel_pearson_pvalue={mantel_p}; k5_mean_host_accuracy={k5_host_accuracy}",
            "recommended_wording": "Frame the paper around decoupling between broad genomic relatedness and fine-grained virulence labels.",
        }
    )
    return pd.DataFrame(rows)


def build_narrative_support(
    primary_summary: pd.DataFrame,
    host_results: pd.DataFrame,
    interpretation_audit: pd.DataFrame,
) -> pd.DataFrame:
    lookup = primary_summary.set_index("field")["value"].to_dict() if not primary_summary.empty else {}
    top_host = ""
    if not host_results.empty:
        top_host = str(host_results.iloc[0]["host_differential"])
    rows = [
        {
            "section": "results",
            "order": 1,
            "sentence": f"The aligned dataset contained {lookup.get('alignment_isolates', np.nan)} isolates and {lookup.get('alignment_length', np.nan)} aligned positions, with {lookup.get('virulence_isolates', np.nan)} isolates represented in the virulence panel.",
        },
        {
            "section": "results",
            "order": 2,
            "sentence": f"Genome-wide genetic distance showed little correspondence with virulence distance (Mantel Pearson = {lookup.get('mantel_pearson', np.nan)}, permutation p = {lookup.get('mantel_pearson_pvalue', np.nan)}).",
        },
        {
            "section": "results",
            "order": 3,
            "sentence": f"Pathotype labels were highly granular, with {lookup.get('unique_pathotypes', np.nan)} pathotypes across {lookup.get('virulence_isolates', np.nan)} isolates and {lookup.get('singleton_pathotypes', np.nan)} singleton pathotypes.",
        },
        {
            "section": "results",
            "order": 4,
            "sentence": f"The strongest host-level signal was observed for {top_host} when host differentials were ranked by permutation evidence and representative site support.",
        },
        {
            "section": "discussion",
            "order": 5,
            "sentence": "These results support a manuscript framed around decoupling between broad genomic relatedness and fine-grained virulence classification, while treating site-level signals as candidates rather than confirmed markers.",
        },
    ]
    for _, row in interpretation_audit.iterrows():
        rows.append(
            {
                "section": "claim_audit",
                "order": len(rows) + 1,
                "sentence": f"{row['claim_id']}: {row['claim']} Status={row['status']}. {row['recommended_wording']}",
            }
        )
    return pd.DataFrame(rows)


def build_sheet_guide() -> pd.DataFrame:
    rows = [
        {"sheet_name": "DatasetTable", "purpose": "Compact dataset-level values used to summarize the analysis."},
        {"sheet_name": "HostResults", "purpose": "Host-level ranking table that prioritises differentials with the strongest non-random signal."},
        {"sheet_name": "PathotypeResults", "purpose": "Compact pathotype, profile, and prediction metrics."},
        {"sheet_name": "InterpretationAudit", "purpose": "Claim-by-claim evidence audit to prevent overstatement."},
        {"sheet_name": "NarrativeSupport", "purpose": "Narrative support statements grounded in workbook statistics."},
    ]
    return pd.DataFrame(rows)
