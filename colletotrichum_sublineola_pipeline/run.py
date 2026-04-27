
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .alignment_processing import (
    build_alignment_summary,
    build_isolate_summary,
    build_pairwise_distance_outputs,
    build_site_summary,
    load_alignment,
)
from .common import build_run_metadata, compute_sha256, read_table
from .excel_reporting import write_excel_workbook
from .extended_analysis import (
    build_analysis_cohorts,
    build_binary_distance_summary,
    build_host_separation_summary,
    build_host_separation_tests,
    build_nearest_neighbor_summary,
    build_pair_class_summary,
    build_pathotype_pair_summary,
    build_pathotype_support_summary,
    build_prediction_outputs,
    build_sample_qc,
    build_site_association_by_host,
    build_site_association_by_host_exact,
    build_site_qc,
    build_top_site_associations,
    build_top_site_associations_exact,
)
from .summary_outputs import (
    build_host_signal_overview,
    build_leave_one_out_influence,
    build_primary_summary,
    build_profile_complexity_summary,
    build_top_host_site_table,
)
from .manuscript_outputs import (
    build_dataset_table,
    build_host_results,
    build_interpretation_audit,
    build_narrative_support,
    build_pathotype_results,
    build_sheet_guide,
)
from .statistics_tools import label_separation_permutation_test, mantel_permutation_test
from .virulence_processing import (
    build_distance_relationship_outputs,
    build_virulence_outputs,
    load_virulence_table,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse aligned Colletotrichum sublineola sequence data and write an Excel workbook."
    )
    parser.add_argument("--alignment", required=True, type=Path, help="Path to an aligned FASTA file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the Excel workbook that will be created.")
    parser.add_argument(
        "--virulence-table",
        type=Path,
        default=None,
        help="Optional CSV or Excel table with virulence calls encoded as R and S.",
    )
    parser.add_argument(
        "--metadata-table",
        type=Path,
        default=None,
        help="Optional CSV or Excel table with additional isolate metadata keyed by isolate_id.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Number of random permutations used in inferential analyses.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1729,
        help="Base random seed used for permutation-based procedures.",
    )
    parser.add_argument(
        "--top-site-count",
        type=int,
        default=25,
        help="Number of top site associations retained per host differential.",
    )
    return parser


def _profile_signature_frame(virulence_profiles: pd.DataFrame) -> pd.DataFrame:
    virulence_columns = []
    for column in virulence_profiles.columns:
        if column in {"isolate_id", "pathotype"}:
            continue
        values = set(virulence_profiles[column].dropna().astype(str).str.upper().str.strip())
        if values and values.issubset({"R", "S"}):
            virulence_columns.append(column)
    dataframe = virulence_profiles[["isolate_id"]].copy()
    dataframe["profile_signature"] = virulence_profiles[virulence_columns].apply(lambda row: "|".join(row.astype(str)), axis=1)
    return dataframe


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    if not arguments.alignment.exists():
        raise FileNotFoundError(f"Alignment file not found: {arguments.alignment}")
    if arguments.permutations <= 0:
        raise ValueError("Permutations must be a positive integer.")
    if arguments.top_site_count <= 0:
        raise ValueError("Top site count must be a positive integer.")

    alignment_data = load_alignment(arguments.alignment)
    alignment_summary = build_alignment_summary(alignment_data)
    isolate_summary = build_isolate_summary(alignment_data)
    site_summary = build_site_summary(alignment_data)
    site_qc = build_site_qc(site_summary)
    genetic_distance_matrix, genetic_distance_pairs = build_pairwise_distance_outputs(alignment_data)

    data_checks: list[dict[str, object]] = []
    sheet_mapping: list[tuple[str, pd.DataFrame]] = []

    metadata_hashes = {"alignment_file": compute_sha256(arguments.alignment)}
    extra_metadata = [
        ("alignment_file_name", arguments.alignment.name),
        ("alignment_isolates", len(alignment_data.isolate_ids)),
        ("alignment_length", alignment_data.alignment_length),
        ("analysis_release", __version__),
        ("analysis_scope", "genome_virulence_reanalysis"),
        ("permutations", arguments.permutations),
        ("random_seed", arguments.random_seed),
        ("top_site_count", arguments.top_site_count),
    ]

    if arguments.metadata_table is not None:
        if not arguments.metadata_table.exists():
            raise FileNotFoundError(f"Metadata table not found: {arguments.metadata_table}")
        metadata_hashes["metadata_table"] = compute_sha256(arguments.metadata_table)
        extra_metadata.append(("metadata_file_name", arguments.metadata_table.name))
        metadata_table = read_table(arguments.metadata_table).copy()
        if "isolate_id" not in metadata_table.columns:
            raise ValueError("The metadata table must contain an 'isolate_id' column.")
        metadata_table["isolate_id"] = metadata_table["isolate_id"].astype(str).str.strip()
        isolate_summary = isolate_summary.merge(metadata_table, on="isolate_id", how="left")
        matched_metadata = isolate_summary.drop(columns=["isolate_id"]).notna().any(axis=1).sum()
        extra_metadata.append(("metadata_matches", int(matched_metadata)))

    virulence_outputs: dict[str, pd.DataFrame] = {}
    extended_outputs: dict[str, pd.DataFrame] = {}
    sample_qc = build_sample_qc(isolate_summary, None)
    if arguments.virulence_table is not None:
        if not arguments.virulence_table.exists():
            raise FileNotFoundError(f"Virulence table not found: {arguments.virulence_table}")
        metadata_hashes["virulence_table"] = compute_sha256(arguments.virulence_table)
        extra_metadata.append(("virulence_file_name", arguments.virulence_table.name))
        virulence_table, metadata_columns, virulence_columns = load_virulence_table(arguments.virulence_table)
        virulence_outputs = build_virulence_outputs(virulence_table, metadata_columns, virulence_columns)

        overlap = sorted(set(alignment_data.isolate_ids).intersection(virulence_outputs["virulence_profiles"]["isolate_id"]))
        alignment_only = sorted(set(alignment_data.isolate_ids).difference(overlap))
        virulence_only = sorted(set(virulence_outputs["virulence_profiles"]["isolate_id"]).difference(overlap))
        extra_metadata.extend(
            [
                ("virulence_rows", len(virulence_outputs["virulence_profiles"])),
                ("virulence_overlap_isolates", len(overlap)),
                ("alignment_only_isolates", len(alignment_only)),
                ("virulence_only_isolates", len(virulence_only)),
            ]
        )
        data_checks.extend(
            [
                {"category": "alignment_only_isolates", "value": ", ".join(alignment_only[:50])},
                {"category": "virulence_only_isolates", "value": ", ".join(virulence_only[:50])},
            ]
        )

        overlap_pairs = genetic_distance_pairs[
            genetic_distance_pairs["isolate_id_left"].isin(overlap)
            & genetic_distance_pairs["isolate_id_right"].isin(overlap)
        ].copy()
        overlap_matrix = genetic_distance_matrix.loc[overlap, overlap].copy()
        virulence_matrix = virulence_outputs["virulence_distance_matrix"].loc[overlap, overlap].copy()

        distance_relationship_pairs, distance_relationship_summary = build_distance_relationship_outputs(
            overlap_pairs,
            virulence_outputs["virulence_distance_pairs"],
            virulence_outputs["virulence_profiles"],
        )
        virulence_outputs["distance_relationship_pairs"] = distance_relationship_pairs
        virulence_outputs["distance_relationship_summary"] = distance_relationship_summary

        isolate_summary = isolate_summary.merge(
            virulence_outputs["virulence_isolate_summary"],
            on="isolate_id",
            how="left",
            suffixes=("", "_virulence"),
        )
        sample_qc = build_sample_qc(isolate_summary, virulence_outputs["virulence_profiles"])

        site_association_by_host = build_site_association_by_host(alignment_data, virulence_outputs["virulence_profiles"])
        site_association_by_host_exact = build_site_association_by_host_exact(alignment_data, virulence_outputs["virulence_profiles"])

        profile_frame = _profile_signature_frame(virulence_outputs["virulence_profiles"]).set_index("isolate_id")
        pathotype_series = virulence_outputs["virulence_profiles"].set_index("isolate_id")["pathotype"]
        profile_series = profile_frame["profile_signature"]

        pathotype_permutation = label_separation_permutation_test(
            overlap_matrix.to_numpy(dtype=float),
            pathotype_series.loc[overlap].tolist(),
            permutations=arguments.permutations,
            random_seed=arguments.random_seed + 101,
            label_name="pathotype",
        )
        profile_permutation = label_separation_permutation_test(
            overlap_matrix.to_numpy(dtype=float),
            profile_series.loc[overlap].tolist(),
            permutations=arguments.permutations,
            random_seed=arguments.random_seed + 202,
            label_name="virulence_profile",
        )
        mantel_summary = mantel_permutation_test(
            overlap_matrix.to_numpy(dtype=float),
            virulence_matrix.to_numpy(dtype=float),
            permutations=arguments.permutations,
            random_seed=arguments.random_seed + 303,
        )

        host_separation_tests = build_host_separation_tests(
            overlap_matrix,
            virulence_outputs["virulence_profiles"],
            permutations=arguments.permutations,
            random_seed=arguments.random_seed + 404,
        )
        prediction_summary, prediction_detail = build_prediction_outputs(
            overlap_pairs,
            virulence_outputs["virulence_profiles"],
        )

        top_site_associations_exact = build_top_site_associations_exact(site_association_by_host_exact, top_n=arguments.top_site_count)
        leave_one_out_influence = build_leave_one_out_influence(
            overlap_matrix,
            virulence_outputs["virulence_profiles"],
            virulence_outputs["virulence_distance_matrix"],
        )
        host_signal_overview = build_host_signal_overview(
            virulence_outputs["virulence_profiles"],
            host_separation_tests,
            site_association_by_host_exact,
        )
        extended_outputs = {
            "analysis_cohorts": build_analysis_cohorts(alignment_data.isolate_ids, virulence_outputs["virulence_profiles"]),
            "virulence_subset_genetic_matrix": overlap_matrix.reset_index(),
            "virulence_subset_genetic_pairs": overlap_pairs,
            "sample_qc": sample_qc,
            "site_qc": site_qc,
            "pathotype_distance_summary": build_binary_distance_summary(
                distance_relationship_pairs, "same_pathotype", "same_pathotype"
            ),
            "profile_distance_summary": build_binary_distance_summary(
                distance_relationship_pairs, "same_virulence_profile", "same_virulence_profile"
            ),
            "pair_class_summary": build_pair_class_summary(distance_relationship_pairs),
            "pathotype_pair_summary": build_pathotype_pair_summary(distance_relationship_pairs),
            "nearest_neighbor_summary": build_nearest_neighbor_summary(
                overlap_pairs,
                virulence_outputs["virulence_profiles"],
                virulence_outputs["virulence_distance_pairs"],
            ),
            "host_separation_summary": build_host_separation_summary(
                distance_relationship_pairs, virulence_outputs["virulence_profiles"]
            ),
            "site_association_by_host": site_association_by_host,
            "top_site_associations": build_top_site_associations(site_association_by_host),
            "pathotype_support_summary": build_pathotype_support_summary(virulence_outputs["virulence_profiles"]),
            "mantel_summary": mantel_summary,
            "pathotype_permutation": pathotype_permutation,
            "profile_permutation": profile_permutation,
            "host_separation_tests": host_separation_tests,
            "site_association_by_host_exact": site_association_by_host_exact,
            "top_site_associations_exact": top_site_associations_exact,
            "prediction_summary": prediction_summary,
            "prediction_detail": prediction_detail,
            "profile_complexity_summary": build_profile_complexity_summary(virulence_outputs["virulence_profiles"]),
            "host_signal_overview": host_signal_overview,
            "top_host_site_table": build_top_host_site_table(site_association_by_host_exact),
            "leave_one_out_influence": leave_one_out_influence,
            "primary_summary": build_primary_summary(
                alignment_summary,
                virulence_outputs["virulence_profiles"],
                mantel_summary,
                pathotype_permutation,
                profile_permutation,
                prediction_summary,
                host_separation_tests,
                site_association_by_host_exact,
                leave_one_out_influence,
            ),
        }

        dataset_table = build_dataset_table(
            alignment_summary,
            sample_qc,
            site_qc,
            virulence_outputs["virulence_profiles"],
            virulence_outputs.get("pathotype_summary", pd.DataFrame()),
        )
        host_results = build_host_results(
            host_signal_overview,
            extended_outputs["top_host_site_table"],
        )
        pathotype_results = build_pathotype_results(
            virulence_outputs.get("pathotype_summary", pd.DataFrame()),
            extended_outputs["pathotype_distance_summary"],
            extended_outputs["profile_distance_summary"],
            prediction_summary,
        )
        interpretation_audit = build_interpretation_audit(
            extended_outputs["primary_summary"],
            host_results,
            top_site_associations_exact,
            prediction_summary,
        )
        extended_outputs.update(
            {
                "dataset_table": dataset_table,
                "host_results": host_results,
                "pathotype_results": pathotype_results,
                "interpretation_audit": interpretation_audit,
                "narrative_support": build_narrative_support(
                    extended_outputs["primary_summary"],
                    host_results,
                    interpretation_audit,
                ),
                "sheet_guide": build_sheet_guide(),
            }
        )
    else:
        extended_outputs = {
            "sample_qc": sample_qc,
            "site_qc": site_qc,
        }

    run_metadata = build_run_metadata(
        package_name="colletotrichum_sublineola_pipeline",
        package_version=__version__,
        input_hashes=metadata_hashes,
        extra_values=extra_metadata,
    )

    sheet_mapping.extend(
        [
            ("RunSummary", run_metadata),
            ("AlignmentSummary", alignment_summary),
            ("IsolateSummary", isolate_summary),
            ("SiteSummary", site_summary),
            ("GeneticDistanceMatrix", genetic_distance_matrix.reset_index()),
            ("GeneticDistancePairs", genetic_distance_pairs),
            ("SampleQC", extended_outputs["sample_qc"]),
            ("SiteQC", extended_outputs["site_qc"]),
        ]
    )

    if data_checks:
        sheet_mapping.append(("DataChecks", pd.DataFrame(data_checks)))

    if virulence_outputs:
        sheet_mapping.extend(
            [
                ("VirulenceProfiles", virulence_outputs["virulence_profiles"]),
                ("VirulenceIsolates", virulence_outputs["virulence_isolate_summary"]),
                ("VirulenceHosts", virulence_outputs["virulence_host_summary"]),
                ("UniqueVirulence", virulence_outputs["virulence_unique_profiles"]),
                ("VirulenceDistanceMatrix", virulence_outputs["virulence_distance_matrix"].reset_index()),
                ("VirulenceDistancePairs", virulence_outputs["virulence_distance_pairs"]),
            ]
        )
        if not virulence_outputs["pathotype_summary"].empty:
            sheet_mapping.append(("PathotypeSummary", virulence_outputs["pathotype_summary"]))
        sheet_mapping.extend(
            [
                ("DistanceRelationship", virulence_outputs["distance_relationship_pairs"]),
                ("DistanceRelationStats", virulence_outputs["distance_relationship_summary"]),
                ("AnalysisCohorts", extended_outputs["analysis_cohorts"]),
                ("VirSubsetGeneticMatrix", extended_outputs["virulence_subset_genetic_matrix"]),
                ("VirSubsetGeneticPairs", extended_outputs["virulence_subset_genetic_pairs"]),
                ("PathotypeDistanceSum", extended_outputs["pathotype_distance_summary"]),
                ("ProfileDistanceSummary", extended_outputs["profile_distance_summary"]),
                ("PairClassSummary", extended_outputs["pair_class_summary"]),
                ("PathotypePairSummary", extended_outputs["pathotype_pair_summary"]),
                ("NearestNeighborSummary", extended_outputs["nearest_neighbor_summary"]),
                ("HostSeparationSummary", extended_outputs["host_separation_summary"]),
                ("SiteAssociationByHost", extended_outputs["site_association_by_host"]),
                ("TopSiteAssociations", extended_outputs["top_site_associations"]),
                ("PathotypeSupport", extended_outputs["pathotype_support_summary"]),
                ("MantelCorrelation", extended_outputs["mantel_summary"]),
                ("PathotypePermTest", extended_outputs["pathotype_permutation"]),
                ("ProfilePermTest", extended_outputs["profile_permutation"]),
                ("HostSeparationTests", extended_outputs["host_separation_tests"]),
                ("SiteAssocByHostExact", extended_outputs["site_association_by_host_exact"]),
                ("TopSiteAssocExact", extended_outputs["top_site_associations_exact"]),
                ("PredictionSummary", extended_outputs["prediction_summary"]),
                ("PredictionDetail", extended_outputs["prediction_detail"]),
                ("PrimarySummary", extended_outputs["primary_summary"]),
                ("ProfileComplexity", extended_outputs["profile_complexity_summary"]),
                ("HostOverview", extended_outputs["host_signal_overview"]),
                ("TopHostSites", extended_outputs["top_host_site_table"]),
                ("LeaveOneOutInfluence", extended_outputs["leave_one_out_influence"]),
                ("DatasetTable", extended_outputs["dataset_table"]),
                ("HostResults", extended_outputs["host_results"]),
                ("PathotypeResults", extended_outputs["pathotype_results"]),
                ("InterpretationAudit", extended_outputs["interpretation_audit"]),
                ("NarrativeSupport", extended_outputs["narrative_support"]),
                ("SheetGuide", extended_outputs["sheet_guide"]),
            ]
        )

    write_excel_workbook(sheet_mapping, arguments.output)


if __name__ == "__main__":
    main()
