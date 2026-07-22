from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis import (
    alignment_summaries,
    marker_distance_pairs,
    matrix_correlation_test,
    merge_marker_virulence_pairs,
    pathotype_separation_test,
    square_distance_matrix,
    virulence_outputs,
)
from .io import read_alignment, read_virulence_table
from .reporting import write_output_workbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RAD-seq-derived marker-profile distances with differential-host "
            "virulence distances in Colletotrichum sublineola."
        )
    )
    parser.add_argument("--alignment", required=True, help="Equal-length FASTA alignment.")
    parser.add_argument("--virulence-table", required=True, help="CSV, TSV, or Excel R/S table.")
    parser.add_argument("--output", required=True, help="Output Excel workbook.")
    parser.add_argument("--isolate-column", default="isolate_id")
    parser.add_argument("--pathotype-column", default="pathotype")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be at least 1.")

    isolate_ids, sequences = read_alignment(args.alignment)
    virulence_table, isolate_col, pathotype_col, host_columns = read_virulence_table(
        args.virulence_table,
        isolate_column=args.isolate_column,
        pathotype_column=args.pathotype_column,
    )
    profiles, virulence_counts, virulence_pairs, assignments = virulence_outputs(
        virulence_table, isolate_col, pathotype_col, host_columns
    )
    pathotype_map = profiles.set_index("Isolate ID")["Pathotype"].to_dict()
    alignment_summary, isolate_summary, position_summary = alignment_summaries(
        isolate_ids,
        sequences,
        pathotype_map=pathotype_map,
        virulence_counts=virulence_counts,
    )
    alignment_summary = pd.concat(
        [
            alignment_summary,
            pd.DataFrame(
                [
                    ["Virulence panel", "Isolates evaluated on host differentials", len(profiles)],
                    ["Virulence panel", "Host differentials evaluated", len(host_columns)],
                    ["Pathotype diversity", "Published pathotypes represented", profiles["Pathotype"].replace("", pd.NA).nunique()],
                    ["Pathotype diversity", "Singleton pathotypes", int((profiles["Pathotype"].value_counts() == 1).sum())],
                ],
                columns=["Category", "Metric", "Value"],
            ),
        ],
        ignore_index=True,
    )

    all_marker_pairs = marker_distance_pairs(isolate_ids, sequences)
    shared = sorted(isolate_id for isolate_id in profiles["Isolate ID"] if isolate_id in set(isolate_ids))
    if len(shared) != len(profiles):
        missing = sorted(set(profiles["Isolate ID"]) - set(shared))
        raise ValueError("Virulence isolates missing from alignment: " + ", ".join(missing))

    shared_set = set(shared)
    marker_pairs = all_marker_pairs[
        all_marker_pairs["Isolate 1"].isin(shared_set)
        & all_marker_pairs["Isolate 2"].isin(shared_set)
    ].reset_index(drop=True)
    merged_pairs = merge_marker_virulence_pairs(marker_pairs, virulence_pairs, pathotype_map)
    marker_matrix = square_distance_matrix(shared, marker_pairs, "Marker distance")
    virulence_matrix = square_distance_matrix(shared, virulence_pairs, "Virulence distance")
    correlation_tests = matrix_correlation_test(
        marker_matrix,
        virulence_matrix,
        permutations=args.permutations,
        random_seed=args.random_seed + 303,
    )
    separation_test = pathotype_separation_test(
        marker_matrix,
        shared,
        pathotype_map,
        permutations=args.permutations,
        random_seed=args.random_seed + 101,
    )
    alignment_summary = pd.concat(
        [
            alignment_summary,
            pd.DataFrame(
                [
                    ["Pathotype comparison", "Within-pathotype isolate pairs", int(separation_test.iloc[0]["Within-pathotype pairs"])],
                    ["Pathotype comparison", "Between-pathotype isolate pairs", int(separation_test.iloc[0]["Between-pathotype pairs"])],
                ],
                columns=["Category", "Metric", "Value"],
            ),
        ],
        ignore_index=True,
    )

    write_output_workbook(
        [
            ("Dataset_Summary", alignment_summary),
            ("Isolate_Summary", isolate_summary),
            ("Marker_Position_Summary", position_summary),
            ("Virulence_Profiles", profiles),
            ("Pathotype_Assignments", assignments),
            ("Marker_Distance_Pairs", marker_pairs),
            ("Virulence_Distance_Pairs", virulence_pairs),
            ("Marker_Virulence_Pairs", merged_pairs),
            ("Matrix_Correlation_Tests", correlation_tests),
            ("Pathotype_Separation_Test", separation_test),
        ],
        args.output,
    )


if __name__ == "__main__":
    main()
