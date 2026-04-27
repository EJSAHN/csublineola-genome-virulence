from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .common import normalise_identifier, read_table


def load_virulence_table(file_path):
    dataframe = read_table(file_path).copy()
    if "isolate_id" not in dataframe.columns:
        raise ValueError("The virulence table must contain an 'isolate_id' column.")

    dataframe["isolate_id"] = dataframe["isolate_id"].map(normalise_identifier)
    virulence_columns = []
    metadata_columns = []
    for column in dataframe.columns:
        if column == "isolate_id":
            continue
        values = set(dataframe[column].dropna().astype(str).str.strip().str.upper())
        if values and values.issubset({"R", "S"}):
            virulence_columns.append(column)
        else:
            metadata_columns.append(column)

    if not virulence_columns:
        raise ValueError("The virulence table does not contain any columns encoded with R/S values.")

    virulence_dataframe = dataframe[["isolate_id", *metadata_columns, *virulence_columns]].copy()
    return virulence_dataframe, metadata_columns, virulence_columns


def build_virulence_outputs(virulence_dataframe: pd.DataFrame, metadata_columns: list[str], virulence_columns: list[str]):
    cleaned = virulence_dataframe.copy()
    for column in virulence_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip().str.upper()

    binary_matrix = cleaned[virulence_columns].apply(lambda column: column.map({"R": 0, "S": 1}))
    isolate_summary = cleaned[["isolate_id", *metadata_columns]].copy()
    isolate_summary["susceptible_count"] = (binary_matrix == 1).sum(axis=1)
    isolate_summary["resistant_count"] = (binary_matrix == 0).sum(axis=1)

    host_summary = pd.DataFrame(
        {
            "host_differential": virulence_columns,
            "susceptible_isolates": (binary_matrix == 1).sum(axis=0).values,
            "resistant_isolates": (binary_matrix == 0).sum(axis=0).values,
        }
    )

    profile_signatures = cleaned[virulence_columns].apply(lambda row: "|".join(row.astype(str)), axis=1)
    profile_table = cleaned[["isolate_id", *metadata_columns]].copy()
    profile_table["profile_signature"] = profile_signatures

    grouped_profiles = profile_table.groupby("profile_signature", dropna=False)
    unique_profiles = grouped_profiles.agg(
        isolate_count=("isolate_id", "size"),
        isolate_ids=("isolate_id", lambda values: ", ".join(sorted(values))),
    ).reset_index()

    if "pathotype" in cleaned.columns:
        unique_profiles = unique_profiles.merge(
            cleaned[["pathotype"]].assign(profile_signature=profile_signatures).drop_duplicates(),
            on="profile_signature",
            how="left",
        )

    isolate_ids = cleaned["isolate_id"].tolist()
    distance_matrix = np.full((len(isolate_ids), len(isolate_ids)), np.nan, dtype=float)
    comparable_matrix = np.zeros((len(isolate_ids), len(isolate_ids)), dtype=int)

    values = cleaned[virulence_columns].to_numpy(dtype=object)
    valid = np.isin(values, ["R", "S"])

    for index in range(len(isolate_ids)):
        distance_matrix[index, index] = 0.0
        comparable_matrix[index, index] = int(valid[index].sum())

    pairwise_rows = []
    for left_index, left_isolate in enumerate(isolate_ids):
        for right_index in range(left_index + 1, len(isolate_ids)):
            comparable = valid[left_index] & valid[right_index]
            comparable_hosts = int(comparable.sum())
            mismatches = int(np.count_nonzero(values[left_index, comparable] != values[right_index, comparable]))
            distance = np.nan if comparable_hosts == 0 else mismatches / comparable_hosts
            distance_matrix[left_index, right_index] = distance
            distance_matrix[right_index, left_index] = distance
            comparable_matrix[left_index, right_index] = comparable_hosts
            comparable_matrix[right_index, left_index] = comparable_hosts

            pairwise_rows.append(
                {
                    "isolate_id_left": left_isolate,
                    "isolate_id_right": isolate_ids[right_index],
                    "comparable_hosts": comparable_hosts,
                    "virulence_mismatches": mismatches,
                    "virulence_distance": distance,
                }
            )

    matrix_dataframe = pd.DataFrame(distance_matrix, index=isolate_ids, columns=isolate_ids)
    matrix_dataframe.index.name = "isolate_id"
    pairwise_dataframe = pd.DataFrame(pairwise_rows)

    pathotype_summary = pd.DataFrame()
    if "pathotype" in cleaned.columns:
        pathotype_summary = (
            cleaned.groupby("pathotype", dropna=False)
            .agg(
                isolate_count=("isolate_id", "size"),
                isolate_ids=("isolate_id", lambda values: ", ".join(sorted(values))),
            )
            .reset_index()
        )

    return {
        "virulence_profiles": cleaned,
        "virulence_isolate_summary": isolate_summary.sort_values("isolate_id").reset_index(drop=True),
        "virulence_host_summary": host_summary,
        "virulence_unique_profiles": unique_profiles.sort_values(["isolate_count", "profile_signature"], ascending=[False, True]),
        "virulence_distance_matrix": matrix_dataframe,
        "virulence_distance_pairs": pairwise_dataframe,
        "pathotype_summary": pathotype_summary,
    }


def build_distance_relationship_outputs(
    genetic_distance_pairs: pd.DataFrame,
    virulence_distance_pairs: pd.DataFrame,
    virulence_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = genetic_distance_pairs.merge(
        virulence_distance_pairs,
        on=["isolate_id_left", "isolate_id_right"],
        how="inner",
    )

    metadata = virulence_profiles[["isolate_id"]].copy()
    if "pathotype" in virulence_profiles.columns:
        metadata["pathotype"] = virulence_profiles["pathotype"]
    else:
        metadata["pathotype"] = pd.NA

    merged = merged.merge(
        metadata.rename(columns={"isolate_id": "isolate_id_left", "pathotype": "pathotype_left"}),
        on="isolate_id_left",
        how="left",
    ).merge(
        metadata.rename(columns={"isolate_id": "isolate_id_right", "pathotype": "pathotype_right"}),
        on="isolate_id_right",
        how="left",
    )
    merged["same_pathotype"] = merged["pathotype_left"].eq(merged["pathotype_right"])

    profile_columns = [column for column in virulence_profiles.columns if column not in {"isolate_id", "pathotype"} and not virulence_profiles[column].dropna().empty and set(virulence_profiles[column].dropna().astype(str).str.upper()).issubset({"R", "S"})]
    profile_signature = virulence_profiles[profile_columns].apply(lambda row: "|".join(row.astype(str)), axis=1)
    signature_map = dict(zip(virulence_profiles["isolate_id"], profile_signature))
    merged["same_virulence_profile"] = merged["isolate_id_left"].map(signature_map).eq(merged["isolate_id_right"].map(signature_map))

    summary_rows = [
        {
            "metric": "pair_count",
            "value": int(len(merged)),
        }
    ]
    valid_rows = merged[["genetic_distance", "virulence_distance"]].dropna()
    if not valid_rows.empty:
        summary_rows.extend(
            [
                {
                    "metric": "pearson_correlation",
                    "value": valid_rows["genetic_distance"].corr(valid_rows["virulence_distance"], method="pearson"),
                },
                {
                    "metric": "spearman_correlation",
                    "value": valid_rows["genetic_distance"].corr(valid_rows["virulence_distance"], method="spearman"),
                },
                {
                    "metric": "mean_genetic_distance",
                    "value": valid_rows["genetic_distance"].mean(),
                },
                {
                    "metric": "mean_virulence_distance",
                    "value": valid_rows["virulence_distance"].mean(),
                },
            ]
        )

    grouped_summary = (
        merged.groupby(["same_pathotype", "same_virulence_profile"], dropna=False)
        .agg(
            pair_count=("genetic_distance", "size"),
            mean_genetic_distance=("genetic_distance", "mean"),
            mean_virulence_distance=("virulence_distance", "mean"),
        )
        .reset_index()
    )

    return merged, pd.concat([pd.DataFrame(summary_rows), grouped_summary], ignore_index=True, sort=False)
