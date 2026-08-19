from __future__ import annotations

import hashlib
import platform
from collections import Counter, defaultdict, deque
from importlib import metadata as importlib_metadata
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .distance import pairwise_ibs_distance, pcoa, upper_triangle, virulence_hamming_distance
from .io import HapMapData, read_hapmap, read_validation_panel_400, read_isolate_metadata, read_virulence
from .panel_design import (
    augment_panel_farthest_first,
    optimize_farthest_first_panel,
    panel_membership_table,
    panel_metrics as optimized_panel_metrics,
)
from .statistics import (
    knn_permutation_test,
    knn_summary,
    majority_baseline,
    marker_panel_metrics,
    matrix_correlation_permutation,
    panel_randomization_audit,
    pathotype_separation_permutation,
    permanova_permutation,
    permanova_pseudo_f,
    permanova_summary,
    permdisp_summary,
    origin_distance_contrast_permutation,
)

LOGGER = logging.getLogger("csub_snp_virulence")
MARKER_SET_ORDER = ["reconstructed_1244", "literal_lt10pct_1135", "low_missing_400"]
MARKER_SET_LABELS = {
    "reconstructed_1244": "Reconstructed 1,244-marker panel",
    "literal_lt10pct_1135": "Literal <10% missing",
    "low_missing_400": "Low-missingness 400-marker panel",
}
FILTER_THRESHOLDS = [0, 1, 2, 3, 4, 5, 7, 10, 13, 14, 15, 20, 25, 30, 40, 50, 60, 70]
PANEL_METRIC_NAMES = [
    "mean_within_distance",
    "median_within_distance",
    "mean_unselected_nearest_distance",
    "p95_unselected_nearest_distance",
    "max_unselected_nearest_distance",
    "median_unselected_nearest_distance",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize type {type(value)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path, index_label="isolate_id")


def _marker_masks(hapmap: HapMapData) -> dict[str, np.ndarray]:
    maf_ok = np.isfinite(hapmap.maf) & (hapmap.maf >= 0.05)
    masks = {
        "reconstructed_1244": maf_ok & (hapmap.calculated_missing <= 15),
        "literal_lt10pct_1135": maf_ok & ((hapmap.calculated_missing / len(hapmap.sample_ids)) < 0.10),
        "low_missing_400": maf_ok & (hapmap.calculated_missing <= 2),
    }
    expected = {
        "reconstructed_1244": 1244,
        "literal_lt10pct_1135": 1135,
        "low_missing_400": 400,
    }
    observed = {key: int(value.sum()) for key, value in masks.items()}
    if observed != expected:
        raise ValueError(
            "The input does not reproduce the expected marker counts. "
            f"Expected {expected}; observed {observed}. Confirm that the correct HapMap file is being used."
        )
    return masks



def _canonical_genotype_call(value: object) -> str:
    call = str(value).strip().upper()
    if len(call) != 2 or "N" in call:
        return "NN"
    return "".join(sorted(call))


def _validate_marker_panel_400(
    hapmap: HapMapData,
    strict_mask: np.ndarray,
    validation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    records = read_validation_panel_400(validation_path)
    if set(records) != set(hapmap.sample_ids):
        raise ValueError(
            "Validation-panel isolate IDs do not match the HapMap samples. "
            f"Missing={sorted(set(hapmap.sample_ids)-set(records))}; "
            f"extra={sorted(set(records)-set(hapmap.sample_ids))}"
        )

    strict = hapmap.markers.loc[strict_mask].copy().reset_index(drop=False).rename(columns={"index": "source_row_index"})
    source_signatures: list[tuple[str, ...]] = []
    for _, row in strict.iterrows():
        source_signatures.append(tuple(_canonical_genotype_call(row[sample]) for sample in hapmap.sample_ids))

    validation_matrix = np.array([records[sample] for sample in hapmap.sample_ids], dtype=object)
    validation_signatures = [
        tuple(_canonical_genotype_call(value) for value in validation_matrix[:, column])
        for column in range(validation_matrix.shape[1])
    ]

    exact_multiset_match = Counter(source_signatures) == Counter(validation_signatures)
    pools: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for source_index, signature in enumerate(source_signatures):
        pools[signature].append(source_index)

    mapping_rows: list[dict[str, Any]] = []
    unmatched = 0
    for validation_index, signature in enumerate(validation_signatures):
        if not pools[signature]:
            unmatched += 1
            mapping_rows.append({"validation_column_1based": validation_index + 1, "matched": False})
            continue
        source_index = pools[signature].popleft()
        row = strict.iloc[source_index]
        mapping_rows.append(
            {
                "validation_column_1based": validation_index + 1,
                "matched": True,
                "hapmap_source_row_0based": int(row["source_row_index"]),
                "marker_id": row["rs"],
                "scaffold": row["chrom"],
                "position": int(row["pos"]),
                "alleles": row["alleles"],
                "missing_calls": int(row["calculated_missing_calls"]),
                "maf": float(row["calculated_maf"]),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(output_dir / "tables" / "validation_panel_400_exact_mapping.csv", index=False)

    summary = {
        "validation_panel_file": str(validation_path.resolve()),
        "validation_panel_sha256": _sha256(validation_path),
        "validation_panel_isolates": len(records),
        "validation_panel_markers": validation_matrix.shape[1],
        "strict_hapmap_markers": int(strict_mask.sum()),
        "exact_genotype_pattern_multiset_match": bool(exact_multiset_match),
        "unmatched_validation_columns": int(unmatched),
        "unique_validation_patterns": int(len(set(validation_signatures))),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "tables" / "validation_panel_400_summary.csv", index=False)
    if not exact_multiset_match or unmatched:
        raise ValueError(f"400-marker validation-panel check failed: {summary}")
    return summary


def _software_versions() -> pd.DataFrame:
    rows = [
        {"component": "pipeline", "version": "1.0.0"},
        {"component": "Python", "version": platform.python_version()},
        {"component": "operating_system", "version": platform.platform()},
    ]
    for package in ["numpy", "pandas", "scipy"]:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"component": package, "version": version})
    return pd.DataFrame(rows)


def _run_mgc_worker(
    arrays_path: Path,
    x_key: str,
    output_path: Path,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    worker = Path(__file__).with_name("mgc_worker.py")
    command = [
        sys.executable,
        str(worker),
        "--input",
        str(arrays_path),
        "--x-key",
        x_key,
        "--y-key",
        "virulence",
        "--reps",
        str(reps),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    LOGGER.info("Running MGC worker for %s with %s permutations", x_key, reps)
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _distance_pairs_all(sample_ids: list[str], distances: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, j in zip(*np.triu_indices(len(sample_ids), 1), strict=True):
        row: dict[str, Any] = {"isolate_1": sample_ids[i], "isolate_2": sample_ids[j]}
        for marker_set, matrix in distances.items():
            short = {"reconstructed_1244": "1244", "literal_lt10pct_1135": "1135", "low_missing_400": "400"}[marker_set]
            row[f"marker_distance_{short}"] = float(matrix[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def _distance_pairs_selected(
    isolate_ids: list[str],
    pathotypes: np.ndarray,
    origins: np.ndarray,
    marker_distances: dict[str, np.ndarray],
    virulence_distance: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, j in zip(*np.triu_indices(len(isolate_ids), 1), strict=True):
        row: dict[str, Any] = {
            "isolate_1": isolate_ids[i],
            "isolate_2": isolate_ids[j],
            "origin_1": origins[i],
            "origin_2": origins[j],
            "same_origin": bool(origins[i] == origins[j]),
            "pathotype_1": pathotypes[i],
            "pathotype_2": pathotypes[j],
            "same_pathotype": bool(pathotypes[i] == pathotypes[j]),
            "virulence_distance": float(virulence_distance[i, j]),
        }
        for marker_set, matrix in marker_distances.items():
            short = {"reconstructed_1244": "1244", "literal_lt10pct_1135": "1135", "low_missing_400": "400"}[marker_set]
            row[f"marker_distance_{short}"] = float(matrix[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(
    hapmap_path: Path,
    virulence_path: Path,
    metadata_path: Path,
    output_dir: Path,
    validation_panel_path: Path | None = None,
    mode: str = "full",
    seed: int = 1729,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for subdir in ["tables", "matrices", "filtered_hapmap", "null_distributions", "logs", "work"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "logs" / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    LOGGER.info("Starting C. sublineola SNP-virulence pipeline")
    LOGGER.info("HapMap input: %s", hapmap_path)
    LOGGER.info("Output: %s", output_dir)

    if mode not in {"full", "fast"}:
        raise ValueError("mode must be 'full' or 'fast'")
    permutation_reps = 20000 if mode == "full" else 1000
    mgc_reps = 20000 if mode == "full" else 1000
    panel_reps = 20000 if mode == "full" else 1000

    hapmap = read_hapmap(hapmap_path)
    virulence = read_virulence(virulence_path)
    metadata = read_isolate_metadata(metadata_path, hapmap.sample_ids)
    masks = _marker_masks(hapmap)
    validation_summary = None
    if validation_panel_path is not None:
        validation_summary = _validate_marker_panel_400(
            hapmap, masks["low_missing_400"], Path(validation_panel_path), output_dir
        )
    _software_versions().to_csv(output_dir / "tables" / "software_versions.csv", index=False)

    selected_indices = np.array([hapmap.sample_ids.index(isolate) for isolate in virulence.isolate_ids], dtype=int)
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Duplicate virulence isolate IDs were detected.")
    selected_metadata = metadata.iloc[selected_indices].reset_index(drop=True).copy()
    selected_metadata["pathotype"] = virulence.pathotypes
    selected_origins = selected_metadata["origin"].to_numpy()
    virulence_distance = virulence_hamming_distance(virulence.response)

    marker_distances_full: dict[str, np.ndarray] = {}
    marker_overlaps_full: dict[str, np.ndarray] = {}
    marker_distances_selected: dict[str, np.ndarray] = {}
    marker_set_rows: list[dict[str, Any]] = []

    for marker_set in MARKER_SET_ORDER:
        mask = masks[marker_set]
        distance, overlap = pairwise_ibs_distance(hapmap.dosage[:, mask])
        marker_distances_full[marker_set] = distance
        marker_overlaps_full[marker_set] = overlap
        marker_distances_selected[marker_set] = distance[np.ix_(selected_indices, selected_indices)]
        subset = hapmap.markers.loc[mask]
        triangle = np.triu_indices(len(hapmap.sample_ids), 1)
        marker_set_rows.append(
            {
                "marker_set": marker_set,
                "display_name": MARKER_SET_LABELS[marker_set],
                "marker_count": int(mask.sum()),
                "maximum_missing_calls": int(hapmap.calculated_missing[mask].max()),
                "maximum_missing_fraction": float(hapmap.calculated_missing[mask].max() / len(hapmap.sample_ids)),
                "scaffold_count": int(subset["chrom"].nunique()),
                "minimum_pairwise_overlap": int(overlap[triangle].min()),
                "mean_pairwise_overlap": float(overlap[triangle].mean()),
                "mean_ibs_distance_all_140": float(distance[triangle].mean()),
                "median_ibs_distance_all_140": float(np.median(distance[triangle])),
            }
        )
        filtered = hapmap.markers.loc[mask, list(hapmap.markers.columns[: hapmap.markers.columns.get_loc("No. of NN") + 1])]
        filtered.to_csv(output_dir / "filtered_hapmap" / f"{marker_set}.hmp.txt", sep="\t", index=False)
        _write_matrix(output_dir / "matrices" / f"marker_distance_{marker_set}_140.csv", distance, hapmap.sample_ids)
        _write_matrix(output_dir / "matrices" / f"marker_overlap_{marker_set}_140.csv", overlap, hapmap.sample_ids)

    _write_matrix(output_dir / "matrices" / "virulence_distance_30.csv", virulence_distance, virulence.isolate_ids)
    marker_sets_table = pd.DataFrame(marker_set_rows)
    marker_sets_table.to_csv(output_dir / "tables" / "marker_set_summary.csv", index=False)

    marker_metadata = hapmap.markers.loc[masks["reconstructed_1244"], [
        "rs", "alleles", "chrom", "pos", "No. of NN", "calculated_missing_calls", "missing_fraction", "calculated_maf"
    ]].copy()
    marker_metadata.to_csv(output_dir / "tables" / "marker_metadata_reconstructed_1244.csv", index=False)

    # Global and origin-stratified marker–virulence tests.
    concordance_rows: list[dict[str, Any]] = []
    pathotype_rows: list[dict[str, Any]] = []
    correlation_nulls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, marker_set in enumerate(MARKER_SET_ORDER):
        matrix = marker_distances_selected[marker_set]
        result = matrix_correlation_permutation(
            matrix,
            virulence_distance,
            reps=permutation_reps,
            seed=seed + 100 * offset,
        )
        correlation_nulls[marker_set] = (result.pearson_null, result.spearman_null)
        np.save(output_dir / "null_distributions" / f"{marker_set}_pearson.npy", result.pearson_null)
        np.save(output_dir / "null_distributions" / f"{marker_set}_spearman.npy", result.spearman_null)
        concordance_rows.append(
            {
                "marker_set": marker_set,
                "pearson_r": result.pearson_r,
                "pearson_p": result.pearson_p,
                "pearson_null_q025": float(np.quantile(result.pearson_null, 0.025)),
                "pearson_null_q975": float(np.quantile(result.pearson_null, 0.975)),
                "spearman_rho": result.spearman_rho,
                "spearman_p": result.spearman_p,
                "spearman_null_q025": float(np.quantile(result.spearman_null, 0.025)),
                "spearman_null_q975": float(np.quantile(result.spearman_null, 0.975)),
                "permutations": permutation_reps,
            }
        )
        sep = pathotype_separation_permutation(
            matrix,
            virulence.pathotypes,
            reps=permutation_reps,
            seed=seed + 100 * offset + 2,
        )
        np.save(output_dir / "null_distributions" / f"{marker_set}_pathotype_separation.npy", sep.null)
        pathotype_rows.append(
            {
                "marker_set": marker_set,
                "within_pairs": sep.within_n,
                "between_pairs": sep.between_n,
                "within_mean": sep.within_mean,
                "between_mean": sep.between_mean,
                "between_minus_within": sep.difference,
                "p_value": sep.p_value,
                "null_q025": float(np.quantile(sep.null, 0.025)),
                "null_q975": float(np.quantile(sep.null, 0.975)),
                "permutations": permutation_reps,
            }
        )

    concordance = pd.DataFrame(concordance_rows)
    pathotype_table = pd.DataFrame(pathotype_rows)
    concordance.to_csv(output_dir / "tables" / "global_concordance_tests.csv", index=False)
    pathotype_table.to_csv(output_dir / "tables" / "pathotype_separation_tests.csv", index=False)

    stratified = matrix_correlation_permutation(
        marker_distances_selected["reconstructed_1244"],
        virulence_distance,
        reps=permutation_reps,
        seed=seed + 11,
        groups=selected_origins,
    )
    pd.DataFrame([
        {
            "analysis": "origin_stratified",
            "marker_set": "reconstructed_1244",
            "pearson_r": stratified.pearson_r,
            "pearson_p": stratified.pearson_p,
            "spearman_rho": stratified.spearman_rho,
            "spearman_p": stratified.spearman_p,
            "permutations": permutation_reps,
        }
    ]).to_csv(output_dir / "tables" / "origin_stratified_concordance.csv", index=False)

    # MGC in isolated worker processes to keep memory use deterministic.
    mgc_arrays_path = output_dir / "work" / "mgc_matrices.npz"
    np.savez_compressed(
        mgc_arrays_path,
        virulence=virulence_distance,
        **{f"marker_{key}": value for key, value in marker_distances_selected.items()},
    )
    mgc_rows: list[dict[str, Any]] = []
    for offset, marker_set in enumerate(MARKER_SET_ORDER):
        result_path = output_dir / "work" / f"mgc_{marker_set}.json"
        result = _run_mgc_worker(
            mgc_arrays_path,
            f"marker_{marker_set}",
            result_path,
            reps=mgc_reps,
            seed=seed + 300 + offset,
        )
        result["marker_set"] = marker_set
        mgc_rows.append(result)
    mgc_table = pd.DataFrame(mgc_rows)
    mgc_table.to_csv(output_dir / "tables" / "multiscale_graph_correlation.csv", index=False)

    # Geographic structure of marker and virulence distances. PERMANOVA is
    # accompanied by PERMDISP and a direct within- versus between-origin
    # distance contrast so that centroid separation is not conflated with
    # unequal multivariate dispersion.
    origin_structure_rows: list[dict[str, Any]] = []
    origin_dispersion_rows: list[dict[str, Any]] = []
    for offset, (distance_type, matrix) in enumerate([
        ("marker_1244", marker_distances_selected["reconstructed_1244"]),
        ("virulence", virulence_distance),
    ]):
        permanova_result, permanova_null = permanova_summary(
            matrix, selected_origins, reps=permutation_reps, seed=seed + 401 + 10 * offset
        )
        permdisp_result, permdisp_null, centroid_distances = permdisp_summary(
            matrix, selected_origins, reps=permutation_reps, seed=seed + 402 + 10 * offset
        )
        contrast_result, contrast_null = origin_distance_contrast_permutation(
            matrix, selected_origins, reps=permutation_reps, seed=seed + 403 + 10 * offset
        )
        np.save(output_dir / "null_distributions" / f"origin_permanova_{distance_type}.npy", permanova_null)
        np.save(output_dir / "null_distributions" / f"origin_permdisp_{distance_type}.npy", permdisp_null)
        np.save(output_dir / "null_distributions" / f"origin_distance_contrast_{distance_type}.npy", contrast_null)
        origin_structure_rows.append(
            {
                "distance_type": distance_type,
                "permanova_F": permanova_result["pseudo_F"],
                "permanova_R2": permanova_result["R2"],
                "permanova_p": permanova_result["p_value"],
                "permdisp_F": permdisp_result["F"],
                "permdisp_p": permdisp_result["p_value"],
                "within_origin_mean": contrast_result["within_origin_mean"],
                "between_origin_mean": contrast_result["between_origin_mean"],
                "between_minus_within": contrast_result["between_minus_within"],
                "distance_contrast_p": contrast_result["p_value"],
                "permutations": permutation_reps,
            }
        )
        for isolate_id, origin, value in zip(
            virulence.isolate_ids, selected_origins, centroid_distances, strict=True
        ):
            origin_dispersion_rows.append(
                {
                    "distance_type": distance_type,
                    "isolate_id": isolate_id,
                    "origin": origin,
                    "distance_to_origin_centroid": float(value),
                }
            )
    origin_structure_table = pd.DataFrame(origin_structure_rows)
    origin_structure_table.to_csv(output_dir / "tables" / "origin_structure_tests.csv", index=False)
    pd.DataFrame(origin_dispersion_rows).to_csv(
        output_dir / "tables" / "origin_distances_to_centroid.csv", index=False
    )

    # Prediction benchmark; unrestricted and origin-stratified k=1 permutation tests.
    majority = majority_baseline(virulence.response)
    knn_rows: list[dict[str, Any]] = []
    summaries = knn_summary(marker_distances_selected["reconstructed_1244"], virulence.response, [1, 3, 5, 7, 9])
    k1_observed, k1_p, k1_null = knn_permutation_test(
        marker_distances_selected["reconstructed_1244"], virulence.response, 1, permutation_reps, seed + 501
    )
    _, k1_stratified_p, k1_stratified_null = knn_permutation_test(
        marker_distances_selected["reconstructed_1244"], virulence.response, 1, permutation_reps, seed + 502, groups=selected_origins
    )
    np.save(output_dir / "null_distributions" / "knn_k1_unrestricted.npy", k1_null)
    np.save(output_dir / "null_distributions" / "knn_k1_origin_stratified.npy", k1_stratified_null)
    for summary in summaries:
        row = {
            "k": summary["k"],
            "micro_accuracy": summary["micro_accuracy"],
            "macro_balanced_accuracy": summary["macro_balanced_accuracy"],
            "exact_profile_accuracy": summary["exact_profile_accuracy"],
            "majority_baseline_micro_accuracy": majority["micro_accuracy"],
            "majority_baseline_balanced_accuracy": majority["macro_balanced_accuracy"],
            "majority_baseline_exact_profile_accuracy": majority["exact_profile_accuracy"],
            "unrestricted_permutation_p": k1_p if summary["k"] == 1 else np.nan,
            "origin_stratified_permutation_p": k1_stratified_p if summary["k"] == 1 else np.nan,
        }
        knn_rows.append(row)
    knn_table = pd.DataFrame(knn_rows)
    knn_table.to_csv(output_dir / "tables" / "nearest_neighbor_prediction.csv", index=False)

    per_host_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for host_index, value in zip(summary["polymorphic_host_indices"], summary["per_host_balanced_accuracy"], strict=True):
            per_host_rows.append({"k": summary["k"], "host": virulence.host_names[int(host_index)], "balanced_accuracy": float(value)})
    pd.DataFrame(per_host_rows).to_csv(output_dir / "tables" / "nearest_neighbor_per_host.csv", index=False)

    # Filter-continuum sensitivity analysis.
    filter_rows: list[dict[str, Any]] = []
    continuum_distances: dict[int, np.ndarray] = {}
    primary_full_vector = upper_triangle(marker_distances_full["reconstructed_1244"])
    pair_triangle_30 = np.triu_indices(len(virulence.isolate_ids), 1)
    same_pathotype = virulence.pathotypes[pair_triangle_30[0]] == virulence.pathotypes[pair_triangle_30[1]]
    for threshold in FILTER_THRESHOLDS:
        mask = np.isfinite(hapmap.maf) & (hapmap.maf >= 0.05) & (hapmap.calculated_missing <= threshold)
        distance, overlap = pairwise_ibs_distance(hapmap.dosage[:, mask])
        continuum_distances[threshold] = distance
        selected = distance[np.ix_(selected_indices, selected_indices)]
        marker_vector = selected[pair_triangle_30]
        virulence_vector = virulence_distance[pair_triangle_30]
        filter_rows.append(
            {
                "max_missing_calls": threshold,
                "max_missing_fraction": threshold / len(hapmap.sample_ids),
                "marker_count": int(mask.sum()),
                "mean_pair_overlap_140": float(upper_triangle(overlap).mean()),
                "min_pair_overlap_140": int(upper_triangle(overlap).min()),
                "pearson_r": float(stats.pearsonr(marker_vector, virulence_vector).statistic),
                "spearman_rho": float(stats.spearmanr(marker_vector, virulence_vector).statistic),
                "pathotype_between_minus_within": float(marker_vector[~same_pathotype].mean() - marker_vector[same_pathotype].mean()),
                "origin_permanova_F": permanova_pseudo_f(selected, selected_origins),
                "distance_pearson_vs_1244": float(stats.pearsonr(primary_full_vector, upper_triangle(distance)).statistic),
                "distance_spearman_vs_1244": float(stats.spearmanr(primary_full_vector, upper_triangle(distance)).statistic),
            }
        )
    filter_table = pd.DataFrame(filter_rows)
    filter_table.to_csv(output_dir / "tables" / "filter_sensitivity.csv", index=False)

    # Retrospective sampling-design audit of the 30 phenotyped isolates.
    primary_full = marker_distances_full["reconstructed_1244"]
    sorghum_origins = {"GA", "NC", "PR", "TX"}
    design_eligible_indices = np.flatnonzero(
        metadata["origin"].astype(str).isin(sorghum_origins).to_numpy()
    )
    global_to_design = {
        int(global_index): local_index
        for local_index, global_index in enumerate(design_eligible_indices)
    }
    design_selected_indices = np.asarray(
        [global_to_design[int(i)] for i in selected_indices], dtype=int
    )
    design_distance = primary_full[np.ix_(design_eligible_indices, design_eligible_indices)]
    design_origins = metadata.iloc[design_eligible_indices]["origin"].astype(str).to_numpy()
    design_sample_ids = [hapmap.sample_ids[int(i)] for i in design_eligible_indices]
    design_metadata = metadata.iloc[design_eligible_indices].reset_index(drop=True)

    observed, unrestricted_null, unrestricted_percentile, unrestricted_p = panel_randomization_audit(
        design_distance,
        design_selected_indices,
        reps=panel_reps,
        seed=seed + 601,
    )
    _, origin_null, origin_percentile, origin_p = panel_randomization_audit(
        design_distance,
        design_selected_indices,
        reps=panel_reps,
        seed=seed + 602,
        groups=design_origins,
    )
    np.save(output_dir / "null_distributions" / "panel_audit_unrestricted.npy", unrestricted_null)
    np.save(output_dir / "null_distributions" / "panel_audit_origin_matched.npy", origin_null)
    panel_rows = []
    for index, metric in enumerate(PANEL_METRIC_NAMES):
        panel_rows.append(
            {
                "metric": metric,
                "observed": observed[index],
                "unrestricted_percentile": unrestricted_percentile[index],
                "unrestricted_directional_p": unrestricted_p[index],
                "unrestricted_q025": float(np.quantile(unrestricted_null[:, index], 0.025)),
                "unrestricted_q50": float(np.quantile(unrestricted_null[:, index], 0.50)),
                "unrestricted_q975": float(np.quantile(unrestricted_null[:, index], 0.975)),
                "origin_matched_percentile": origin_percentile[index],
                "origin_matched_directional_p": origin_p[index],
                "origin_matched_q025": float(np.quantile(origin_null[:, index], 0.025)),
                "origin_matched_q50": float(np.quantile(origin_null[:, index], 0.50)),
                "origin_matched_q975": float(np.quantile(origin_null[:, index], 0.975)),
                "random_panels": panel_reps,
            }
        )
    panel_table = pd.DataFrame(panel_rows)
    panel_table.to_csv(output_dir / "tables" / "sampling_design_audit.csv", index=False)

    # Candidate isolates for targeted future phenotyping.
    unselected = np.setdiff1d(np.arange(len(hapmap.sample_ids)), selected_indices)
    nearest_pos = np.argmin(primary_full[np.ix_(unselected, selected_indices)], axis=1)
    nearest_distance = np.min(primary_full[np.ix_(unselected, selected_indices)], axis=1)
    candidate_table = metadata.iloc[unselected].reset_index(drop=True).copy()
    candidate_table["nearest_phenotyped_isolate"] = [virulence.isolate_ids[position] for position in nearest_pos]
    candidate_table["nearest_panel_distance"] = nearest_distance
    candidate_table["coverage_priority_rank"] = candidate_table["nearest_panel_distance"].rank(method="min", ascending=False).astype(int)
    candidate_table["replication_priority_within_origin"] = (
        candidate_table.groupby("origin")["nearest_panel_distance"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    candidate_table["eligible_for_sorghum_panel_design"] = candidate_table["origin"].isin(["GA", "NC", "PR", "TX"])
    candidate_table["origin_balanced_replication_shortlist"] = (
        candidate_table["origin"].isin(["GA", "NC", "PR", "TX"])
        & (candidate_table["replication_priority_within_origin"] <= 3)
    )
    candidate_table.to_csv(output_dir / "tables" / "unphenotyped_candidate_isolates.csv", index=False)

    # Retrospective isolate-panel optimization. This is not presented as a
    # replacement for biological replication; it quantifies marker-space
    # tradeoffs and generates an auditable augmentation order for future
    # phenotyping.
    observed_metrics = optimized_panel_metrics(design_distance, design_selected_indices)
    optimized_design_indices, optimized_metrics = optimize_farthest_first_panel(
        design_distance, design_sample_ids, len(design_selected_indices)
    )
    origin_counts = Counter(design_metadata.iloc[design_selected_indices]["origin"].astype(str))
    origin_optimized_design_indices, origin_optimized_metrics = optimize_farthest_first_panel(
        design_distance,
        design_sample_ids,
        len(design_selected_indices),
        groups=design_origins,
        quotas=dict(origin_counts),
    )
    augmented_design_indices, augmentation_order, augmentation_metrics = augment_panel_farthest_first(
        design_distance,
        design_sample_ids,
        design_selected_indices,
        final_size=min(50, len(design_sample_ids)),
    )
    augmented_50_metrics = optimized_panel_metrics(design_distance, augmented_design_indices)
    optimized_indices = [int(design_eligible_indices[i]) for i in optimized_design_indices]
    origin_optimized_indices = [int(design_eligible_indices[i]) for i in origin_optimized_design_indices]
    augmented_indices = [int(design_eligible_indices[i]) for i in augmented_design_indices]

    scenario_metrics = pd.DataFrame(
        [
            {"scenario": "Observed 30", **observed_metrics},
            {"scenario": "Coverage-optimized 30", **optimized_metrics},
            {"scenario": "Origin-quota-optimized 30", **origin_optimized_metrics},
            {"scenario": "Observed + 20", **augmented_50_metrics},
        ]
    )
    scenario_metrics.to_csv(output_dir / "tables" / "panel_scenario_metrics.csv", index=False)
    augmentation_order.to_csv(output_dir / "tables" / "panel_augmentation_order.csv", index=False)
    augmentation_metrics.to_csv(output_dir / "tables" / "panel_augmentation_curve.csv", index=False)
    panel_membership_table(
        design_sample_ids, design_metadata, optimized_design_indices, "Coverage-optimized 30"
    ).to_csv(output_dir / "tables" / "coverage_optimized_panel_30.csv", index=False)
    panel_membership_table(
        design_sample_ids, design_metadata, origin_optimized_design_indices, "Origin-quota-optimized 30"
    ).to_csv(output_dir / "tables" / "origin_quota_optimized_panel_30.csv", index=False)

    # Per-isolate nearest-panel distances for observed and optimized scenarios.
    nearest_rows: list[dict[str, Any]] = []
    for scenario, panel in [
        ("Observed 30", list(design_selected_indices)),
        ("Coverage-optimized 30", optimized_design_indices),
        ("Origin-quota-optimized 30", origin_optimized_design_indices),
        ("Observed + 20", augmented_design_indices),
    ]:
        panel_array = np.asarray(panel, dtype=int)
        remaining = np.setdiff1d(np.arange(len(design_sample_ids)), panel_array)
        nearest = design_distance[np.ix_(remaining, panel_array)].min(axis=1) if len(remaining) else np.array([])
        for isolate_index, value in zip(remaining, nearest, strict=True):
            nearest_rows.append(
                {
                    "scenario": scenario,
                    "isolate_id": design_sample_ids[int(isolate_index)],
                    "origin": design_metadata.iloc[int(isolate_index)]["origin"],
                    "nearest_panel_distance": float(value),
                }
            )
    nearest_distance_table = pd.DataFrame(nearest_rows)
    nearest_distance_table.to_csv(output_dir / "tables" / "panel_nearest_distances.csv", index=False)

    # PCoA and pairwise source tables.
    coordinates, eigenvalues = pcoa(primary_full, dimensions=4)
    positive = eigenvalues[eigenvalues > 0]
    pcoa_table = metadata.copy()
    for axis in range(coordinates.shape[1]):
        pcoa_table[f"PCoA{axis + 1}"] = coordinates[:, axis]
        pcoa_table[f"PCoA{axis + 1}_variance_fraction"] = positive[axis] / positive.sum()
    pcoa_table["phenotyped"] = pcoa_table["isolate_id"].isin(virulence.isolate_ids)
    pcoa_table.to_csv(output_dir / "tables" / "marker_pcoa_140.csv", index=False)

    all_pairs = _distance_pairs_all(hapmap.sample_ids, marker_distances_full)
    all_pairs.to_csv(output_dir / "tables" / "pairwise_marker_distances_140.csv", index=False)
    selected_pairs = _distance_pairs_selected(
        virulence.isolate_ids,
        virulence.pathotypes,
        selected_origins,
        marker_distances_selected,
        virulence_distance,
    )
    selected_pairs.to_csv(output_dir / "tables" / "marker_virulence_pairs_30.csv", index=False)
    selected_metadata.to_csv(output_dir / "tables" / "phenotyped_isolate_metadata.csv", index=False)
    metadata.to_csv(output_dir / "tables" / "isolate_metadata_140.csv", index=False)
    virulence.frame.to_csv(output_dir / "tables" / "virulence_profiles_30x18.csv", index=False)

    # Dataset audit table.
    audit_table = pd.DataFrame(
        [
            {"item": "HapMap marker rows after summary-row removal", "value": len(hapmap.markers)},
            {"item": "Genotyped isolates", "value": len(hapmap.sample_ids)},
            {"item": "Phenotyped isolates", "value": len(virulence.isolate_ids)},
            {"item": "Host differentials", "value": len(virulence.host_names)},
            {"item": "Published pathotypes", "value": int(pd.Series(virulence.pathotypes).nunique())},
            {"item": "Singleton pathotypes", "value": int((pd.Series(virulence.pathotypes).value_counts() == 1).sum())},
            {"item": "Within-pathotype isolate pairs", "value": int(selected_pairs["same_pathotype"].sum())},
            {"item": "Between-pathotype isolate pairs", "value": int((~selected_pairs["same_pathotype"]).sum())},
            {"item": "HapMap missing-count column validated", "value": True},
            {"item": "Minimum recalculated MAF", "value": float(np.nanmin(hapmap.maf))},
            {"item": "Reference assembly named by source study", "value": "TX430BB, GenBank JMSE00000000.1"},
        ]
    )
    audit_table.to_csv(output_dir / "tables" / "dataset_audit.csv", index=False)


    # Machine-readable and plain-text summaries.
    primary_corr = concordance.set_index("marker_set").loc["reconstructed_1244"]
    primary_sep = pathotype_table.set_index("marker_set").loc["reconstructed_1244"]
    primary_mgc = mgc_table.set_index("marker_set").loc["reconstructed_1244"]
    marker_origin = origin_structure_table.set_index("distance_type").loc["marker_1244"]
    vir_origin = origin_structure_table.set_index("distance_type").loc["virulence"]
    summary: dict[str, Any] = {
        "pipeline_version": "1.0.0",
        "mode": mode,
        "seed": seed,
        "inputs": {
            "hapmap": str(hapmap_path.resolve()),
            "hapmap_sha256": _sha256(hapmap_path),
            "virulence": str(virulence_path.resolve()),
            "virulence_sha256": _sha256(virulence_path),
            "validation_panel_400": str(Path(validation_panel_path).resolve()) if validation_panel_path else None,
        },
        "dataset": {
            "prefilter_markers": len(hapmap.markers),
            "genotyped_isolates": len(hapmap.sample_ids),
            "phenotyped_isolates": len(virulence.isolate_ids),
            "host_differentials": len(virulence.host_names),
            "pathotypes": int(pd.Series(virulence.pathotypes).nunique()),
        },
        "primary_marker_set": {
            "name": "reconstructed_1244",
            "marker_count": 1244,
            "filter": "MAF >= 0.05 and no more than 15 missing genotype calls",
            "archived_final_export_compared": False,
        },
        "primary_results": {
            "pearson_r": primary_corr["pearson_r"],
            "pearson_p": primary_corr["pearson_p"],
            "spearman_rho": primary_corr["spearman_rho"],
            "spearman_p": primary_corr["spearman_p"],
            "pathotype_within_mean": primary_sep["within_mean"],
            "pathotype_between_mean": primary_sep["between_mean"],
            "pathotype_difference": primary_sep["between_minus_within"],
            "pathotype_p": primary_sep["p_value"],
            "mgc_statistic": primary_mgc["statistic"],
            "mgc_p": primary_mgc["p_value"],
            "marker_origin_permanova_F": marker_origin["permanova_F"],
            "marker_origin_permanova_R2": marker_origin["permanova_R2"],
            "marker_origin_permanova_p": marker_origin["permanova_p"],
            "marker_origin_permdisp_F": marker_origin["permdisp_F"],
            "marker_origin_permdisp_p": marker_origin["permdisp_p"],
            "marker_origin_between_minus_within": marker_origin["between_minus_within"],
            "marker_origin_distance_contrast_p": marker_origin["distance_contrast_p"],
            "virulence_origin_permanova_F": vir_origin["permanova_F"],
            "virulence_origin_permanova_R2": vir_origin["permanova_R2"],
            "virulence_origin_permanova_p": vir_origin["permanova_p"],
            "virulence_origin_permdisp_F": vir_origin["permdisp_F"],
            "virulence_origin_permdisp_p": vir_origin["permdisp_p"],
            "virulence_origin_between_minus_within": vir_origin["between_minus_within"],
            "virulence_origin_distance_contrast_p": vir_origin["distance_contrast_p"],
            "knn_k1_balanced_accuracy": k1_observed,
            "knn_k1_unrestricted_p": k1_p,
            "knn_k1_origin_stratified_p": k1_stratified_p,
        },
        "validation_panel_400_summary": validation_summary,
        "sampling_audit": panel_table.to_dict(orient="records"),
        "panel_optimization": {
            "scenario_metrics": scenario_metrics.to_dict(orient="records"),
            "first_20_augmentation_isolates": augmentation_order["isolate_id"].head(20).tolist(),
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    text = f"""C. sublineola SNP-virulence analysis summary
===========================================

Input marker table: {len(hapmap.markers):,} coordinate-resolved HapMap markers across {len(hapmap.sample_ids)} isolates
Primary marker set: reconstructed 1,244-marker panel
Sensitivity sets: 1,135 markers (literal <10% missing) and 400 markers (low-missingness subset)
Virulence panel: {len(virulence.isolate_ids)} isolates × {len(virulence.host_names)} differentials; {int(pd.Series(virulence.pathotypes).nunique())} pathotypes

Primary global concordance
- Pearson r = {primary_corr['pearson_r']:.4f}; permutation P = {primary_corr['pearson_p']:.4f}
- Spearman rho = {primary_corr['spearman_rho']:.4f}; permutation P = {primary_corr['spearman_p']:.4f}
- MGC statistic = {primary_mgc['statistic']:.4f}; permutation P = {primary_mgc['p_value']:.4f}

Pathotype separation
- Within-pathotype mean IBS distance = {primary_sep['within_mean']:.4f} (n = {int(primary_sep['within_pairs'])} pairs)
- Between-pathotype mean IBS distance = {primary_sep['between_mean']:.4f} (n = {int(primary_sep['between_pairs'])} pairs)
- Difference = {primary_sep['between_minus_within']:.4f}; permutation P = {primary_sep['p_value']:.4f}

Geographic structure
- Marker distance: PERMANOVA pseudo-F = {marker_origin['permanova_F']:.4f}, R2 = {marker_origin['permanova_R2']:.4f}, P = {marker_origin['permanova_p']:.5f}; PERMDISP P = {marker_origin['permdisp_p']:.5f}
- Marker between-origin minus within-origin distance = {marker_origin['between_minus_within']:.4f}; permutation P = {marker_origin['distance_contrast_p']:.5f}
- Virulence distance: PERMANOVA pseudo-F = {vir_origin['permanova_F']:.4f}, R2 = {vir_origin['permanova_R2']:.4f}, P = {vir_origin['permanova_p']:.4f}; PERMDISP P = {vir_origin['permdisp_p']:.4f}
- Virulence between-origin minus within-origin distance = {vir_origin['between_minus_within']:.4f}; permutation P = {vir_origin['distance_contrast_p']:.4f}

Nearest-neighbor prediction
- k=1 macro balanced accuracy = {k1_observed:.4f}
- Unrestricted permutation P = {k1_p:.4f}
- Origin-stratified permutation P = {k1_stratified_p:.4f}

Retrospective panel design
- Observed 30-isolate panel: maximum nearest-panel gap = {observed_metrics['max_unselected_nearest_distance']:.4f}
- Coverage-optimized 30-isolate panel: maximum gap = {optimized_metrics['max_unselected_nearest_distance']:.4f}
- Observed panel plus 20 farthest-first additions: maximum gap = {augmented_50_metrics['max_unselected_nearest_distance']:.4f}

Marker reconstruction
- Filter: MAF >= 0.05 and no more than 15 missing genotype calls
- Markers retained: 1,244
- Archived final 1,244-marker export comparison: not performed
"""
    (output_dir / "RUN_SUMMARY.txt").write_text(text, encoding="utf-8")

    shutil.rmtree(output_dir / "work", ignore_errors=True)
    LOGGER.info("Pipeline completed successfully")
    for handler in logging.getLogger().handlers:
        handler.flush()
    logging.shutdown()

    manifest_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "RESULT_MANIFEST.csv":
            manifest_rows.append({
                "relative_path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(manifest_rows).to_csv(output_dir / "RESULT_MANIFEST.csv", index=False)
    return summary
