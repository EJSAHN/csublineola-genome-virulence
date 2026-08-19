from __future__ import annotations

import numpy as np


def pairwise_ibs_distance(dosage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Allele-sharing distance: mean |dosage_i-dosage_j|/2 over comparable markers."""
    dosage = np.asarray(dosage, dtype=np.float32)
    n_samples = dosage.shape[0]
    distance = np.zeros((n_samples, n_samples), dtype=np.float64)
    overlap = np.zeros((n_samples, n_samples), dtype=np.int32)
    for i in range(n_samples):
        xi = dosage[i]
        for j in range(i + 1, n_samples):
            xj = dosage[j]
            valid = np.isfinite(xi) & np.isfinite(xj)
            n_valid = int(valid.sum())
            if n_valid == 0:
                raise ValueError(f"No comparable markers for samples {i} and {j}")
            value = float(np.mean(np.abs(xi[valid] - xj[valid]) / 2.0))
            distance[i, j] = distance[j, i] = value
            overlap[i, j] = overlap[j, i] = n_valid
    np.fill_diagonal(overlap, dosage.shape[1])
    return distance, overlap


def virulence_hamming_distance(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response)
    if response.ndim != 2:
        raise ValueError("Virulence response must be a 2D isolate-by-host matrix.")
    return np.mean(response[:, None, :] != response[None, :, :], axis=2, dtype=np.float64)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[np.triu_indices_from(matrix, k=1)]


def pcoa(distance: np.ndarray, dimensions: int = 4) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-12
    kept_values = eigenvalues[positive][:dimensions]
    coordinates = eigenvectors[:, positive][:, :dimensions] * np.sqrt(kept_values)
    return coordinates, eigenvalues
