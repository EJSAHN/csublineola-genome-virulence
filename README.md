# C. sublineola marker–virulence concordance pipeline

This repository contains the reproducible Python workflow used to compare a processed RAD-seq-derived marker alignment with differential-host virulence phenotypes in a defined *Colletotrichum sublineola* isolate panel.

The workflow calculates pairwise marker distances, pairwise virulence distances, permutation-based matrix correlations, and within- versus between-pathotype marker-distance summaries. It does **not** treat alignment columns as validated genomic coordinates and does not perform marker-discovery or causal-association analyses.

## Required inputs

1. An equal-length FASTA alignment containing isolate marker profiles.
2. A CSV, TSV, or Excel table containing an isolate identifier, an optional pathotype label, and host-differential responses encoded as `R` or `S`.

Input data are not bundled with this repository because redistribution requires authorization from the original data custodians.

## Environment

```bash
conda env create -f environment.yml
conda activate csublineola-marker-virulence
```

Alternatively:

```bash
pip install -e .
```

## Example

```bash
python -m csublineola_marker_virulence.cli \
  --alignment path/to/marker_alignment.fas \
  --virulence-table path/to/virulence_matrix.csv \
  --output outputs/csublineola_marker_virulence_results.xlsx
```

Windows Anaconda Prompt:

```bash
python -m csublineola_marker_virulence.cli --alignment "path\to\marker_alignment.fas" --virulence-table "path\to\virulence_matrix.csv" --output "outputs\csublineola_marker_virulence_results.xlsx"
```

## Output workbook

The output workbook contains:

- dataset, isolate, and marker-position summaries
- published R/S virulence profiles and pathotype assignments
- pairwise marker and virulence distances
- merged marker–virulence pair data
- Pearson and Spearman matrix-correlation permutation tests
- a permutation comparison of within- and between-pathotype marker distances

No plotting or manuscript-writing code is included.

## Reproducibility

Permutation counts and the random seed are command-line options. The default analysis uses 5,000 permutations and a fixed seed.

## License

MIT License. See `LICENSE`.
