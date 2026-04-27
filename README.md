# Colletotrichum sublineola genome–virulence analysis pipeline

This repository contains a reproducible Python pipeline for comparing genome-wide relatedness, virulence-profile distance, and host-specific genomic signal in *Colletotrichum sublineola*.

The workflow reads an aligned FASTA file and a binary host differential response matrix, calculates pairwise genomic and virulence distances, performs permutation-based matrix and host-specific analyses, screens candidate host-associated alignment sites, and exports traceable Excel workbooks for downstream manuscript preparation.

## Repository scope

This repository contains only the *C. sublineola* analysis pipeline. Figure-generation code, manuscript drafting files, generated workbooks, and unrelated pathogen analyses are intentionally excluded.

## Required inputs

The command-line workflow expects:

1. An aligned FASTA file containing *C. sublineola* isolate sequences.
2. A CSV or Excel virulence table with an `isolate_id` column and host differential columns encoded as `R` or `S`.

Optional isolate metadata can be supplied with a separate table keyed by `isolate_id`.

## Environment setup

Using conda:

```bash
conda env create -f environment.yml
conda activate csublineola-genome-virulence
```

Using pip inside an existing Python environment:

```bash
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

## Example command

Linux/macOS:

```bash
python -m colletotrichum_sublineola_pipeline.run \
  --alignment path/to/csub_alignment.fas \
  --virulence-table path/to/csublineola_RS_table_30x18.csv \
  --output outputs/colletotrichum_sublineola_results.xlsx
```

Windows Anaconda Prompt:

```bash
python -m colletotrichum_sublineola_pipeline.run --alignment "path\to\csub_alignment.fas" --virulence-table "path\to\csublineola_RS_table_30x18.csv" --output "outputs\colletotrichum_sublineola_results.xlsx"
```

To view all command-line options:

```bash
python -m colletotrichum_sublineola_pipeline.run --help
```

## Main outputs

The workflow writes one Excel workbook containing:

- sequence and isolate summaries
- pairwise genetic distance matrices and pair tables
- virulence profile summaries
- pairwise virulence distance matrices and pair tables
- distance relationship summaries
- pathotype and profile separation tests
- host-specific separation tests
- site-by-host association results
- nearest-neighbor recovery summaries
- compact analysis summary tables

## Analysis rules

- Pairwise genetic distance is calculated by pairwise deletion of non-canonical characters.
- Canonical DNA bases are `A`, `C`, `G`, and `T`.
- Gaps and ambiguous characters are excluded from pairwise genetic distance calculations.
- Virulence distances are calculated from observed `R` and `S` values only.
- Permutation-based analyses use a fixed base random seed by default and deterministic offsets across analysis modules.
- No plotting code is included.
- Final outputs are exported as Excel workbooks.

## Data availability

Input data are not bundled with this repository unless permitted by the relevant data source. The associated manuscript and supplementary data file should be cited for the complete analytical output workbook.

## License

This code is released under the MIT License. See `LICENSE` for details.
