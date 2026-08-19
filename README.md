# csublineola-genome-virulence

Reproducible analysis of RAD-seq marker relatedness, geographic structure, virulence-profile concordance, and isolate-panel design in *Colletotrichum sublineola*.

## Analyses

- validation and filtering of a 7,398-marker HapMap table
- reconstruction of 1,244-, 1,135-, and 400-marker panels
- direct allele-sharing (IBS) distances
- Pearson and Spearman matrix-concordance tests
- pathotype-separation permutation tests
- multiscale graph correlation
- PERMANOVA, PERMDISP, and within-origin versus between-origin contrasts
- nearest-neighbor prediction with unrestricted and origin-stratified null models
- isolate-panel coverage and augmentation analyses

## Installation

```bash
conda env create -f environment.yml
conda run -n csub_snp_virulence python -m unittest discover -s tests -v
```

Alternatively:

```bash
python -m pip install .
```

## Usage

```bash
csub-snp-virulence \
  --hapmap /path/to/C_sublineola_140_isolates_7398_markers.hmp.txt \
  --output /path/to/results \
  --mode full \
  --seed 1729
```

An optional 400-marker validation panel can be supplied in HapMap format:

```bash
csub-snp-virulence \
  --hapmap /path/to/C_sublineola_140_isolates_7398_markers.hmp.txt \
  --validation-panel /path/to/C_sublineola_140_isolates_400_low_missing_markers.hmp.txt \
  --output /path/to/results \
  --mode full \
  --seed 1729
```

`--mode full` uses 20,000 permutations; `--mode fast` uses 1,000.

## Inputs

The public 30-isolate by 18-host virulence matrix and the 140-isolate metadata table are included under `data/`. Genotype files are not bundled with this repository.

The primary genotype input is a HapMap table containing marker identifiers, scaffold positions, alleles, missing-call counts, and isolate genotype calls. The optional validation panel is used only to verify exact recovery of the 400-marker low-missingness subset.

## Outputs

The pipeline writes CSV tables, distance matrices, filtered HapMap files, permutation distributions, software versions, execution logs, and SHA-256 checksums. Plotting code is not included.

## Data source

Prom LK, Ahn EJS, Perumal R, Cuevas HE, Rooney WL, Isakeit TS, Magill CW. 2024. Genetic Diversity and Classification of *Colletotrichum sublineola* Pathotypes Using a Standard Set of Sorghum Differentials. *Journal of Fungi* 10:3. DOI: `10.3390/jof10010003`.

USDA Ag Data Commons: `10.15482/USDA.ADC/25988122.v1`.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

MIT License. See `LICENSE`.
