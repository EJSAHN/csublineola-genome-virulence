# Data sources

## Virulence matrix

`Prom_Anthracnose_Diversity_Data_2024.csv` contains the 30-isolate by 18-host resistant/susceptible matrix and pathotype assignments associated with Prom et al. (2024).

- Article DOI: `10.3390/jof10010003`
- USDA Ag Data Commons DOI: `10.15482/USDA.ADC/25988122.v1`
- License: CC BY 4.0

## Marker genotypes

The analysis requires an author-held HapMap genotype table for 140 isolates. The public repository does not redistribute genotype calls.

Recommended publication-facing filenames are:

- `C_sublineola_140_isolates_7398_markers.hmp.txt` for the source HapMap table
- `C_sublineola_140_isolates_400_low_missing_markers.hmp.txt` for the optional validation panel

The optional validation panel is used only to confirm exact recovery of the 400-marker low-missingness subset from the source HapMap table.
