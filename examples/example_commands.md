# Example commands

## Basic run

```bash
python -m colletotrichum_sublineola_pipeline.run \
  --alignment data/csub_alignment.fas \
  --virulence-table data/csublineola_RS_table_30x18.csv \
  --output outputs/colletotrichum_sublineola_results.xlsx
```

## Run with optional metadata

```bash
python -m colletotrichum_sublineola_pipeline.run \
  --alignment data/csub_alignment.fas \
  --virulence-table data/csublineola_RS_table_30x18.csv \
  --metadata-table data/csub_isolate_metadata.csv \
  --output outputs/colletotrichum_sublineola_results.xlsx
```

## Explicit permutation settings

```bash
python -m colletotrichum_sublineola_pipeline.run \
  --alignment data/csub_alignment.fas \
  --virulence-table data/csublineola_RS_table_30x18.csv \
  --permutations 5000 \
  --random-seed 1729 \
  --output outputs/colletotrichum_sublineola_results.xlsx
```
