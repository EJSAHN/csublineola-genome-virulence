# Example commands

```bash
python -m csublineola_marker_virulence.cli \
  --alignment data/marker_alignment.fas \
  --virulence-table data/virulence_matrix.csv \
  --permutations 5000 \
  --random-seed 1729 \
  --output outputs/marker_virulence_results.xlsx
```

Use `python -m csublineola_marker_virulence.cli --help` to list all options.
