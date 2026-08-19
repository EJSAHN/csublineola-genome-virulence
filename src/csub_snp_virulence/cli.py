from __future__ import annotations

import argparse
from pathlib import Path

from .io import resolve_input_path
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct HapMap marker-virulence analysis for C. sublineola"
    )
    parser.add_argument("--hapmap", required=True, help="Path to the 7,398-marker HapMap TXT file")
    parser.add_argument("--virulence", default=None, help="Optional path to the 30 x 18 R/S CSV")
    parser.add_argument("--metadata", default=None, help="Optional path to the 140-isolate metadata CSV")
    parser.add_argument(
        "--validation-panel",
        default=None,
        help="Optional 400-marker validation panel in HapMap format",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["full", "fast"], default="full")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    hapmap = resolve_input_path(args.hapmap, "HapMap input")
    validation_panel = (
        resolve_input_path(args.validation_panel, "400-marker validation panel")
        if args.validation_panel
        else None
    )
    virulence = (
        resolve_input_path(args.virulence, "Virulence matrix")
        if args.virulence
        else package_root / "data" / "Prom_Anthracnose_Diversity_Data_2024.csv"
    )
    metadata = (
        resolve_input_path(args.metadata, "Isolate metadata")
        if args.metadata
        else package_root / "data" / "isolate_metadata_140.csv"
    )
    output = Path(args.output).expanduser().resolve()

    run_pipeline(
        hapmap_path=hapmap,
        virulence_path=virulence,
        metadata_path=metadata,
        output_dir=output,
        validation_panel_path=validation_panel,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
