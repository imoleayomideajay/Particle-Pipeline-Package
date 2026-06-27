"""
particle_pipeline.cli
======================
Command-line interface.

Usage
-----
    particle-pipeline path/to/image.tif --nominal 0.48 --output results/
    particle-pipeline path/to/image.tif --px-per-um 213.0 --no-refinement
    particle-pipeline --help
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="particle-pipeline",
        description=(
            "Automated SEM particle size analysis with physics-based "
            "DLS/RPS estimation and GUM uncertainty quantification."
        ),
    )
    parser.add_argument(
        "image",
        help="Path to SEM TIFF image."
    )
    parser.add_argument(
        "--nominal", "-n",
        type=float, default=None, metavar="µm",
        help="Nominal particle diameter in µm (optional)."
    )
    parser.add_argument(
        "--px-per-um", "-c",
        type=float, default=None, metavar="px/µm",
        help="Manual calibration override (pixels per micrometre)."
    )
    parser.add_argument(
        "--output", "-o",
        default="particle_pipeline_output", metavar="DIR",
        help="Output directory (default: particle_pipeline_output)."
    )
    parser.add_argument(
        "--prefix", "-p",
        default=None,
        help="Output filename prefix (default: image stem)."
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Skip Chan-Vese sub-pixel boundary refinement."
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Skip GUM uncertainty budget computation."
    )
    parser.add_argument(
        "--beam-shrinkage",
        type=float, default=1.0, metavar="%",
        help="Beam-induced shrinkage correction (%%). Default 1.0. "
             "Set to 0 for beam-stable materials."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output."
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Print version and exit."
    )

    args = parser.parse_args()

    if args.version:
        from particle_pipeline._version import __version__
        print(f"particle-pipeline {__version__}")
        sys.exit(0)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or image_path.stem

    from particle_pipeline import run_pipeline

    results = run_pipeline(
        image_path=image_path,
        nominal_d_um=args.nominal,
        px_per_um=args.px_per_um,
        run_refinement=not args.no_refinement,
        run_uncertainty=not args.no_uncertainty,
        beam_shrinkage_pct=args.beam_shrinkage,
        verbose=not args.quiet,
    )

    print("\n" + results.summary())

    saved = results.save_report(args.output, prefix=prefix)
    print(f"\nOutputs saved to: {args.output}/")
    for k, p in saved.items():
        print(f"  {k:12s}: {p.name}")


if __name__ == "__main__":
    main()
