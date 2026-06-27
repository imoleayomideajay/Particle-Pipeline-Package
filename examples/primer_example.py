"""
examples/primer_example.py
==========================
Demonstrates the three-line API shown in the JoM Primers paper.
Runs on a JEOL SEM image of 480 nm PSL particles.

Usage
-----
    python examples/primer_example.py path/to/PS-L-RE-003-480nm.tif

Or with a synthetic image (no real SEM needed):
    python examples/primer_example.py --synthetic
"""

import argparse
import sys
from pathlib import Path


def run_on_real_image(image_path: str):
    """The exact three-line example from the primer."""
    from particle_pipeline import run_pipeline

    # ── Three lines as shown in the primer ────────────────────────────
    results = run_pipeline(
        image_path   = image_path,
        nominal_d_um = 0.48,        # known nominal diameter (optional)
        px_per_um    = None,        # auto-extracted from JEOL metadata
    )
    results.save_report("output/")  # PDF report + CSV + figures
    # ──────────────────────────────────────────────────────────────────

    print(results.summary())
    print(f"\nOutputs written to: output/")


def run_on_synthetic():
    """Demonstrate pipeline on a fully synthetic image (no SEM needed)."""
    import numpy as np
    from PIL import Image as PILImage
    import tempfile, os
    from skimage.draw import disk
    from particle_pipeline import run_pipeline

    print("Generating synthetic 480 nm PSL-like image...")
    rng = np.random.default_rng(42)
    img = np.ones((512, 512), dtype=np.float64) * 0.15
    # Place ~50 bright circles, diameter ~102 px (480 nm at 213 px/µm)
    placed = []
    for _ in range(200):
        r = int(rng.normal(51, 5))
        r = max(20, min(r, 70))
        cx = rng.integers(r+5, 512-r-5)
        cy = rng.integers(r+5, 512-r-5)
        if any(np.sqrt((cx-px)**2+(cy-py)**2) < (r+pr)*0.9
               for px, py, pr in placed):
            continue
        rr, cc = disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = rng.uniform(0.55, 0.80)
        placed.append((cx, cy, r))
        if len(placed) >= 50:
            break

    img += rng.normal(0, 0.04, img.shape)
    img = np.clip(img, 0, 1)

    # Save to temp file with a companion .txt calibration file
    tmp = tempfile.mkdtemp()
    img_path = Path(tmp) / "synthetic_psl.tif"
    PILImage.fromarray((img * 255).astype(np.uint8)).save(str(img_path))

    # Write JEOL-style calibration metadata
    txt_path = img_path.with_suffix(".txt")
    txt_path.write_text(
        "$$SM_MICRON_BAR 213\n"
        "$$SM_MICRON_MARKER 1um\n"
    )

    print(f"Synthetic image: {img_path}")
    print(f"Placed {len(placed)} particles")
    print()

    results = run_pipeline(
        image_path   = img_path,
        nominal_d_um = 0.48,
    )
    results.save_report("output_synthetic/")
    print(results.summary())
    print("\nOutputs written to: output_synthetic/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to SEM TIFF image")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on a generated synthetic image")
    args = parser.parse_args()

    if args.synthetic or args.image is None:
        run_on_synthetic()
    else:
        run_on_real_image(args.image)
