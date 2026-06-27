"""
particle_pipeline.io
====================
Load and preprocess SEM images; extract pixel-per-micron calibration
from instrument metadata headers.

Supported instruments
---------------------
JEOL        $$SM_MICRON_BAR  <n_px> in companion .txt file
Zeiss       PixelSizeX in TIFF metadata
FEI/Thermo  PixelWidth in SFEG TIFF tag block
Generic     Manual px_per_um parameter
"""

from __future__ import annotations

import re
import struct
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ── Public constants ──────────────────────────────────────────────────────────

JEOL_STRIP_INTENSITY_THRESHOLD = 0.02   # rows below this mean are data strip
JEOL_STRIP_MIN_HEIGHT          = 30     # minimum data strip height to trigger


# ── Calibration extraction ────────────────────────────────────────────────────

def extract_calibration(
    image_path: str | Path,
    px_per_um: Optional[float] = None,
) -> float:
    """
    Return pixels-per-micrometre calibration for an SEM image.

    Priority order:
      1. User-supplied ``px_per_um`` (always wins if provided)
      2. Companion .txt file (JEOL  $$SM_MICRON_BAR)
      3. TIFF metadata tag (Zeiss PixelSizeX, FEI PixelWidth)
      4. Raises CalibrationError if nothing found

    Parameters
    ----------
    image_path : str or Path
        Path to the SEM TIFF image.
    px_per_um : float, optional
        Manual override.  If supplied, all metadata parsing is skipped.

    Returns
    -------
    float
        Pixels per micrometre.

    Raises
    ------
    CalibrationError
        When calibration cannot be determined automatically and
        ``px_per_um`` was not supplied.

    Examples
    --------
    >>> cal = extract_calibration("image.tif")
    >>> cal = extract_calibration("image.tif", px_per_um=213.0)
    """
    if px_per_um is not None:
        return float(px_per_um)

    image_path = Path(image_path)

    # 1. JEOL companion .txt
    cal = _jeol_calibration(image_path)
    if cal is not None:
        return cal

    # 2. TIFF embedded metadata
    cal = _tiff_calibration(image_path)
    if cal is not None:
        return cal

    raise CalibrationError(
        f"Could not extract calibration from '{image_path.name}'. "
        "Please supply px_per_um manually."
    )


def _jeol_calibration(image_path: Path) -> Optional[float]:
    """Parse JEOL $$SM_MICRON_BAR from companion .txt."""
    txt_path = image_path.with_suffix(".txt")
    if not txt_path.exists():
        return None
    text = txt_path.read_text(errors="replace")

    # $$SM_MICRON_BAR <n_px>  and  $$SM_MICRON_MARKER <label>
    bar_match    = re.search(r"\$\$SM_MICRON_BAR\s+(\d+)", text)
    marker_match = re.search(r"\$\$SM_MICRON_MARKER\s+(\S+)", text)
    if not bar_match:
        return None

    bar_px = int(bar_match.group(1))
    # Parse scale from marker string e.g. "1um", "500nm", "2.5um"
    if marker_match:
        marker = marker_match.group(1).lower()
        num = re.search(r"[\d.]+", marker)
        if num:
            val = float(num.group())
            if "nm" in marker:
                scale_um = val / 1000.0
            else:
                scale_um = val          # assume µm
            return bar_px / scale_um

    # Fallback: assume bar = 1 µm
    return float(bar_px)


def _tiff_calibration(image_path: Path) -> Optional[float]:
    """
    Extract pixel size from TIFF metadata.
    Handles Zeiss (XResolution tag) and FEI/Thermo (SFEG block).
    """
    try:
        with Image.open(image_path) as img:
            tag_data = img.tag_v2 if hasattr(img, "tag_v2") else {}

            # Zeiss: XResolution tag (282) in pixels per unit
            # ResolutionUnit tag (296): 1=no unit, 2=inch, 3=cm
            if 282 in tag_data:
                xres = tag_data[282]
                if isinstance(xres, tuple):
                    xres = xres[0] / xres[1] if xres[1] else xres[0]
                unit = tag_data.get(296, 2)
                if unit == 3:        # pixels per cm → per µm
                    return float(xres) / 1e4
                elif unit == 2:      # pixels per inch → per µm
                    return float(xres) / 25400.0

            # FEI/Thermo: PixelWidth in the SFEG metadata block (tag 34682)
            if 34682 in tag_data:
                sfeg_text = tag_data[34682]
                if isinstance(sfeg_text, bytes):
                    sfeg_text = sfeg_text.decode("latin-1", errors="replace")
                pw_match = re.search(
                    r"<PixelWidth[^>]*>\s*([\d.eE+\-]+)", sfeg_text
                )
                if pw_match:
                    pw_m = float(pw_match.group(1))   # metres per pixel
                    return 1.0 / (pw_m * 1e6)         # px per µm
    except Exception:
        pass
    return None


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(
    image_path: str | Path,
    px_per_um: Optional[float] = None,
    remove_data_strip: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Load an SEM image as a normalised float64 array.

    Steps:
      - Open TIFF, convert to 8-bit greyscale
      - Detect and remove JEOL data strip (bottom rows)
      - Normalise intensities to [0, 1]
      - Extract calibration

    Parameters
    ----------
    image_path : str or Path
    px_per_um : float, optional
        Manual calibration override.
    remove_data_strip : bool
        If True, automatically detect and remove the instrument
        data strip at the bottom of the image.

    Returns
    -------
    arr : np.ndarray
        Normalised greyscale image, shape (H, W), dtype float64.
    cal : float
        Calibration in pixels per micrometre.

    Examples
    --------
    >>> arr, cal = load_image("particles.tif")
    >>> arr, cal = load_image("particles.tif", px_per_um=213.0)
    """
    image_path = Path(image_path)
    cal = extract_calibration(image_path, px_per_um=px_per_um)

    pil_img = Image.open(image_path).convert("L")
    arr     = np.array(pil_img, dtype=np.float64)

    if remove_data_strip:
        arr = _remove_data_strip(arr)

    # Normalise to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr, cal


def _remove_data_strip(arr: np.ndarray) -> np.ndarray:
    """
    Detect and remove the JEOL black data strip at the bottom of the image.
    Scans from the bottom; the strip starts at the first row whose mean
    intensity drops below JEOL_STRIP_INTENSITY_THRESHOLD.
    Only removes the strip if it is at least JEOL_STRIP_MIN_HEIGHT rows tall.
    """
    # Normalise temporarily to detect the strip
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo + 1e-9)
    row_means = norm.mean(axis=1)

    cutoff = arr.shape[0]   # default: no strip detected
    for i in range(arr.shape[0] - 1, -1, -1):
        if row_means[i] < JEOL_STRIP_INTENSITY_THRESHOLD:
            cutoff = i
        else:
            break

    strip_height = arr.shape[0] - cutoff
    if strip_height >= JEOL_STRIP_MIN_HEIGHT:
        return arr[:cutoff, :]
    return arr


# ── Exceptions ────────────────────────────────────────────────────────────────

class CalibrationError(RuntimeError):
    """Raised when pixel calibration cannot be determined."""
