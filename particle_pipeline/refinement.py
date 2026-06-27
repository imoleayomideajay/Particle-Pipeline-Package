"""
particle_pipeline.refinement
=============================
Chan-Vese morphological active contour for sub-pixel boundary refinement.

For each particle ROI, the method minimises:

    E(Γ) = λ₁∫_in(u − c₁)² dx + λ₂∫_out(u − c₂)² dx + μ|Γ|

where u is the image, c₁/c₂ are mean intensities inside/outside the
contour, and |Γ| is the contour perimeter (smoothing term).

The initialisation contour is the dilated binary mask from the global
threshold step. The refined mask area gives a sub-pixel equivalent
diameter. Returns None for ROIs that are too small or fail to converge.

Reference
---------
Getreuer P. (2012). Chan-Vese segmentation. Image Processing On Line,
2, 214–224. https://doi.org/10.5201/ipol.2012.g-cv
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.segmentation import morphological_chan_vese
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties

warnings.filterwarnings("ignore", category=FutureWarning)


def refine_particle(
    image: np.ndarray,
    prop: RegionProperties,
    px_per_um: float,
    pad: int = 8,
    n_iter: int = 50,
    smoothing: int = 2,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> float | None:
    """
    Apply Chan-Vese active contour to one particle ROI.

    Parameters
    ----------
    image : np.ndarray
        Full normalised SEM image.
    prop : RegionProperties
        scikit-image region property for this particle.
    px_per_um : float
        Calibration in pixels per micrometre.
    pad : int
        Padding around the bounding box (pixels).
    n_iter : int
        Maximum Chan-Vese iterations.
    smoothing : int
        Smoothing parameter (µ): higher = smoother boundary.
    lambda1, lambda2 : float
        Weights for inside/outside intensity homogeneity terms.

    Returns
    -------
    float or None
        Refined equivalent diameter in nm, or None if refinement
        failed or the ROI is too small.
    """
    minr, minc, maxr, maxc = prop.bbox
    r0 = max(0, minr - pad)
    r1 = min(image.shape[0], maxr + pad)
    c0 = max(0, minc - pad)
    c1 = min(image.shape[1], maxc + pad)
    roi = image[r0:r1, c0:c1]

    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None

    # Build initial mask from particle coordinates
    init_mask = np.zeros(roi.shape, dtype=bool)
    for coord in prop.coords:
        rr = coord[0] - r0
        cc = coord[1] - c0
        if 0 <= rr < roi.shape[0] and 0 <= cc < roi.shape[1]:
            init_mask[rr, cc] = True
    init_mask = morphology.dilation(init_mask, morphology.disk(1))

    try:
        refined = morphological_chan_vese(
            roi,
            num_iter=n_iter,
            init_level_set=init_mask.astype(float),
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        area = refined.sum()
        if area < 4:
            return None
        d_refined_px = 2.0 * np.sqrt(area / np.pi)
        return float(d_refined_px / px_per_um * 1000)  # nm
    except Exception:
        return None


def refine_all(
    image: np.ndarray,
    props: list[RegionProperties],
    df: pd.DataFrame,
    px_per_um: float,
    pad: int = 8,
    n_iter: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply Chan-Vese refinement to all particles in a DataFrame.

    Parameters
    ----------
    image : np.ndarray
        Full normalised SEM image.
    props : list of RegionProperties
        Must correspond 1-to-1 with rows in ``df``.
    df : pd.DataFrame
        Morphometry DataFrame (output of morphometry.extract_morphometry).
    px_per_um : float
    pad, n_iter : int
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three new columns:
        refined_d_nm, boundary_shift_nm, refinement_converged.
    """
    refined_nm   = []
    shifts_nm    = []
    converged    = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i < len(props):
            r_nm = refine_particle(image, props[i], px_per_um, pad, n_iter)
        else:
            r_nm = None

        orig_nm = row["diam_nm"]
        if r_nm is not None:
            refined_nm.append(round(r_nm, 2))
            shifts_nm.append(round(r_nm - orig_nm, 2))
            converged.append(True)
        else:
            refined_nm.append(None)
            shifts_nm.append(None)
            converged.append(False)

    df = df.copy()
    df["refined_d_nm"]       = refined_nm
    df["boundary_shift_nm"]  = shifts_nm
    df["refinement_converged"] = converged

    if verbose:
        n_ok = sum(converged)
        ok_df = df[df["refinement_converged"]]
        print(f"  Chan-Vese: {n_ok}/{len(df)} converged")
        if len(ok_df):
            print(f"  Mean shift: {ok_df['boundary_shift_nm'].mean():+.2f} nm")
            print(f"  Refined D50: {ok_df['refined_d_nm'].median():.1f} nm")

    return df
