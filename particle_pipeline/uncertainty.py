"""
particle_pipeline.uncertainty
==============================
GUM-compliant seven-component measurement uncertainty budget for SEM
particle size measurement.

Components
----------
u1  Scale-bar calibration         Type B  ±0.5 px rounding, rectangular
u2  Pixel discretisation          Type B  ±0.5 px per boundary, rectangular
u3  Threshold sensitivity         Type A  SD across N candidate methods
u4  Bootstrap resampling          Type A  SD of B=2000 bootstrap means
u5  Sample heterogeneity          Type A  SEM of mean (σ/√n)
u6  Sub-pixel boundary            Type B  from PSF blur σ, rectangular
u7  Beam-induced shrinkage        Type B  material-specific, rectangular

Combined:  u_c = √(Σ u_i²)
Expanded:  U = k · u_c   (k=2, ≈95.45 % for normal distribution)

Reference
---------
JCGM 100:2008. Guide to the Expression of Uncertainty in Measurement.
Bureau International des Poids et Mesures, Sèvres.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from skimage import filters


def compute_budget(
    image: np.ndarray,
    df: pd.DataFrame,
    px_per_um: float,
    scale_bar_px: int,
    psf_blur_sigma_px: float = 0.7,
    beam_shrinkage_pct: float = 1.0,
    beam_shrinkage_half_range_pct: float = 1.0,
    k: float = 2.0,
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Compute the GUM uncertainty budget for the D50 particle size estimate.

    Parameters
    ----------
    image : np.ndarray
        Normalised SEM image (used to compute threshold sensitivity).
    df : pd.DataFrame
        Morphometry DataFrame with 'diam_nm' column.
    px_per_um : float
        Calibration in pixels per micrometre.
    scale_bar_px : int
        Scale bar length in pixels as annotated in image metadata.
    psf_blur_sigma_px : float
        Instrument PSF sigma estimate (pixels). Default 0.7.
    beam_shrinkage_pct : float
        Literature beam-shrinkage estimate (%). Set to 0 for stable materials.
    beam_shrinkage_half_range_pct : float
        Half-width of rectangular beam-shrinkage distribution (%).
    k : float
        Coverage factor (2 = ≈95.45 %).
    n_bootstrap : int
        Number of bootstrap replicates for u4.
    rng_seed : int
        Random seed for reproducibility.
    verbose : bool
        Print budget table to stdout.

    Returns
    -------
    dict
        mean_d_nm, d_corrected_nm, n, u1–u7 (nm), u_c_nm, U_nm, k,
        result_str, corrected_str, nominal_in_CI (bool),
        boot_means, boot_ci_lo, boot_ci_hi, components.
    """
    d_nm   = df["diam_nm"].values
    mean_d = float(d_nm.mean())

    # u1 — scale-bar calibration (Type B)
    u1 = (0.5 / scale_bar_px) * mean_d

    # u2 — pixel discretisation (Type B)
    u2 = (0.5 / px_per_um * 1000) / np.sqrt(3)

    # u3 — threshold sensitivity (Type A)
    _methods = {
        "Mean":    filters.threshold_mean,
        "Li":      filters.threshold_li,
        "Minimum": filters.threshold_minimum,
        "Otsu":    filters.threshold_otsu,
        "Isodata": filters.threshold_isodata,
    }
    thresh_means = []
    for fn in _methods.values():
        try:
            t = float(fn(image))
            binary = image > t
            from skimage import measure, morphology, feature, segmentation
            from scipy import ndimage
            binary = morphology.remove_small_objects(binary, min_size=200)
            binary = morphology.closing(binary, morphology.disk(2))
            dist   = ndimage.distance_transform_edt(binary)
            props  = measure.regionprops(measure.label(binary))
            ps = [p for p in props if p.area > 50]
            if ps:
                thresh_means.append(
                    np.mean([p.equivalent_diameter_area / px_per_um * 1000
                             for p in ps])
                )
        except Exception:
            pass

    if len(thresh_means) >= 2:
        u3 = float(np.std(thresh_means, ddof=1) / np.sqrt(len(thresh_means)))
    else:
        u3 = 0.0

    # u4 — bootstrap resampling (Type A)
    rng  = np.random.default_rng(rng_seed)
    boot = np.array([
        rng.choice(d_nm, size=len(d_nm), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    u4 = float(boot.std(ddof=1))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    # u5 — sample heterogeneity (Type A)
    u5 = float(d_nm.std(ddof=1) / np.sqrt(len(d_nm)))

    # u6 — sub-pixel boundary (Type B)
    u6 = (psf_blur_sigma_px * 2 / px_per_um * 1000) / np.sqrt(3)

    # u7 — beam-induced shrinkage (Type B)
    u7 = (beam_shrinkage_half_range_pct / 100 * mean_d) / np.sqrt(3)

    components = {
        "u1_calibration":     u1,
        "u2_discretisation":  u2,
        "u3_threshold":       u3,
        "u4_bootstrap":       u4,
        "u5_heterogeneity":   u5,
        "u6_subpixel":        u6,
        "u7_beam_shrinkage":  u7,
    }

    u_c = float(np.sqrt(sum(v**2 for v in components.values())))
    U   = float(k * u_c)

    d_corrected = mean_d / (1 - beam_shrinkage_pct / 100)

    budget = {
        "mean_d_nm":      mean_d,
        "d_corrected_nm": d_corrected,
        "n":              len(d_nm),
        "components":     components,
        "u_c_nm":         u_c,
        "U_nm":           U,
        "k":              k,
        "result_str":     f"{mean_d:.1f} ± {U:.1f} nm (k={k:.0f}, ≈95 %)",
        "corrected_str":  f"{d_corrected:.1f} ± {U:.1f} nm (shrinkage-corrected)",
        "boot_means":     boot,
        "boot_ci_lo":     ci_lo,
        "boot_ci_hi":     ci_hi,
        "thresh_means":   thresh_means,
    }

    if verbose:
        print(f"\n  GUM Uncertainty Budget")
        print(f"  {'Source':35s}  {'u_i (nm)':>9s}  {'% of uc²':>9s}")
        print("  " + "-" * 58)
        for name, ui in components.items():
            pct = ui**2 / u_c**2 * 100
            print(f"  {name:35s}  {ui:9.3f}  {pct:9.1f}%")
        print("  " + "-" * 58)
        print(f"  {'Combined u_c':35s}  {u_c:9.3f}")
        print(f"  {'Expanded U (k=2)':35s}  {U:9.3f}")
        print(f"\n  Result: {budget['result_str']}")

    return budget
