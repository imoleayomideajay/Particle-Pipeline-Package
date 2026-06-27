"""
particle_pipeline.morphometry
==============================
Extract per-particle shape descriptors from segmented regions.

Descriptors
-----------
diam_um         Equivalent circular diameter (area-based), µm
diam_nm         Same, nm
aspect_ratio    Major axis / minor axis  (≥ 1.0)
circularity     4πA / P²  (1.0 = perfect circle)
solidity        Area / convex-hull area  (< 0.80 → probable agglomerate)
elongation      1 − minor/major  (0 = circle, 1 = line)

Size classes
------------
Fine            d < 5 µm
Medium          5 µm ≤ d < 15 µm
Coarse          d ≥ 15 µm

Shape classes (applied in priority order)
-----------------------------------------
Irregular/Agglomerate  solidity < 0.80  (aggregates, touching particles)
Fibre                  AR ≥ 3.0 AND circularity < 0.30
                       (microplastic fibres, needle crystals)
Elongated/Needle       AR ≥ 2.5  (rods, elongated fragments)
Spherical              circularity ≥ 0.85
Sub-spherical          all remaining
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties


def extract_morphometry(
    props: list[RegionProperties],
    px_per_um: float,
    nominal_d_um: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert a list of scikit-image RegionProperties to a tidy DataFrame.

    Parameters
    ----------
    props : list of RegionProperties
        Output of ``skimage.measure.regionprops``.
    px_per_um : float
        Calibration in pixels per micrometre.
    nominal_d_um : float, optional
        Nominal particle diameter (µm). If supplied, particles outside
        [0.40 × nominal, 1.80 × nominal] are excluded.

    Returns
    -------
    pd.DataFrame
        One row per particle with columns:
        particle_id, diam_um, diam_nm, area_px, perimeter_px,
        circularity, aspect_ratio, solidity, elongation,
        centroid_x, centroid_y, size_class, shape_class,
        is_agglomerate.

    Examples
    --------
    >>> df = extract_morphometry(props, px_per_um=213.0, nominal_d_um=0.48)
    >>> df[["diam_nm", "circularity", "shape_class"]].describe()
    """
    records = []
    for i, p in enumerate(props):
        area  = p.area
        perim = p.perimeter if p.perimeter > 0 else 1e-6
        maj   = p.axis_major_length
        min_  = p.axis_minor_length if p.axis_minor_length > 0 else 1e-6

        d_um  = p.equivalent_diameter_area / px_per_um
        AR    = maj / min_
        circ  = (4 * np.pi * area) / (perim**2)
        sol   = p.solidity
        elong = 1 - (min_ / maj) if maj > 0 else 0.0

        # Classification
        if d_um < 5.0:
            size_class = "Fine (<5 µm)"
        elif d_um < 15.0:
            size_class = "Medium (5–15 µm)"
        else:
            size_class = "Coarse (>15 µm)"

        if sol < 0.80:
            shape_class = "Irregular/Agglomerate"
        elif AR >= 3.0 and circ < 0.30:
            shape_class = "Fibre"
        elif AR >= 2.5:
            shape_class = "Elongated/Needle"
        elif circ >= 0.85:
            shape_class = "Spherical"
        else:
            shape_class = "Sub-spherical"

        records.append({
            "particle_id":    i + 1,
            "diam_um":        round(d_um, 4),
            "diam_nm":        round(d_um * 1000, 2),
            "area_px":        int(area),
            "perimeter_px":   round(perim, 2),
            "circularity":    round(circ, 4),
            "aspect_ratio":   round(AR, 4),
            "solidity":       round(sol, 4),
            "elongation":     round(elong, 4),
            "centroid_x":     round(p.centroid[0], 1),
            "centroid_y":     round(p.centroid[1], 1),
            "size_class":     size_class,
            "shape_class":    shape_class,
            "is_agglomerate": sol < 0.80,
        })

    df = pd.DataFrame(records)

    # Size filter
    if nominal_d_um and len(df):
        lo = nominal_d_um * 0.40
        hi = nominal_d_um * 1.80
        df = df[(df["diam_um"] >= lo) & (df["diam_um"] <= hi)]
    elif len(df):
        df = df[df["diam_um"] >= 0.05]   # remove sub-pixel noise

    return df.reset_index(drop=True)


def compute_psd(df: pd.DataFrame) -> dict:
    """
    Compute number-weighted PSD statistics from a morphometry DataFrame.

    Returns
    -------
    dict
        n, mean_um, std_um, min_um, max_um, d10, d50, d90,
        span, cv_pct, n_agglomerates.
    """
    d = df["diam_um"].values
    if len(d) == 0:
        return {}
    d10, d50, d90 = np.percentile(d, [10, 50, 90])
    return {
        "n":              len(d),
        "mean_um":        float(np.mean(d)),
        "std_um":         float(np.std(d)),
        "min_um":         float(np.min(d)),
        "max_um":         float(np.max(d)),
        "d10":            float(d10),
        "d50":            float(d50),
        "d90":            float(d90),
        "span":           float((d90 - d10) / d50) if d50 > 0 else 0.0,
        "cv_pct":         float(np.std(d) / np.mean(d) * 100),
        "n_agglomerates": int(df["is_agglomerate"].sum()),
    }


def shape_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a concise shape-class frequency table from a morphometry DataFrame.

    Useful for microplastics classification reports and regulatory submissions.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_morphometry().

    Returns
    -------
    pd.DataFrame
        Columns: shape_class, count, pct, mean_diam_nm, mean_AR,
        mean_circularity, mean_solidity.

    Examples
    --------
    >>> summary = shape_summary(df)
    >>> print(summary.to_string(index=False))
    """
    if df.empty or "shape_class" not in df.columns:
        return pd.DataFrame()

    rows = []
    for cls, grp in df.groupby("shape_class", sort=False):
        rows.append({
            "shape_class":      cls,
            "count":            len(grp),
            "pct":              round(len(grp) / len(df) * 100, 1),
            "mean_diam_nm":     round(grp["diam_nm"].mean(), 1),
            "mean_AR":          round(grp["aspect_ratio"].mean(), 2),
            "mean_circularity": round(grp["circularity"].mean(), 3),
            "mean_solidity":    round(grp["solidity"].mean(), 3),
        })
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    return out.reset_index(drop=True)
