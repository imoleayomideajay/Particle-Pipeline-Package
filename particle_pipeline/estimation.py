"""
particle_pipeline.estimation
=============================
Physics-based estimation of DLS hydrodynamic diameter and RPS
volume-equivalent diameter from SEM image morphometry.

Theory
------
Each particle is modelled as a prolate spheroid with semi-major axis
a = r·√AR and semi-minor axis b = r/√AR, where r is the projected
equivalent radius and AR is the image-derived aspect ratio.

DLS hydrodynamic diameter
  Dh = Dv · (f/f₀)
  where f/f₀ is the Perrin shape correction for translational friction:
  f/f₀ = √(1−p²) / [p^(2/3) · ln((1+√(1−p²))/p)]   p = 1/AR

  Population DLS statistics are intensity-weighted (∝ Dh⁶) per ISO 22412.

RPS volume-equivalent diameter
  Dv = 2·(3V/4π)^(1/3)  where  V = (4/3)π·a·b²

  Population RPS statistics are volume-weighted (∝ Dv³).

References
----------
Perrin F. (1936). J. Phys. Radium 7(1), 1–11.
ISO 22412:2017. Particle Size Analysis — Dynamic Light Scattering.
Weatherall & Bhella (2015). ACS Nano 9(10), 10226–10234.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min, kstest


# ── 3D geometry ───────────────────────────────────────────────────────────────

def prolate_spheroid_geometry(
    d_um: np.ndarray,
    AR: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute prolate spheroid semi-axes, volume, and volume-equivalent diameter.

    Parameters
    ----------
    d_um : array-like
        Projected area-equivalent diameter (µm).
    AR : array-like
        Aspect ratio (≥ 1.0).

    Returns
    -------
    a_um, b_um : np.ndarray
        Semi-major and semi-minor axes (µm).
    volume_um3 : np.ndarray
        Spheroid volume (µm³).
    dv_um : np.ndarray
        Volume-equivalent sphere diameter (µm).
    """
    d_um = np.asarray(d_um, dtype=float)
    AR   = np.asarray(AR,   dtype=float)
    r    = d_um / 2.0
    a    = r * np.sqrt(AR)
    b    = r / np.sqrt(AR)
    vol  = (4.0 / 3.0) * np.pi * a * b**2
    dv   = 2.0 * (vol / ((4.0 / 3.0) * np.pi))**(1.0 / 3.0)
    return a, b, vol, dv


# ── Perrin correction ─────────────────────────────────────────────────────────

def perrin_correction(AR: np.ndarray) -> np.ndarray:
    """
    Perrin translational shape correction factor f/f₀ for a prolate spheroid.

    f/f₀ = √(1−p²) / [p^(2/3) · ln((1+√(1−p²))/p)]   where p = 1/AR

    For spheres (AR = 1): f/f₀ = 1.0.
    Always ≥ 1.0.

    Parameters
    ----------
    AR : array-like
        Aspect ratio values (≥ 1.0).

    Returns
    -------
    np.ndarray
        Correction factors, same shape as AR.
    """
    AR = np.asarray(AR, dtype=float)
    p  = 1.0 / np.maximum(AR, 1.0001)
    p  = np.minimum(p, 0.9999)
    num = np.sqrt(1.0 - p**2)
    den = (p**(2.0 / 3.0)) * np.log((1.0 + np.sqrt(1.0 - p**2)) / p)
    with np.errstate(divide="ignore", invalid="ignore"):
        ff0 = np.where(den > 0, num / den, 1.0)
    ff0 = np.where(AR <= 1.01, 1.0, ff0)
    return np.maximum(ff0, 1.0)


# ── DLS estimation ────────────────────────────────────────────────────────────

def estimate_dls(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimate per-particle DLS hydrodynamic diameter and population statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: diam_um, aspect_ratio.

    Returns
    -------
    dh_nm : np.ndarray
        Per-particle hydrodynamic diameter (nm).
    ff0 : np.ndarray
        Per-particle Perrin correction factor.
    stats : dict
        z_average_nm, z_average_um, PDI, dh_d10_um, dh_d50_um,
        dh_d90_um, number_mean_dh_nm, mean_ff0, max_ff0.
    """
    d_um = df["diam_um"].values
    AR   = df["aspect_ratio"].values

    _, _, vol, dv_um = prolate_spheroid_geometry(d_um, AR)
    ff0  = perrin_correction(AR)
    dh_um = dv_um * ff0
    dh_nm = dh_um * 1000.0

    # Intensity weighting ∝ Dh⁶
    w = dh_nm**6
    w = w / w.sum()

    z_avg  = float(np.sum(w * dh_nm))
    var    = float(np.sum(w * (dh_nm - z_avg)**2))
    PDI    = var / z_avg**2

    # Weighted percentiles
    idx = np.argsort(dh_nm)
    cumw = np.cumsum(w[idx])
    dh_d10 = float(np.interp(0.10, cumw, dh_nm[idx]))
    dh_d50 = float(np.interp(0.50, cumw, dh_nm[idx]))
    dh_d90 = float(np.interp(0.90, cumw, dh_nm[idx]))

    stats = {
        "z_average_nm":      z_avg,
        "z_average_um":      z_avg / 1000.0,
        "PDI":               PDI,
        "dh_d10_um":         dh_d10 / 1000.0,
        "dh_d50_um":         dh_d50 / 1000.0,
        "dh_d90_um":         dh_d90 / 1000.0,
        "number_mean_dh_nm": float(np.mean(dh_nm)),
        "mean_ff0":          float(np.mean(ff0)),
        "max_ff0":           float(np.max(ff0)),
    }
    return dh_nm, ff0, stats


# ── RPS estimation ────────────────────────────────────────────────────────────

def estimate_rps(
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """
    Estimate per-particle RPS volume-equivalent diameter and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: diam_um, aspect_ratio.

    Returns
    -------
    dv_nm : np.ndarray
        Per-particle volume-equivalent diameter (nm).
    stats : dict
        vol_mean_nm, vol_mean_um, vol_std_nm, rps_d10_um,
        rps_d50_um, rps_d90_um, span_rps, number_mean_dv_nm.
    """
    d_um = df["diam_um"].values
    AR   = df["aspect_ratio"].values

    _, _, _, dv_um = prolate_spheroid_geometry(d_um, AR)
    dv_nm = dv_um * 1000.0

    # Volume weighting ∝ Dv³
    w = dv_nm**3
    w = w / w.sum()

    vol_mean = float(np.sum(w * dv_nm))
    vol_std  = float(np.sqrt(np.sum(w * (dv_nm - vol_mean)**2)))

    idx = np.argsort(dv_nm)
    cumw = np.cumsum(w[idx])
    rps_d10 = float(np.interp(0.10, cumw, dv_nm[idx]))
    rps_d50 = float(np.interp(0.50, cumw, dv_nm[idx]))
    rps_d90 = float(np.interp(0.90, cumw, dv_nm[idx]))

    stats = {
        "vol_mean_nm":      vol_mean,
        "vol_mean_um":      vol_mean / 1000.0,
        "vol_std_nm":       vol_std,
        "rps_d10_um":       rps_d10 / 1000.0,
        "rps_d50_um":       rps_d50 / 1000.0,
        "rps_d90_um":       rps_d90 / 1000.0,
        "span_rps":         (rps_d90 - rps_d10) / rps_d50 if rps_d50 > 0 else 0.0,
        "number_mean_dv_nm": float(np.mean(dv_nm)),
    }
    return dv_nm, stats


# ── Distribution fitting ──────────────────────────────────────────────────────

def fit_distributions(diameters_nm: np.ndarray) -> dict:
    """
    Fit log-normal and Weibull distributions by MLE; select best by KS test.

    Parameters
    ----------
    diameters_nm : np.ndarray
        Array of particle diameters in nm.

    Returns
    -------
    dict with keys 'lognormal', 'weibull', 'best_fit'.
    Each sub-dict contains: params, ks_stat, ks_p, dist_obj.
    """
    results = {}

    shape_ln, loc_ln, scale_ln = lognorm.fit(diameters_nm, floc=0)
    ks_ln, p_ln = kstest(diameters_nm, "lognorm",
                         args=(shape_ln, loc_ln, scale_ln))
    results["lognormal"] = {
        "sigma": shape_ln, "loc": loc_ln, "scale": scale_ln,
        "mu_log": float(np.log(scale_ln)),
        "ks_stat": float(ks_ln), "ks_p": float(p_ln),
        "dist_obj": lognorm(shape_ln, loc=loc_ln, scale=scale_ln),
    }

    shape_wb, loc_wb, scale_wb = weibull_min.fit(diameters_nm, floc=0)
    ks_wb, p_wb = kstest(diameters_nm, "weibull_min",
                         args=(shape_wb, loc_wb, scale_wb))
    results["weibull"] = {
        "k": shape_wb, "loc": loc_wb, "lambda": scale_wb,
        "ks_stat": float(ks_wb), "ks_p": float(p_wb),
        "dist_obj": weibull_min(shape_wb, loc=loc_wb, scale=scale_wb),
    }

    results["best_fit"] = "lognormal" if p_ln >= p_wb else "weibull"
    return results
