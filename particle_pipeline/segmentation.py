"""
particle_pipeline.segmentation
===============================
Adaptive threshold selection, morphological cleanup, and watershed
separation of touching particles.

The adaptive selector evaluates six global threshold methods and scores
each on four image-quality criteria:

  S_circ  (weight 0.40) — mean particle circularity; rewards well-separated
                           round objects; penalises over/under-segmentation
  S_count (weight 0.20) — log-Gaussian penalty for outlier particle counts
  S_fg    (weight 0.20) — foreground fraction; rewards 5–45 % coverage
  S_size  (weight 0.20) — size accuracy relative to a known nominal diameter

Composite score:  S = 0.40·S_circ + 0.20·S_count + 0.20·S_fg + 0.20·S_size
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage, signal
from skimage import feature, filters, measure, morphology, segmentation

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ImageDiagnostics:
    """Histogram and illumination analysis of the input image."""
    mean_intensity:     float = 0.0
    std_intensity:      float = 0.0
    skewness:           float = 0.0
    n_histogram_peaks:  int   = 0
    peak_intensities:   list  = field(default_factory=list)
    illumination_std:   float = 0.0
    is_bright_particles: bool = True
    recommended_family: str  = "global"


@dataclass
class ThresholdResult:
    """Segmentation quality scores for one threshold method."""
    name:             str   = ""
    threshold:        float = 0.0
    n_raw:            int   = 0
    n_filtered:       int   = 0
    mean_d_nm:        float = 0.0
    std_d_nm:         float = 0.0
    mean_circ:        float = 0.0
    mean_solidity:    float = 0.0
    fg_pct:           float = 0.0
    score_circularity: float = 0.0
    score_count:      float = 0.0
    score_fg:         float = 0.0
    score_size:       float = 0.0
    composite_score:  float = 0.0
    rank:             int   = 0


# ── Image diagnostics ─────────────────────────────────────────────────────────

def diagnose_image(image: np.ndarray) -> ImageDiagnostics:
    """
    Analyse image histogram and illumination uniformity.

    Returns
    -------
    ImageDiagnostics
        Mean, std, skewness, number of histogram peaks, illumination
        uniformity, particle polarity, and recommended threshold family.
    """
    d = ImageDiagnostics()
    flat = image.ravel()
    d.mean_intensity = float(flat.mean())
    d.std_intensity  = float(flat.std())
    d.skewness = float(np.mean(((flat - flat.mean()) / (flat.std() + 1e-9))**3))

    # Histogram peak count
    hist, edges = np.histogram(flat, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    smoothed = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
    peaks, _ = signal.find_peaks(
        smoothed, height=smoothed.max() * 0.05, distance=15
    )
    d.n_histogram_peaks = len(peaks)
    d.peak_intensities  = [float(centres[p]) for p in peaks]

    # Illumination uniformity
    h, w = image.shape
    q_means = [
        image[:h//2, :w//2].mean(), image[:h//2, w//2:].mean(),
        image[h//2:, :w//2].mean(), image[h//2:, w//2:].mean(),
    ]
    d.illumination_std = float(np.std(q_means))

    # Particle polarity
    try:
        t_ref = filters.threshold_otsu(image)
        fg_at_otsu = (image > t_ref).mean()
        d.is_bright_particles = fg_at_otsu < 0.55
    except Exception:
        d.is_bright_particles = True

    d.recommended_family = (
        "local" if d.illumination_std > 0.05 else "global"
    )
    return d


# ── Threshold scoring ─────────────────────────────────────────────────────────

def _sigmoid(x, mu=0.70, k=12):
    return 1.0 / (1.0 + np.exp(-k * (x - mu)))


def _score_circularity(mean_circ: float) -> float:
    if mean_circ <= 0:
        return 0.0
    return float(np.clip(_sigmoid(mean_circ), 0, 1))


def _score_count(n: int, all_counts: list) -> float:
    if not all_counts or n == 0:
        return 0.0
    median_n = np.median(all_counts)
    if median_n == 0:
        return 0.0
    ratio = n / median_n
    return float(np.clip(np.exp(-2.0 * (np.log(ratio + 1e-9))**2), 0, 1))


def _score_fg(fg_pct: float, lo=5.0, hi=45.0) -> float:
    if lo <= fg_pct <= hi:
        return 1.0
    elif fg_pct < lo:
        return float(fg_pct / lo)
    else:
        return float(max(0, 1.0 - (fg_pct - hi) / hi))


def _score_size(mean_d_nm: float, nominal_d_nm: Optional[float]) -> float:
    if nominal_d_nm is None or nominal_d_nm <= 0 or mean_d_nm <= 0:
        return 1.0
    err = abs(mean_d_nm - nominal_d_nm) / nominal_d_nm
    return float(max(0.0, 1.0 - err / 0.30))


# ── Core segmentation worker ──────────────────────────────────────────────────

def _segment_with_threshold(
    image: np.ndarray,
    thresh: float,
    is_bright: bool,
    min_size_px: float,
    min_dist_px: float,
    size_lo_px: float,
    size_hi_px: float,
    px_per_um: float,
    use_watershed: bool = True,
) -> dict:
    """Run a complete segmentation for a single threshold value.

    Parameters
    ----------
    use_watershed : bool
        If True (default), apply watershed separation of touching particles.
        If False, return simple connected-components labelling — equivalent
        to the standard approach used by most detector software.
    """
    binary = image > thresh if is_bright else image < thresh
    binary = morphology.remove_small_objects(binary, min_size=int(min_size_px))
    binary = morphology.remove_small_holes(
        binary, area_threshold=int(min_size_px // 2)
    )
    binary = morphology.closing(binary, morphology.disk(3))

    if use_watershed:
        distance = ndimage.distance_transform_edt(binary)
        coords   = feature.peak_local_max(
            distance, min_distance=int(min_dist_px), labels=binary
        )
        mask = np.zeros(distance.shape, dtype=bool)
        if len(coords):
            mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        labeled     = segmentation.watershed(-distance, markers, mask=binary)
    else:
        # Connected-components only — standard detector software approach
        labeled, _ = ndimage.label(binary)

    props_all   = measure.regionprops(labeled, intensity_image=image)

    props_f = [
        p for p in props_all
        if size_lo_px <= p.equivalent_diameter_area <= size_hi_px
    ]
    diams = [p.equivalent_diameter_area / px_per_um * 1000 for p in props_f]
    circ  = [
        (4 * np.pi * p.area) / p.perimeter**2
        for p in props_f if p.perimeter > 0
    ]
    sols  = [p.solidity for p in props_f]

    return {
        "labeled":    labeled,
        "binary":     binary,
        "props_f":    props_f,
        "n_raw":      len(props_all),
        "n_filtered": len(props_f),
        "mean_d":     float(np.mean(diams)) if diams else 0.0,
        "std_d":      float(np.std(diams))  if diams else 0.0,
        "mean_circ":  float(np.mean(circ))  if circ  else 0.0,
        "mean_sol":   float(np.mean(sols))  if sols  else 0.0,
        "fg_pct":     float(binary.sum() / binary.size * 100),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def segment(
    image: np.ndarray,
    px_per_um: float,
    nominal_d_um: Optional[float] = None,
    min_d_um: Optional[float] = None,
    max_d_um: Optional[float] = None,
    w_circularity: float = 0.40,
    w_count:       float = 0.20,
    w_fg:          float = 0.20,
    w_size:        float = 0.20,
    force_method: Optional[str] = None,
    no_watershed: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, ThresholdResult, list, ImageDiagnostics]:
    """
    Adaptively segment particles from a normalised SEM image.

    Evaluates six global threshold methods (Otsu, Li, Yen, Isodata,
    Mean, Minimum), scores each on four quality criteria, and selects
    the best automatically.  Touching particles are separated by
    watershed segmentation on the Euclidean distance transform.

    Parameters
    ----------
    image : np.ndarray
        Normalised [0, 1] greyscale image.
    px_per_um : float
        Calibration in pixels per micrometre.
    nominal_d_um : float, optional
        Expected particle diameter (µm). Used for size-accuracy scoring
        and size filter bounds.
    min_d_um : float, optional
        Minimum particle diameter to retain (default 0.40 × nominal).
    max_d_um : float, optional
        Maximum particle diameter to retain (default 1.80 × nominal).
    w_circularity, w_count, w_fg, w_size : float
        Scoring weights (must sum to 1.0).
    force_method : str, optional
        Force a specific threshold method by name, bypassing adaptive
        scoring. Useful for comparisons and demonstrations.
        Options: "Otsu", "Li", "Yen", "Isodata", "Mean", "Minimum".
        If None (default), the adaptive selector chooses automatically.
    no_watershed : bool
        If True, skip watershed separation and return connected
        components only. Useful for comparing watershed vs no-watershed.
        Default False.
    verbose : bool
        Print diagnostic table to stdout.

    Returns
    -------
    labeled : np.ndarray
        Integer label array (0 = background).
    binary : np.ndarray
        Boolean binary mask from the selected threshold.
    props : list
        scikit-image RegionProperties for retained particles.
    best : ThresholdResult
        Scores and metadata for the winning threshold method.
    all_results : list of ThresholdResult
        All candidates, ranked by composite score.
    diagnostics : ImageDiagnostics
        Image characterisation results.

    Examples
    --------
    >>> labeled, binary, props, best, all_results, diag = segment(
    ...     image, px_per_um=213.0, nominal_d_um=0.48
    ... )
    >>> print(f"Selected: {best.name}  score={best.composite_score:.3f}")
    """
    diag = diagnose_image(image)

    # Size filter bounds
    if nominal_d_um:
        min_d_um = min_d_um or nominal_d_um * 0.40
        max_d_um = max_d_um or nominal_d_um * 1.80
    min_d_um = min_d_um or 0.1
    max_d_um = max_d_um or max(image.shape) / px_per_um

    size_lo_px   = min_d_um * px_per_um
    size_hi_px   = max_d_um * px_per_um
    min_size_px  = np.pi * (size_lo_px / 2)**2 * 0.5
    min_dist_px  = (size_lo_px + size_hi_px) / 4 * 0.35
    nominal_d_nm = nominal_d_um * 1000 if nominal_d_um else None

    # Candidate thresholds
    _methods = {
        "Otsu":    filters.threshold_otsu,
        "Li":      filters.threshold_li,
        "Yen":     filters.threshold_yen,
        "Isodata": filters.threshold_isodata,
        "Mean":    filters.threshold_mean,
        "Minimum": filters.threshold_minimum,
    }
    raw_thresholds = {}
    for name, fn in _methods.items():
        try:
            raw_thresholds[name] = float(fn(image))
        except Exception:
            pass

    # CLAHE + Otsu when illumination is uneven
    if diag.illumination_std > 0.05:
        try:
            from skimage import exposure
            img_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
            raw_thresholds["CLAHE+Otsu"] = float(filters.threshold_otsu(img_eq))
        except Exception:
            pass

    # Apply force_method override if requested
    if force_method is not None:
        fname = force_method.capitalize()
        # Normalise common aliases
        fname = {"Otsu": "Otsu", "Li": "Li", "Yen": "Yen",
                 "Isodata": "Isodata", "Mean": "Mean",
                 "Minimum": "Minimum"}.get(fname, fname)
        if fname in raw_thresholds:
            raw_thresholds = {fname: raw_thresholds[fname]}
        else:
            import warnings as _w
            _w.warn(f"force_method '{force_method}' not available; "
                    f"using adaptive selection.")

    # Segment with each candidate
    seg_results = {
        name: _segment_with_threshold(
            image, t, diag.is_bright_particles,
            min_size_px, min_dist_px,
            size_lo_px, size_hi_px, px_per_um,
            use_watershed=not no_watershed,
        )
        for name, t in raw_thresholds.items()
    }

    all_counts = [r["n_filtered"] for r in seg_results.values()]

    # Score
    candidates = []
    for name, t in raw_thresholds.items():
        r = seg_results[name]
        c = ThresholdResult()
        c.name           = name
        c.threshold      = t
        c.n_raw          = r["n_raw"]
        c.n_filtered     = r["n_filtered"]
        c.mean_d_nm      = r["mean_d"]
        c.std_d_nm       = r["std_d"]
        c.mean_circ      = r["mean_circ"]
        c.mean_solidity  = r["mean_sol"]
        c.fg_pct         = r["fg_pct"]

        c.score_circularity = _score_circularity(r["mean_circ"])
        c.score_count       = _score_count(r["n_filtered"], all_counts)
        c.score_fg          = _score_fg(r["fg_pct"])
        c.score_size        = _score_size(r["mean_d"], nominal_d_nm)

        c.composite_score = (
            w_circularity * c.score_circularity
            + w_count     * c.score_count
            + w_fg        * c.score_fg
            + w_size      * c.score_size
        )
        candidates.append(c)

    candidates.sort(key=lambda x: x.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    best    = candidates[0]
    best_r  = seg_results[best.name]

    if verbose:
        print(f"\n  Adaptive threshold selection:")
        print(f"  {'Method':12s} {'t':7s} {'n':5s} {'d_mean':8s} "
              f"{'circ':6s} {'Score':6s}")
        print("  " + "-"*50)
        for c in candidates:
            star = " ★" if c.rank == 1 else ""
            print(f"  {c.name:12s} {c.threshold:.4f}  "
                  f"{c.n_filtered:4d}  {c.mean_d_nm:7.1f}  "
                  f"{c.mean_circ:.3f}  {c.composite_score:.3f}{star}")

    return (
        best_r["labeled"],
        best_r["binary"],
        best_r["props_f"],
        best,
        candidates,
        diag,
    )
