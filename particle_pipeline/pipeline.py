"""
particle_pipeline.pipeline
===========================
High-level public API.

Usage
-----
>>> from particle_pipeline import run_pipeline
>>> results = run_pipeline("my_particles.tif", nominal_d_um=0.48)
>>> results.save_report("output/")
>>> print(results.summary())
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Results container ─────────────────────────────────────────────────────────

@dataclass
class PipelineResults:
    """
    Container for all outputs of a particle_pipeline analysis run.

    Attributes
    ----------
    image : np.ndarray
        Normalised SEM image used for analysis.
    labeled : np.ndarray
        Labelled segmentation array (0 = background).
    binary : np.ndarray
        Binary segmentation mask.
    px_per_um : float
        Calibration used (pixels per micrometre).
    image_path : str
        Path to the source image.

    df : pd.DataFrame
        Per-particle morphometry with DLS/RPS estimates.
        Columns: particle_id, diam_um, diam_nm, aspect_ratio,
        circularity, solidity, elongation, shape_class, size_class,
        is_agglomerate, Dh_nm, Dv_nm, ff0,
        refined_d_nm, boundary_shift_nm (if refinement run).

    psd_stats : dict
        Number-weighted PSD statistics (mean, std, D10/D50/D90, span, CV).
    dls_stats : dict
        Estimated DLS population statistics (Z-average, PDI, D50, etc.).
    rps_stats : dict
        Estimated RPS population statistics (vol-mean, D50, span, etc.).
    fit_results : dict
        Log-normal and Weibull MLE fit results.
    budget : dict
        GUM seven-component uncertainty budget.

    best_threshold : ThresholdResult
        Winning threshold method and its scores.
    all_thresholds : list
        All six threshold candidates, ranked.
    diagnostics : ImageDiagnostics
        Image histogram and illumination diagnostics.
    """
    image:         np.ndarray          = field(repr=False)
    labeled:       np.ndarray          = field(repr=False)
    binary:        np.ndarray          = field(repr=False)
    px_per_um:     float               = 0.0
    image_path:    str                 = ""

    df:            pd.DataFrame        = field(default_factory=pd.DataFrame)
    psd_stats:     dict                = field(default_factory=dict)
    dls_stats:     dict                = field(default_factory=dict)
    rps_stats:     dict                = field(default_factory=dict)
    fit_results:   dict                = field(default_factory=dict)
    budget:        dict                = field(default_factory=dict)

    best_threshold: object             = None
    all_thresholds: list               = field(default_factory=list)
    diagnostics:    object             = None

    # Internal arrays (not serialised)
    _dh_nm:  np.ndarray = field(default=None, repr=False)
    _dv_nm:  np.ndarray = field(default=None, repr=False)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Number of retained particles."""
        return len(self.df)

    @property
    def d50_nm(self) -> float:
        """Number-weighted D50 (nm)."""
        return float(self.psd_stats.get("d50", 0.0)) * 1000

    @property
    def expanded_uncertainty_nm(self) -> float:
        """GUM expanded uncertainty U (nm), k=2."""
        return float(self.budget.get("U_nm", 0.0))

    @property
    def result_string(self) -> str:
        """Formatted measurement result string."""
        return self.budget.get("result_str", "")

    # ── Save / export ─────────────────────────────────────────────────────────

    def save_report(
        self,
        output_dir: str | Path,
        prefix: str = "particle_pipeline",
        figures: bool = True,
        pdf: bool = True,
        csv: bool = True,
        json_budget: bool = True,
    ) -> dict[str, Path]:
        """
        Save all outputs to a directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write outputs (created if it does not exist).
        prefix : str
            Filename prefix for all output files.
        figures, pdf, csv, json_budget : bool
            Toggle individual output types.

        Returns
        -------
        dict
            Mapping of output type to Path object.

        Examples
        --------
        >>> paths = results.save_report("output/", prefix="psl_480nm")
        >>> print(paths["pdf"])
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        if csv:
            csv_path = output_dir / f"{prefix}_particles.csv"
            self.df.to_csv(csv_path, index=False)
            saved["csv"] = csv_path

        if json_budget and self.budget:
            jpath = output_dir / f"{prefix}_uncertainty_budget.json"
            _safe_budget = {
                k: v for k, v in self.budget.items()
                if not isinstance(v, np.ndarray)
            }
            _safe_budget["components"] = self.budget.get("components", {})
            with open(jpath, "w") as f:
                json.dump(_safe_budget, f, indent=2, default=float)
            saved["json"] = jpath

        if figures:
            fig_paths = self._save_figures(output_dir, prefix)
            saved.update(fig_paths)

        if pdf:
            pdf_path = self._save_pdf(output_dir, prefix)
            saved["pdf"] = pdf_path

        return saved

    def _save_figures(self, output_dir: Path, prefix: str) -> dict:
        """Generate and save the four publication figures."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage import color as skcolor
        import matplotlib.gridspec as mgridspec

        saved = {}

        # ── Figure 1: Pipeline overview ───────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("A  Raw SEM image"); axes[0].axis("off")

        overlay = skcolor.label2rgb(self.labeled, image=self.image,
                                    bg_label=0, alpha=0.45)
        axes[1].imshow(overlay)
        axes[1].set_title(f"B  Segmentation (n={self.n})"); axes[1].axis("off")

        d_nm = self.df["diam_nm"].values
        bins = np.linspace(d_nm.min() * 0.8, d_nm.max() * 1.2, 25)
        axes[2].hist(d_nm, bins=bins, color="#2E86AB", edgecolor="white",
                     alpha=0.75, label=f"Image D50={np.median(d_nm):.0f} nm")
        if self._dh_nm is not None:
            axes[2].hist(self._dh_nm, bins=bins, color="#E84855",
                         edgecolor="white", alpha=0.55,
                         label=f"DLS est. D50={np.median(self._dh_nm):.0f} nm")
        axes[2].set_xlabel("Diameter (nm)"); axes[2].legend(fontsize=8)
        axes[2].set_title("C  Particle size distributions")
        for sp in ["top", "right"]:
            axes[2].spines[sp].set_visible(False)

        plt.tight_layout()
        p = output_dir / f"{prefix}_fig1_overview.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved["fig1"] = p

        # ── Figure 2: Uncertainty budget ──────────────────────────────────
        if self.budget:
            comp = self.budget.get("components", {})
            u_c  = self.budget.get("u_c_nm", 1.0)
            names = [k.replace("u", "u").replace("_", " ").title()
                     for k in comp]
            vals  = list(comp.values())
            pcts  = [v**2 / u_c**2 * 100 for v in vals]

            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.barh(names[::-1], vals[::-1], color="#2E86AB", edgecolor="white")
            ax.axvline(u_c, color="black", lw=1.5, ls="--",
                       label=f"u_c = {u_c:.1f} nm")
            ax.set_xlabel("Standard uncertainty (nm)")
            ax.set_title("GUM Uncertainty Components")
            ax.legend(fontsize=9)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.tight_layout()
            p2 = output_dir / f"{prefix}_fig2_uncertainty.png"
            fig2.savefig(p2, dpi=200, bbox_inches="tight")
            plt.close(fig2)
            saved["fig2"] = p2

        return saved

    def _save_pdf(self, output_dir: Path, prefix: str) -> Path:
        """Generate a PDF report."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER
        from datetime import datetime

        pdf_path = output_dir / f"{prefix}_report.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        S = getSampleStyleSheet()
        ts = ParagraphStyle("T", parent=S["Title"], fontSize=15,
                             textColor=rl_colors.HexColor("#1a1a2e"),
                             alignment=TA_CENTER, spaceAfter=4)
        ss = ParagraphStyle("S", parent=S["Normal"], fontSize=9,
                             textColor=rl_colors.HexColor("#555555"),
                             alignment=TA_CENTER, spaceAfter=3)
        hs = ParagraphStyle("H", parent=S["Heading2"], fontSize=11,
                             textColor=rl_colors.HexColor("#2E86AB"),
                             spaceBefore=10, spaceAfter=4)
        bs = ParagraphStyle("B", parent=S["Normal"], fontSize=9.5, leading=14)

        def HR():
            return HRFlowable(width="100%", thickness=0.5,
                              color=rl_colors.HexColor("#cccccc"), spaceAfter=6)

        def make_table(data, col_widths, hcol="#2E86AB"):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,0), rl_colors.HexColor(hcol)),
                ("TEXTCOLOR",     (0,0),(-1,0), rl_colors.white),
                ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0),(-1,-1), 8.5),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [rl_colors.HexColor("#f7f9fc"), rl_colors.white]),
                ("ALIGN",         (1,0),(-1,-1), "CENTER"),
                ("GRID",          (0,0),(-1,-1), 0.4,
                 rl_colors.HexColor("#cccccc")),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ]))
            return t

        story = []
        now = datetime.now().strftime("%d %B %Y, %H:%M")

        story.append(Paragraph("Particle Size Analysis Report", ts))
        story.append(Paragraph(
            f"particle-pipeline v{self._version()}  ·  {self.image_path}  ·  {now}",
            ss))
        story.append(HR())

        # Summary
        story.append(Paragraph("Analysis Summary", hs))
        story.append(Paragraph(
            f"<b>Threshold method:</b> {self.best_threshold.name if self.best_threshold else '—'} "
            f"(score {self.best_threshold.composite_score:.3f})<br/>"
            f"<b>Particles retained:</b> {self.n}<br/>"
            f"<b>D50 (image, number-wt):</b> {self.d50_nm:.1f} nm<br/>"
            f"<b>GUM result:</b> {self.result_string}",
            bs))

        # Size table
        psd = self.psd_stats
        dls = self.dls_stats
        rps = self.rps_stats
        if psd:
            story.append(Paragraph("Particle Size Results", hs))
            tdata = [
                ["Metric", "Image (number-wt)", "DLS (est.)", "RPS (est.)"],
                ["n",     str(psd.get("n","")),        "—",    "—"],
                ["Mean (nm)", f"{psd.get('mean_um',0)*1000:.1f}",
                 f"{dls.get('number_mean_dh_nm',0):.1f}",
                 f"{rps.get('number_mean_dv_nm',0):.1f}"],
                ["D10 (nm)",  f"{psd.get('d10',0)*1000:.1f}",
                 f"{dls.get('dh_d10_um',0)*1000:.1f}",
                 f"{rps.get('rps_d10_um',0)*1000:.1f}"],
                ["D50 (nm)",  f"{psd.get('d50',0)*1000:.1f}",
                 f"{dls.get('dh_d50_um',0)*1000:.1f}",
                 f"{rps.get('rps_d50_um',0)*1000:.1f}"],
                ["D90 (nm)",  f"{psd.get('d90',0)*1000:.1f}",
                 f"{dls.get('dh_d90_um',0)*1000:.1f}",
                 f"{rps.get('rps_d90_um',0)*1000:.1f}"],
            ]
            story.append(make_table(
                tdata, [4*cm, 3.5*cm, 3.5*cm, 3.5*cm]))

        # Uncertainty budget
        if self.budget and self.budget.get("components"):
            story.append(Paragraph("GUM Uncertainty Budget", hs))
            comp = self.budget["components"]
            u_c  = self.budget["u_c_nm"]
            bdata = [["Source", "Type", "u_i (nm)", "% of u_c²"]]
            type_map = {
                "u1_calibration": "B", "u2_discretisation": "B",
                "u3_threshold": "A",   "u4_bootstrap": "A",
                "u5_heterogeneity": "A", "u6_subpixel": "B",
                "u7_beam_shrinkage": "B",
            }
            for name, val in comp.items():
                bdata.append([
                    name.replace("_", " ").title(),
                    type_map.get(name, "?"),
                    f"{val:.3f}",
                    f"{val**2/u_c**2*100:.1f}%",
                ])
            bdata.append(["Combined u_c", "—",
                          f"{u_c:.3f}", "100%"])
            bdata.append([f"Expanded U (k={self.budget['k']:.0f})", "—",
                          f"{self.budget['U_nm']:.3f}", "—"])
            story.append(make_table(
                bdata, [6*cm, 1.5*cm, 2.5*cm, 2.5*cm],
                hcol="#A23B72"))

        story.append(Spacer(1, 16))
        story.append(HR())
        story.append(Paragraph(
            "Generated by particle-pipeline. "
            "https://github.com/imoleayomide/particle-pipeline",
            ParagraphStyle("F", parent=S["Normal"], fontSize=8,
                           textColor=rl_colors.HexColor("#888888"),
                           alignment=TA_CENTER)))
        doc.build(story)
        return pdf_path

    @staticmethod
    def _version():
        try:
            from particle_pipeline._version import __version__
            return __version__
        except Exception:
            return "?"

    # ── Text summary ──────────────────────────────────────────────────────────

    def shape_summary(self):
        """
        Return a shape-class frequency table for the retained particles.

        Particularly useful for microplastics classification reports.
        Delegates to morphometry.shape_summary().

        Returns
        -------
        pd.DataFrame
            Columns: shape_class, count, pct, mean_diam_nm,
            mean_AR, mean_circularity, mean_solidity.

        Examples
        --------
        >>> print(results.shape_summary())
        """
        from particle_pipeline.morphometry import shape_summary
        return shape_summary(self.df)

    def summary(self) -> str:
        """Return a concise text summary of the analysis."""
        lines = [
            "particle-pipeline results",
            "=" * 40,
            f"Image:           {Path(self.image_path).name}",
            f"Calibration:     {self.px_per_um:.1f} px/µm",
            f"Threshold:       {self.best_threshold.name if self.best_threshold else '—'}"
            f"  (score={self.best_threshold.composite_score:.3f})" if self.best_threshold else "",
            f"Particles:       {self.n}",
            f"Image D50:       {self.d50_nm:.1f} nm",
        ]
        if self.dls_stats:
            lines.append(
                f"DLS est. D50:    "
                f"{self.dls_stats.get('dh_d50_um', 0)*1000:.1f} nm  "
                f"(Z-avg={self.dls_stats.get('z_average_nm', 0):.1f} nm, "
                f"PDI={self.dls_stats.get('PDI', 0):.4f})"
            )
        if self.rps_stats:
            lines.append(
                f"RPS est. D50:    "
                f"{self.rps_stats.get('rps_d50_um', 0)*1000:.1f} nm"
            )
        if self.budget:
            lines.append(f"GUM result:      {self.result_string}")
        return "\n".join(lines)


# ── Public run_pipeline function ──────────────────────────────────────────────

# ── Material presets for beam_shrinkage_pct ──────────────────────────────────
MATERIAL_PRESETS = {
    "psl":          1.0,   # polystyrene latex — ~1 % per min at 5 kV
    "polymer":      1.0,   # generic polymer (PE, PP, PMMA)
    "microplastic": 1.0,   # same as polymer
    "organic":      0.5,   # biological / organic (conservative)
    "metal_oxide":  0.0,   # TiO2, CeO2, ZnO — beam stable
    "metal":        0.0,   # Au, Ag, Pt — beam stable
    "silica":       0.0,   # SiO2 — beam stable
    "inorganic":    0.0,   # generic inorganic
}


def run_pipeline(
    image_path: str | Path,
    nominal_d_um: Optional[float] = None,
    px_per_um: Optional[float] = None,
    min_d_um: Optional[float] = None,
    max_d_um: Optional[float] = None,
    run_refinement: bool = True,
    run_uncertainty: bool = True,
    scale_bar_px: Optional[int] = None,
    material: Optional[str] = None,
    beam_shrinkage_pct: Optional[float] = None,
    force_method: Optional[str] = None,
    no_watershed: bool = False,
    verbose: bool = True,
) -> PipelineResults:
    """
    Run the complete particle analysis pipeline on a single SEM image.

    Parameters
    ----------
    image_path : str or Path
        Path to the SEM TIFF image.
    nominal_d_um : float, optional
        Expected particle diameter (µm). Used for size-accuracy scoring
        and the GUM uncertainty budget.
    px_per_um : float, optional
        Manual calibration override (pixels per micrometre).
        Auto-extracted from JEOL/Zeiss/FEI metadata if not supplied.
    min_d_um, max_d_um : float, optional
        Size filter bounds (µm).
        Defaults: 0.40 × nominal and 1.80 × nominal.
    run_refinement : bool
        Apply Chan-Vese sub-pixel boundary refinement. Default True.
    run_uncertainty : bool
        Compute GUM uncertainty budget. Default True.
    scale_bar_px : int, optional
        Scale bar length in pixels (for u1 calculation).
        Auto-detected from JEOL metadata if not supplied.
    force_method : str, optional
        Force a specific threshold method, bypassing adaptive scoring.
        Options: "Otsu", "Li", "Yen", "Isodata", "Mean", "Minimum".
        Primarily useful for comparisons and demonstrations.
    no_watershed : bool
        Skip watershed separation; use connected components only.
        Replicates the standard detector software approach.
        Default False.
    material : str, optional
        Material preset key. Sets beam_shrinkage_pct automatically.
        Options: "psl", "polymer", "microplastic", "organic",
        "metal_oxide", "metal", "silica", "inorganic".
        Overridden by explicit beam_shrinkage_pct if both supplied.
        See MATERIAL_PRESETS for values.
    beam_shrinkage_pct : float, optional
        Beam-induced shrinkage correction (%) applied to u7.
        Defaults to 1.0 for polymers/PSL, 0.0 for inorganics.
        Use material= for convenience, or supply directly.
    verbose : bool
        Print progress and diagnostic tables.

    Returns
    -------
    PipelineResults
        All analysis outputs in a single container.

    Examples
    --------
    Basic usage with auto-calibration:

    >>> from particle_pipeline import run_pipeline
    >>> results = run_pipeline("my_particles.tif", nominal_d_um=0.48)
    >>> results.save_report("output/")
    >>> print(results.summary())

    Manual calibration:

    >>> results = run_pipeline(
    ...     "my_particles.tif",
    ...     nominal_d_um=0.48,
    ...     px_per_um=213.0,
    ... )

    Access per-particle data:

    >>> df = results.df
    >>> df[["diam_nm", "circularity", "Dh_nm", "Dv_nm"]].head()

    Access GUM result:

    >>> print(results.result_string)
    '479.0 ± 24.0 nm (k=2, ≈95 %)'
    """
    # Resolve beam shrinkage from material preset
    if beam_shrinkage_pct is None:
        if material is not None:
            key = material.lower().replace(" ", "_")
            beam_shrinkage_pct = MATERIAL_PRESETS.get(key, 1.0)
            if verbose and key not in MATERIAL_PRESETS:
                print(f"  Warning: unknown material '{material}'; "
                      f"using default beam_shrinkage_pct=1.0")
        else:
            beam_shrinkage_pct = 1.0

    from particle_pipeline.io import load_image, _jeol_calibration
    from particle_pipeline.segmentation import segment
    from particle_pipeline.morphometry import extract_morphometry, compute_psd
    from particle_pipeline.refinement import refine_all
    from particle_pipeline.estimation import (
        estimate_dls, estimate_rps, fit_distributions
    )
    from particle_pipeline.uncertainty import compute_budget

    image_path = Path(image_path)
    if verbose:
        print(f"\n── particle-pipeline ──")
        print(f"  Image: {image_path.name}")
        if material:
            print(f"  Material: {material}  "
                  f"(beam_shrinkage={beam_shrinkage_pct:.1f}%)")

    # 1. Load
    if verbose: print("  [1/6] Loading image...")
    arr, cal = load_image(image_path, px_per_um=px_per_um)
    if verbose:
        print(f"        {arr.shape[1]}×{arr.shape[0]} px  "
              f"calibration={cal:.1f} px/µm")

    # 2. Segment
    if verbose: print("  [2/6] Segmenting (adaptive threshold)...")
    labeled, binary, props, best, all_thresh, diag = segment(
        arr, px_per_um=cal,
        nominal_d_um=nominal_d_um,
        min_d_um=min_d_um,
        max_d_um=max_d_um,
        force_method=force_method,
        no_watershed=no_watershed,
        verbose=verbose,
    )
    if verbose:
        print(f"        Selected: {best.name}  score={best.composite_score:.3f}  "
              f"n={best.n_filtered}")

    # 3. Morphometry
    if verbose: print("  [3/6] Extracting morphometry...")
    df = extract_morphometry(props, px_per_um=cal, nominal_d_um=nominal_d_um)
    psd = compute_psd(df)
    if verbose:
        print(f"        {len(df)} particles  "
              f"D50={psd.get('d50',0)*1000:.1f} nm  "
              f"CV={psd.get('cv_pct',0):.1f}%")

    # 4. Refinement
    if run_refinement and len(df):
        if verbose: print("  [4/6] Chan-Vese sub-pixel refinement...")
        df = refine_all(arr, props, df, px_per_um=cal, verbose=verbose)
    else:
        if verbose: print("  [4/6] Refinement skipped.")

    # 5. DLS / RPS estimation
    if verbose: print("  [5/6] Estimating DLS and RPS sizes...")
    dh_nm, ff0, dls_stats = estimate_dls(df)
    dv_nm, rps_stats       = estimate_rps(df)
    df["Dh_nm"] = dh_nm
    df["Dv_nm"] = dv_nm
    df["ff0"]   = ff0

    fit = fit_distributions(df["diam_nm"].values)
    if verbose:
        print(f"        DLS Z-avg={dls_stats['z_average_nm']:.1f} nm  "
              f"PDI={dls_stats['PDI']:.4f}")
        print(f"        RPS vol-mean={rps_stats['vol_mean_nm']:.1f} nm")
        print(f"        Best fit: {fit['best_fit']}")

    # 6. Uncertainty
    budget = {}
    if run_uncertainty and len(df):
        if verbose: print("  [6/6] GUM uncertainty budget...")
        # Auto-detect scale_bar_px from JEOL metadata
        if scale_bar_px is None:
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists():
                import re
                txt = txt_path.read_text(errors="replace")
                m = re.search(r"\$\$SM_MICRON_BAR\s+(\d+)", txt)
                scale_bar_px = int(m.group(1)) if m else int(round(cal))
            else:
                scale_bar_px = int(round(cal))

        budget = compute_budget(
            image=arr,
            df=df,
            px_per_um=cal,
            scale_bar_px=scale_bar_px,
            beam_shrinkage_pct=beam_shrinkage_pct,
            verbose=verbose,
        )
    else:
        if verbose: print("  [6/6] Uncertainty skipped.")

    if verbose:
        print(f"\n── Complete ──")
        if budget:
            print(f"  Result: {budget.get('result_str','')}")

    return PipelineResults(
        image=arr,
        labeled=labeled,
        binary=binary,
        px_per_um=cal,
        image_path=str(image_path),
        df=df,
        psd_stats=psd,
        dls_stats=dls_stats,
        rps_stats=rps_stats,
        fit_results=fit,
        budget=budget,
        best_threshold=best,
        all_thresholds=all_thresh,
        diagnostics=diag,
        _dh_nm=dh_nm,
        _dv_nm=dv_nm,
    )
