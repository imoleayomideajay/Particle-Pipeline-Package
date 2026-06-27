"""
particle_pipeline
=================
Automated SEM image analysis with physics-based DLS/RPS size
estimation and GUM-compliant measurement uncertainty quantification.

Quick start
-----------
>>> from particle_pipeline import run_pipeline
>>> results = run_pipeline("my_particles.tif", nominal_d_um=0.48)
>>> results.save_report("output/")

Modules
-------
io          Load and preprocess SEM images; extract calibration
            from instrument metadata (JEOL, Zeiss, FEI/Thermo Fisher)
segmentation Adaptive threshold selection, watershed separation,
            morphological cleanup
morphometry  Per-particle shape descriptors (d, AR, circularity,
            solidity, elongation)
refinement  Chan-Vese sub-pixel boundary refinement
estimation  Perrin-corrected DLS hydrodynamic diameter and
            prolate-spheroid RPS volume-equivalent diameter
uncertainty  GUM seven-component measurement uncertainty budget
reporting   PDF report + CSV + publication-quality figures
pipeline    High-level run_pipeline() API

Reference
---------
Ajayi I. (2025). From SEM Image to Particle Size Report: A Practical
Primer on Automated Segmentation, Cross-Technique Size Estimation, and
Metrologically Traceable Uncertainty Quantification. Journal of
Microscopy (Primers in Microscopy special issue).
"""

from particle_pipeline.pipeline import run_pipeline, PipelineResults, MATERIAL_PRESETS
from particle_pipeline._version import __version__

__all__ = ["run_pipeline", "PipelineResults", "MATERIAL_PRESETS", "__version__"]
