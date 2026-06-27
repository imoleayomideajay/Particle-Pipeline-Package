# particle-pipeline

**Automated SEM image analysis with physics-based DLS/RPS size estimation
and GUM-compliant measurement uncertainty quantification.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What it does

Takes a single SEM image and produces:

1. **Particle size distribution** — number-weighted, from automated
   image segmentation with adaptive threshold selection
2. **Estimated DLS hydrodynamic diameter** — using the Perrin shape
   correction and intensity-weighted population statistics (ISO 22412)
3. **Estimated RPS volume-equivalent diameter** — from prolate-spheroid
   volume model and volume-weighted statistics
4. **GUM uncertainty budget** — seven-component expanded uncertainty
   $U = k \cdot u_c$ ($k=2$, ≈95 %) with full component attribution

## Quick start

```bash
git clone https://github.com/imoleayomideajay/Particle-Pipeline-Package.git
cd Particle-Pipeline-Package
pip install -e .
```

```python
from particle_pipeline import run_pipeline

results = run_pipeline("my_particles.tif", nominal_d_um=0.48)
results.save_report("output/")
print(results.summary())
```

Or from the command line:

```bash
particle-pipeline my_particles.tif --nominal 0.48 --output results/
```

## Why the three techniques disagree

SEM, DLS, and RPS measure different physical quantities:

| Technique | Quantity measured | Weighting |
|-----------|-------------------|-----------|
| SEM | Projected area-equivalent diameter | Number |
| DLS | Hydrodynamic diameter (Stokes-Einstein) | Intensity (∝ d⁶) |
| RPS | Volume-equivalent sphere diameter | Volume (∝ d³) |

For a polydisperse sample, the systematic ordering is:

$$D_{50}^{\text{SEM}} \leq D_{50}^{\text{RPS}} \leq D_{50}^{\text{DLS}}$$

This pipeline converts SEM morphometry to physics-justified DLS and RPS
estimates so you can compare techniques without being confused by
measurement artefacts.

## Key features

- **Adaptive threshold selection** — scores six global methods (Otsu,
  Li, Yen, Isodata, Mean, Minimum) on circularity, count stability,
  foreground fraction, and size accuracy; selects automatically
- **Watershed separation** — splits touching/overlapping particles
  using Euclidean distance transform seeds
- **Perrin shape correction** — per-particle DLS estimation accounting
  for non-spherical diffusion
- **Chan-Vese refinement** — sub-pixel boundary accuracy via active
  contour (optional)
- **GUM uncertainty** — calibration, discretisation, threshold
  sensitivity, bootstrap, sample heterogeneity, PSF boundary,
  beam shrinkage
- **Auto-calibration** — extracts pixels/µm from JEOL, Zeiss, and
  FEI/Thermo Fisher metadata
- **PDF + CSV reports** — publication-ready figures and per-particle data

## Calibration support

| Instrument | Metadata field | Auto-detected |
|------------|----------------|---------------|
| JEOL | `$$SM_MICRON_BAR` in companion .txt | ✓ |
| Zeiss | `XResolution` TIFF tag | ✓ |
| FEI/Thermo | `PixelWidth` in SFEG block | ✓ |
| Any | `px_per_um` parameter | manual |

## Installation

```bash
git clone https://github.com/imoleayomideajay/Particle-Pipeline-Package.git
cd Particle-Pipeline-Package
pip install -e ".[dev]"
```

**Citable archived release:** [10.5281/zenodo.20952534](https://doi.org/10.5281/zenodo.20952534)

## Running tests

```bash
pytest tests/ -v
```

## API reference

### `run_pipeline(image_path, ...)`

```python
results = run_pipeline(
    image_path    = "particles.tif",   # SEM TIFF
    nominal_d_um  = 0.48,              # expected diameter µm (optional)
    px_per_um     = None,              # auto-extracted from metadata
    min_d_um      = None,              # size filter lower bound µm
    max_d_um      = None,              # size filter upper bound µm
    run_refinement = True,             # Chan-Vese sub-pixel refinement
    run_uncertainty = True,            # GUM uncertainty budget
    beam_shrinkage_pct = 1.0,          # set 0 for beam-stable materials
    verbose       = True,
)
```

### `PipelineResults`

```python
results.n                      # number of particles retained
results.d50_nm                 # number-weighted D50 (nm)
results.expanded_uncertainty_nm # GUM U (nm), k=2
results.result_string          # "479.0 ± 24.0 nm (k=2, ≈95 %)"
results.df                     # per-particle DataFrame
results.dls_stats              # DLS population statistics dict
results.rps_stats              # RPS population statistics dict
results.budget                 # GUM budget dict
results.summary()              # text summary
results.save_report("output/") # PDF + CSV + figures + JSON
```

## Citation

If you use this pipeline in your research, please cite the paper:

> Ajayi I. (2025). From SEM Image to Particle Size Report: A Practical
> Primer on Automated Segmentation, Cross-Technique Size Estimation, and
> Metrologically Traceable Uncertainty Quantification.
> *Journal of Microscopy* (Primers in Microscopy special issue).

and/or the software directly:

> Ajayi I. (2025). particle-pipeline (v1.2.0) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.20952534

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

## Licence

MIT © Imoleayomide Ajayi, Loughborough University

## Multi-material usage

### Microplastics (PE/PP fragments)

```python
results = run_pipeline(
    "microplastic_sem.tif",
    material="microplastic",   # sets beam_shrinkage_pct=1.0 automatically
    nominal_d_um=50.0,         # expected fragment size range
)

# Shape classification is especially useful for microplastics
print(results.shape_summary())
#     shape_class  count   pct  mean_diam_nm  mean_AR  mean_circularity
# Irregular/Agglomerate   12  40.0        2150.0     2.41             0.512
#              Fibre        8  26.7        3200.0     4.12             0.198
#       Elongated/Needle    6  20.0        1800.0     2.81             0.421
#          Sub-spherical    4  13.3         980.0     1.21             0.782
```

### TiO₂ nanomaterials (metal oxide, beam-stable)

```python
results = run_pipeline(
    "tio2_sem.tif",
    material="metal_oxide",    # sets beam_shrinkage_pct=0.0
    nominal_d_um=0.2,          # aggregate-level nominal
)
```

### Silver nanoparticles

```python
results = run_pipeline(
    "agnp_sem.tif",
    material="metal",          # beam-stable: shrinkage=0.0
    nominal_d_um=0.020,        # 20 nm AgNPs
)
```

## Material presets

| `material=` | `beam_shrinkage_pct` | Suitable for |
|---|---|---|
| `"psl"` | 1.0 | Polystyrene latex standards |
| `"polymer"` / `"microplastic"` | 1.0 | PE, PP, PMMA, organic polymers |
| `"organic"` | 0.5 | Biological, soft matter |
| `"metal_oxide"` | 0.0 | TiO₂, CeO₂, ZnO, Fe₂O₃ |
| `"metal"` | 0.0 | Au, Ag, Pt nanoparticles |
| `"silica"` / `"inorganic"` | 0.0 | SiO₂, ceramics |

## Shape classes

| `shape_class` | Criteria | Typical material |
|---|---|---|
| `"Fibre"` | AR ≥ 3.0 and circularity < 0.30 | Microplastic fibres, needle crystals |
| `"Elongated/Needle"` | AR ≥ 2.5 | Rods, elongated drug crystals |
| `"Spherical"` | circularity ≥ 0.85 | PSL, near-spherical NPs |
| `"Sub-spherical"` | all other | Rounded but irregular |
| `"Irregular/Agglomerate"` | solidity < 0.80 | Aggregates, touching particles |
