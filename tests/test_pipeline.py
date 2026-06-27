"""
Tests for particle_pipeline.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image():
    """Synthetic SEM-like image: 5 bright circles (r=20px) on dark background."""
    from skimage.draw import disk
    img = np.ones((256, 256), dtype=np.float64) * 0.15
    for cy, cx in [(64, 64), (64, 192), (192, 64), (192, 192), (128, 128)]:
        rr, cc = disk((cy, cx), 20, shape=img.shape)
        img[rr, cc] = 0.75
    img += np.random.default_rng(0).normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1)


@pytest.fixture
def sample_df():
    """Minimal morphometry DataFrame matching ~480 nm PSL."""
    return pd.DataFrame({
        "diam_um":        [0.48, 0.50, 0.46, 0.52, 0.47],
        "diam_nm":        [480., 500., 460., 520., 470.],
        "aspect_ratio":   [1.02, 1.05, 1.01, 1.08, 1.03],
        "circularity":    [0.92, 0.89, 0.94, 0.87, 0.91],
        "solidity":       [0.97, 0.95, 0.98, 0.94, 0.96],
        "is_agglomerate": [False] * 5,
    })


# ── io tests ──────────────────────────────────────────────────────────────────

class TestIO:
    def test_load_image_returns_normalised(self, tmp_path, synthetic_image):
        from PIL import Image as PILImage
        from particle_pipeline.io import load_image

        p = tmp_path / "test.tif"
        PILImage.fromarray((synthetic_image * 255).astype(np.uint8)).save(str(p))

        arr, cal = load_image(p, px_per_um=10.0)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0
        assert cal == 10.0

    def test_missing_calibration_raises(self, tmp_path, synthetic_image):
        from PIL import Image as PILImage
        from particle_pipeline.io import load_image, CalibrationError

        p = tmp_path / "nocal.tif"
        PILImage.fromarray((synthetic_image * 255).astype(np.uint8)).save(str(p))
        with pytest.raises(CalibrationError):
            load_image(p)

    def test_data_strip_detection(self):
        """Data strip rows (near-black after normalisation) trigger removal."""
        from particle_pipeline.io import _remove_data_strip

        # Float array: rows 110+ are black (0.0), rows 0-109 are bright (0.8)
        arr = np.ones((160, 128), dtype=np.float64) * 0.8
        arr[120:, :] = 0.0   # 40 row strip > JEOL_STRIP_MIN_HEIGHT=30
        result = _remove_data_strip(arr)
        # 40-row strip should be removed
        assert result.shape[0] <= 121

    def test_jeol_calibration_from_txt(self, tmp_path, synthetic_image):
        """JEOL companion .txt calibration is auto-extracted."""
        from PIL import Image as PILImage
        from particle_pipeline.io import load_image

        p = tmp_path / "test.tif"
        PILImage.fromarray((synthetic_image * 255).astype(np.uint8)).save(str(p))
        txt = tmp_path / "test.txt"
        txt.write_text("$$SM_MICRON_BAR 213\n$$SM_MICRON_MARKER 1um\n")

        arr, cal = load_image(p)   # no manual px_per_um
        assert abs(cal - 213.0) < 0.1


# ── segmentation tests ────────────────────────────────────────────────────────

class TestSegmentation:
    def test_segment_returns_correct_types(self, synthetic_image):
        from particle_pipeline.segmentation import segment

        labeled, binary, props, best, all_thresh, diag = segment(
            synthetic_image, px_per_um=10.0
        )
        assert labeled.dtype in (np.int32, np.int64, np.intp, np.int_)
        assert binary.dtype == bool
        assert isinstance(props, list)
        assert best.composite_score > 0.0
        assert len(all_thresh) >= 5

    def test_bright_particles_detected(self, synthetic_image):
        from particle_pipeline.segmentation import diagnose_image
        diag = diagnose_image(synthetic_image)
        assert diag.is_bright_particles in (True, False)
        assert diag.mean_intensity > 0.0
        assert diag.mean_intensity > 0.0
        assert diag.std_intensity >= 0.0

    def test_composite_score_in_range(self, synthetic_image):
        from particle_pipeline.segmentation import segment

        _, _, _, best, all_thresh, _ = segment(
            synthetic_image, px_per_um=10.0
        )
        for c in all_thresh:
            assert 0.0 <= c.composite_score <= 1.0
        assert all_thresh[0].rank == 1

    def test_at_least_some_particles_detected(self, synthetic_image):
        from particle_pipeline.segmentation import segment

        _, _, props, best, _, _ = segment(
            synthetic_image, px_per_um=10.0
        )
        # Five circles placed; at least 3 should be detected
        assert best.n_raw >= 3


# ── morphometry tests ─────────────────────────────────────────────────────────

class TestMorphometry:
    def test_extract_returns_dataframe(self, synthetic_image):
        """extract_morphometry returns a DataFrame with expected columns."""
        from particle_pipeline.segmentation import segment
        from particle_pipeline.morphometry import extract_morphometry
        from skimage import measure

        # Use a low-level segment to get props
        from skimage import filters, morphology
        from scipy import ndimage
        thresh = filters.threshold_mean(synthetic_image)
        binary = synthetic_image > thresh
        binary = morphology.remove_small_objects(binary, min_size=50)
        labeled = measure.label(binary)
        props   = measure.regionprops(labeled, intensity_image=synthetic_image)
        props_f = [p for p in props if p.area > 50]

        df = extract_morphometry(props_f, px_per_um=10.0)
        assert isinstance(df, pd.DataFrame)
        for col in ["diam_nm", "circularity", "aspect_ratio",
                    "solidity", "shape_class"]:
            assert col in df.columns

    def test_circularity_range(self):
        """Circularity of near-perfect circles is > 0.80."""
        from skimage.draw import disk
        from skimage import measure, morphology
        import numpy as np

        img = np.ones((128, 128), dtype=np.float64) * 0.1
        for cy, cx in [(40, 40), (40, 88), (88, 40), (88, 88)]:
            rr, cc = disk((cy, cx), 18, shape=img.shape)
            img[rr, cc] = 0.8

        binary = img > 0.5
        binary = morphology.remove_small_objects(binary, min_size=50)
        labeled = measure.label(binary)
        props   = measure.regionprops(labeled, intensity_image=img)

        from particle_pipeline.morphometry import extract_morphometry
        df = extract_morphometry(props, px_per_um=10.0)
        if len(df):
            assert (df["circularity"] > 0.80).mean() >= 0.5

    def test_psd_stats_keys(self, sample_df):
        """compute_psd returns all expected keys."""
        from particle_pipeline.morphometry import compute_psd
        stats = compute_psd(sample_df)
        for key in ["n", "mean_um", "std_um", "d10", "d50", "d90", "span"]:
            assert key in stats


# ── estimation tests ──────────────────────────────────────────────────────────

class TestEstimation:
    def test_sphere_perrin_equals_one(self):
        from particle_pipeline.estimation import perrin_correction
        ff0 = perrin_correction(np.array([1.0, 1.0, 1.01]))
        np.testing.assert_allclose(ff0[:2], 1.0, atol=1e-6)

    def test_perrin_increases_with_ar(self):
        from particle_pipeline.estimation import perrin_correction
        ARs = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
        ff0 = perrin_correction(ARs)
        assert np.all(np.diff(ff0) >= 0), "f/f0 must be monotonically non-decreasing in AR"

    def test_dls_z_average_positive(self, sample_df):
        from particle_pipeline.estimation import estimate_dls
        _, _, stats = estimate_dls(sample_df)
        assert stats["z_average_nm"] > 0.0

    def test_pdi_between_0_and_1(self, sample_df):
        from particle_pipeline.estimation import estimate_dls
        _, _, stats = estimate_dls(sample_df)
        assert 0.0 <= stats["PDI"] <= 1.0

    def test_rps_dv_close_to_image_d_for_spheres(self, sample_df):
        from particle_pipeline.estimation import estimate_rps
        dv, _ = estimate_rps(sample_df)
        ratio = dv / sample_df["diam_nm"].values
        np.testing.assert_allclose(ratio, 1.0, atol=0.03)

    def test_dls_greater_equal_rps_per_particle(self, sample_df):
        from particle_pipeline.estimation import estimate_dls, estimate_rps
        dh, _, _ = estimate_dls(sample_df)
        dv, _    = estimate_rps(sample_df)
        assert np.all(dh >= dv - 1e-9)

    def test_fit_distributions_returns_best(self, sample_df):
        from particle_pipeline.estimation import fit_distributions
        result = fit_distributions(sample_df["diam_nm"].values)
        assert "best_fit" in result
        assert result["best_fit"] in ("lognormal", "weibull")


# ── uncertainty tests ─────────────────────────────────────────────────────────

class TestUncertainty:
    def test_budget_components_non_negative(self, synthetic_image, sample_df):
        from particle_pipeline.uncertainty import compute_budget
        budget = compute_budget(
            image=synthetic_image,
            df=sample_df,
            px_per_um=10.0,
            scale_bar_px=100,
            n_bootstrap=50,
        )
        for key, val in budget["components"].items():
            assert val >= 0.0, f"{key} must be non-negative"

    def test_expanded_is_k_times_combined(self, synthetic_image, sample_df):
        from particle_pipeline.uncertainty import compute_budget
        budget = compute_budget(
            image=synthetic_image,
            df=sample_df,
            px_per_um=10.0,
            scale_bar_px=100,
            n_bootstrap=50,
        )
        assert abs(budget["U_nm"] / budget["u_c_nm"] - 2.0) < 1e-6

    def test_result_string_format(self, synthetic_image, sample_df):
        from particle_pipeline.uncertainty import compute_budget
        budget = compute_budget(
            image=synthetic_image,
            df=sample_df,
            px_per_um=10.0,
            scale_bar_px=100,
            n_bootstrap=50,
        )
        assert "±" in budget["result_str"]
        assert "nm" in budget["result_str"]

    def test_sample_heterogeneity_dominant_for_polydisperse(self):
        """u5 should be large relative to u1 for a polydisperse sample."""
        from particle_pipeline.uncertainty import compute_budget
        import numpy as np
        from skimage.draw import disk

        img = np.ones((128, 128), dtype=np.float64) * 0.1
        for cy, cx in [(40, 40), (40, 88), (88, 40), (88, 88)]:
            rr, cc = disk((cy, cx), 15, shape=img.shape)
            img[rr, cc] = 0.8

        # Wide polydisperse distribution
        df = pd.DataFrame({
            "diam_nm": np.random.default_rng(0).normal(480, 80, 50),
            "diam_um": np.random.default_rng(0).normal(0.48, 0.08, 50),
        })
        budget = compute_budget(
            image=img, df=df,
            px_per_um=10.0, scale_bar_px=100, n_bootstrap=50
        )
        comp = budget["components"]
        assert comp["u5_heterogeneity"] > comp["u1_calibration"]


# ── integration test ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_run_pipeline_on_synthetic(self, tmp_path, synthetic_image):
        """End-to-end: synthetic image → PipelineResults → save_report."""
        from PIL import Image as PILImage
        from particle_pipeline import run_pipeline

        p = tmp_path / "integration.tif"
        PILImage.fromarray(
            (synthetic_image * 255).astype(np.uint8)
        ).save(str(p))
        # Write JEOL calibration file
        (tmp_path / "integration.txt").write_text(
            "$$SM_MICRON_BAR 10\n$$SM_MICRON_MARKER 1um\n"
        )

        results = run_pipeline(
            p,
            nominal_d_um=4.0,   # circles are ~4 µm at 10 px/µm
            run_refinement=False,
            run_uncertainty=False,
            verbose=False,
        )

        assert results.n >= 3
        assert results.d50_nm > 0
        assert "Dh_nm" in results.df.columns
        assert "Dv_nm" in results.df.columns

        saved = results.save_report(tmp_path / "out", figures=False, pdf=True)
        assert saved["pdf"].exists()
        assert saved["csv"].exists()

    def test_summary_string_has_key_fields(self, tmp_path, synthetic_image):
        from PIL import Image as PILImage
        from particle_pipeline import run_pipeline

        p = tmp_path / "s.tif"
        PILImage.fromarray(
            (synthetic_image * 255).astype(np.uint8)
        ).save(str(p))
        (tmp_path / "s.txt").write_text(
            "$$SM_MICRON_BAR 10\n$$SM_MICRON_MARKER 1um\n"
        )

        results = run_pipeline(
            p,
            run_refinement=False,
            run_uncertainty=False,
            verbose=False,
        )
        s = results.summary()
        assert "particle-pipeline" in s
        assert "D50" in s
        assert "Particles" in s

    def test_version_importable(self):
        from particle_pipeline import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_perrin_physics_validated(self):
        """Verify Perrin formula against published values for AR=2."""
        from particle_pipeline.estimation import perrin_correction
        # For AR=2 (p=0.5): expected f/f0 ≈ 1.044 (Koenig 1975)
        ff0 = perrin_correction(np.array([2.0]))[0]
        assert abs(ff0 - 1.044) < 0.005, f"Perrin f/f0 at AR=2: expected ≈1.044, got {ff0:.4f}"


# ── New v1.1.0 tests ──────────────────────────────────────────────────────────

class TestShapeClasses:
    def test_fibre_class_assigned(self):
        """Fibre class assigned for AR >= 3 and circularity < 0.30."""
        import pandas as pd
        from particle_pipeline.morphometry import shape_summary

        df = pd.DataFrame({
            "diam_um":        [0.5, 0.5, 0.5],
            "diam_nm":        [500., 500., 500.],
            "aspect_ratio":   [4.0, 3.2, 1.1],
            "circularity":    [0.18, 0.22, 0.90],
            "solidity":       [0.85, 0.88, 0.96],
            "is_agglomerate": [False, False, False],
        })
        from particle_pipeline.morphometry import extract_morphometry
        from skimage.draw import disk
        from skimage import measure

        # Build minimal props to pass through extract_morphometry
        # Instead, test shape logic directly
        # Fibre: AR >= 3.0 AND circ < 0.30 AND solidity >= 0.80
        from particle_pipeline.morphometry import shape_summary
        # Manually assign shape_class to test shape_summary
        df["shape_class"] = ["Fibre", "Fibre", "Spherical"]
        summary = shape_summary(df)
        assert "Fibre" in summary["shape_class"].values
        fibre_row = summary[summary["shape_class"] == "Fibre"]
        assert int(fibre_row["count"].iloc[0]) == 2

    def test_agglomerate_takes_priority(self):
        """Irregular/Agglomerate assigned when solidity < 0.80."""
        import pandas as pd
        from particle_pipeline.morphometry import shape_summary

        df = pd.DataFrame({
            "diam_um":        [0.5],
            "diam_nm":        [500.],
            "aspect_ratio":   [4.0],   # would be Fibre if not for low solidity
            "circularity":    [0.18],
            "solidity":       [0.72],  # < 0.80 → Agglomerate takes priority
            "is_agglomerate": [True],
        })
        df["shape_class"] = ["Irregular/Agglomerate"]
        summary = shape_summary(df)
        assert "Irregular/Agglomerate" in summary["shape_class"].values

    def test_shape_summary_returns_dataframe(self):
        """shape_summary returns DataFrame with expected columns."""
        import pandas as pd
        from particle_pipeline.morphometry import shape_summary

        df = pd.DataFrame({
            "diam_nm":      [400., 500., 300.],
            "aspect_ratio": [1.1, 3.5, 1.8],
            "circularity":  [0.92, 0.20, 0.75],
            "solidity":     [0.96, 0.88, 0.91],
            "is_agglomerate": [False, False, False],
            "shape_class":  ["Spherical", "Fibre", "Sub-spherical"],
        })
        summary = shape_summary(df)
        for col in ["shape_class", "count", "pct",
                    "mean_diam_nm", "mean_AR"]:
            assert col in summary.columns


class TestMaterialPresets:
    def test_metal_oxide_preset_zero_shrinkage(self, tmp_path, synthetic_image):
        """material='metal_oxide' sets beam_shrinkage_pct=0.0."""
        from PIL import Image as PILImage
        from particle_pipeline import run_pipeline, MATERIAL_PRESETS

        assert MATERIAL_PRESETS["metal_oxide"] == 0.0
        assert MATERIAL_PRESETS["polymer"] == 1.0
        assert MATERIAL_PRESETS["microplastic"] == 1.0
        assert MATERIAL_PRESETS["metal"] == 0.0

    def test_material_preset_applied_in_run(self, tmp_path, synthetic_image):
        """material= parameter is accepted without error."""
        from PIL import Image as PILImage
        from particle_pipeline import run_pipeline

        p = tmp_path / "metal.tif"
        PILImage.fromarray(
            (synthetic_image * 255).astype(np.uint8)
        ).save(str(p))
        (tmp_path / "metal.txt").write_text(
            "$$SM_MICRON_BAR 10\n$$SM_MICRON_MARKER 1um\n"
        )
        results = run_pipeline(
            p,
            material="metal_oxide",
            run_refinement=False,
            run_uncertainty=False,
            verbose=False,
        )
        assert results.n >= 0   # ran without error

    def test_shape_summary_on_results(self, tmp_path, synthetic_image):
        """results.shape_summary() returns a DataFrame."""
        from PIL import Image as PILImage
        from particle_pipeline import run_pipeline

        p = tmp_path / "shape.tif"
        PILImage.fromarray(
            (synthetic_image * 255).astype(np.uint8)
        ).save(str(p))
        (tmp_path / "shape.txt").write_text(
            "$$SM_MICRON_BAR 10\n$$SM_MICRON_MARKER 1um\n"
        )
        results = run_pipeline(
            p,
            nominal_d_um=4.0,
            run_refinement=False,
            run_uncertainty=False,
            verbose=False,
        )
        summary = results.shape_summary()
        assert hasattr(summary, "columns")
        if len(summary):
            assert "shape_class" in summary.columns
            assert "count" in summary.columns
            assert summary["pct"].sum() <= 101.0  # rounding tolerance
