from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import unittest

from source.data import eligible_origins
from source.inference import SeparationEngine
from source.model import ForecastConditionedSeparator
from source.synthesis import make_mixture
from source.train import engineering_usability_gate


class JointPipelineTests(unittest.TestCase):
    def test_origins_do_not_cross_time_gap(self):
        index = pd.to_datetime(
            ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 04:00"]
        )
        frame = pd.DataFrame(np.ones((4, 2)), index=index, columns=["TOL_25", "TOL_31.5"])
        self.assertEqual(eligible_origins(frame, 2).tolist(), [2])

    def test_waveform_mixture_is_deterministic_and_additive(self):
        levels = np.linspace(90, 60, 8)
        bands = np.geomspace(25, 4000, 8)
        first = make_mixture(levels, bands, 8000, 1024, 7)
        second = make_mixture(levels, bands, 8000, 1024, 7)
        self.assertTrue(np.allclose(first[0], first[1] + first[2]))
        self.assertTrue(np.array_equal(first[0], second[0]))

    def test_requested_signal_strength_is_exact(self):
        levels = np.linspace(90, 60, 8)
        bands = np.geomspace(25, 4000, 8)
        mixture, source, ocean, params = make_mixture(
            levels, bands, 8000, 4096, 11, snr_range=(-12.0, -12.0)
        )
        measured = 20 * np.log10(
            np.sqrt(np.mean(source * source)) / np.sqrt(np.mean(ocean * ocean))
        )
        self.assertAlmostEqual(params["snr_db"], -12.0, places=6)
        self.assertAlmostEqual(float(measured), -12.0, places=3)

    def test_condition_path_receives_gradient(self):
        torch.manual_seed(3)
        model = ForecastConditionedSeparator(
            spectrum_dim=6,
            variant="spectrum",
            latent_dim=4,
            model_dim=8,
            depth=1,
            heads=1,
            dim_head=8,
            n_fft=32,
            hop_length=8,
            bands=4,
        )
        mixture = torch.randn(2, 256)
        history = torch.randn(2, 5, 6)
        output = model(mixture, history)["estimate"]
        output.square().mean().backward()
        gradient = model.separator.injector.projection[1].weight.grad
        self.assertEqual(output.shape, mixture.shape)
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_separator_may_return_shorter_istft_without_shape_error(self):
        torch.manual_seed(4)
        model = ForecastConditionedSeparator(
            spectrum_dim=6, variant="none", model_dim=8, depth=1,
            heads=1, dim_head=8, n_fft=32, hop_length=8, bands=4,
        )
        mixture = torch.randn(1, 253)
        history = torch.randn(1, 5, 6)
        estimate = model(mixture, history)["estimate"]
        self.assertLessEqual(estimate.shape[-1], mixture.shape[-1])

    def test_ablation_separator_core_has_matched_initialisation(self):
        common = dict(
            spectrum_dim=6, latent_dim=4, initialization_seed=99,
            model_dim=8, depth=1, heads=1, dim_head=8,
            n_fft=32, hop_length=8, bands=4,
        )
        baseline = ForecastConditionedSeparator(variant="none", **common)
        conditioned = ForecastConditionedSeparator(variant="latent", **common)
        baseline_weight = next(baseline.separator.separator.layers.parameters())
        conditioned_weight = next(conditioned.separator.separator.layers.parameters())
        self.assertTrue(torch.equal(baseline_weight, conditioned_weight))

    def test_engineering_gate_uses_gain_for_weak_and_quality_for_strong(self):
        test = {"source_si_sdri_db": 5.7, "real_time_factor": 0.003}
        sweep = {
            "-12_dB": {"source_si_sdri_db": 6.5},
            "-6_dB": {"source_si_sdri_db": 6.3},
            "+0_dB": {"source_si_sdri_db": 5.7},
            "+6_dB": {"source_si_sdr_db": 9.8},
            "+12_dB": {"source_si_sdr_db": 11.9},
        }
        self.assertTrue(engineering_usability_gate(test, sweep)["passed"])

    def test_frontend_accepts_native_tol_header_spacing(self):
        engine = SeparationEngine.__new__(SeparationEngine)
        engine.context_hours = 2
        engine.columns = ["TOL_25", "TOL_31.5"]
        engine.mean = np.array([100.0, 90.0], dtype=np.float32)
        engine.scale = np.array([10.0, 5.0], dtype=np.float32)
        history = engine.parse_history_csv(
            "yyyy-mm-ddTHH:MM:SSZ, TOL_25, TOL_31.5\n"
            "2020-01-01T00:00:00Z, 110, 95\n"
            "2020-01-01T01:00:00Z, 90, 85\n"
        )
        np.testing.assert_allclose(history, [[1.0, 1.0], [-1.0, -1.0]])


if __name__ == "__main__":
    unittest.main()
