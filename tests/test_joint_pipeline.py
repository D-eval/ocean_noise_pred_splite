from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import unittest

from source.data import eligible_origins
from source.model import ForecastConditionedSeparator
from source.synthesis import make_mixture


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


if __name__ == "__main__":
    unittest.main()
