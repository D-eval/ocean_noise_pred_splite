from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import tempfile
import torch
import unittest

from source.data import SpectrumNormaliser, eligible_origins
from source.inference import SeparationEngine
from source.metrics import (
    jensen_shannon_bits,
    narrow_source_reference,
    normalised_spectral_entropy,
    spectral_overlap,
    spectral_probability,
)
from source.model import ForecastConditionedSeparator
from source.real_audio_data import RealAudioMixtureDataset
from source.synthesis import BROAD_FAMILIES, make_mixture, mix_with_real_ocean
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

    def test_real_ocean_mixture_preserves_gt_and_requested_snr(self):
        rng = np.random.default_rng(19)
        raw_ocean = rng.standard_normal(4096).astype(np.float32) * 0.03
        mixture, source, ocean, params = mix_with_real_ocean(
            raw_ocean, 8000, 22, snr_range=(-6.0, -6.0)
        )
        self.assertTrue(np.allclose(mixture, source + ocean, atol=1e-7))
        measured = 20 * np.log10(
            np.sqrt(np.mean(source * source)) / np.sqrt(np.mean(ocean * ocean))
        )
        self.assertAlmostEqual(params["snr_db"], -6.0, places=6)
        self.assertAlmostEqual(float(measured), -6.0, places=3)

    def test_real_dataset_history_is_causal_and_shuffle_is_wrong_hour(self):
        index = pd.date_range("2020-01-01", periods=28, freq="h", tz="UTC")
        values = np.stack((np.arange(28), np.arange(28) * 2), axis=1).astype(np.float32)
        frame = pd.DataFrame(values, index=index, columns=["TOL_25", "TOL_31.5"])
        normaliser = SpectrumNormaliser.fit([frame])
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="tmp") as directory:
            records = []
            for origin in (24, 25):
                path = Path(directory) / f"{origin}.flac"
                sf.write(path, np.sin(np.arange(16000) * 0.1).astype(np.float32), 8000)
                records.append(
                    {
                        "deployment": "TEST",
                        "timestamp": index[origin].isoformat(),
                        "local": str(path),
                    }
                )
            dataset = RealAudioMixtureDataset(
                frame, "TEST", normaliser, records, condition="shuffled",
                context_hours=24, clips_per_hour=1, seconds=0.25,
            )
            item = dataset[0]
            expected_history = normaliser.transform(values[:24])
            np.testing.assert_allclose(item["history"].numpy(), expected_history)
            np.testing.assert_allclose(
                item["external_condition"].numpy(), normaliser.transform(values[[25]])[0]
            )
            self.assertFalse(
                np.array_equal(item["external_condition"], item["target_spectrum"])
            )

    def test_broad_source_families_are_finite_and_distinct(self):
        levels = np.linspace(90, 60, 8)
        bands = np.geomspace(25, 4000, 8)
        signatures = []
        for index, family in enumerate(BROAD_FAMILIES):
            _, source, _, params = make_mixture(
                levels, bands, 8000, 2048, 100 + index,
                source_profile="broad", source_family=family,
            )
            self.assertEqual(params["family"], family)
            self.assertTrue(np.isfinite(source).all())
            signatures.append(spectral_probability(source, 8000, bins=16))
        self.assertGreater(np.mean(np.std(signatures, axis=0)), 0.005)

    def test_js_divergence_is_bounded_and_zero_for_reference(self):
        reference = narrow_source_reference(8000, 1024, examples=8, bins=16)
        self.assertAlmostEqual(jensen_shannon_bits(reference, reference), 0.0, places=10)
        shifted = np.roll(reference, 5)
        self.assertGreater(jensen_shannon_bits(shifted, reference), 0.0)
        self.assertLessEqual(jensen_shannon_bits(shifted, reference), 1.0)
        self.assertAlmostEqual(spectral_overlap(reference, reference), 1.0, places=10)
        self.assertGreaterEqual(spectral_overlap(reference, shifted), 0.0)
        self.assertGreaterEqual(normalised_spectral_entropy(reference), 0.0)
        self.assertLessEqual(normalised_spectral_entropy(reference), 1.0)

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

    def test_conditioning_ablation_modes_have_output_and_gradient(self):
        common = dict(
            spectrum_dim=6, latent_dim=4, model_dim=8, depth=1,
            heads=1, dim_head=8, n_fft=32, hop_length=8, bands=4,
        )
        mixture = torch.randn(2, 256)
        history = torch.randn(2, 5, 6)
        for variant in ("concat", "film", "cross_attention"):
            with self.subTest(variant=variant):
                model = ForecastConditionedSeparator(variant=variant, **common)
                estimate = model(mixture, history)["estimate"]
                estimate.square().mean().backward()
                injector_gradients = [
                    parameter.grad for name, parameter in model.separator.injector.named_parameters()
                    if not name.startswith("band_split.") and parameter.grad is not None
                ]
                self.assertEqual(estimate.shape, mixture.shape)
                self.assertTrue(injector_gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in injector_gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in injector_gradients), 0.0)

    def test_oracle_and_shuffled_require_external_condition(self):
        common = dict(
            spectrum_dim=6, latent_dim=4, model_dim=8, depth=1,
            heads=1, dim_head=8, n_fft=32, hop_length=8, bands=4,
        )
        mixture = torch.randn(2, 256)
        history = torch.randn(2, 5, 6)
        condition = torch.randn(2, 6)
        for variant in ("oracle_film", "shuffled_film"):
            model = ForecastConditionedSeparator(variant=variant, **common)
            with self.assertRaises(ValueError):
                model(mixture, history)
            self.assertEqual(model(mixture, history, condition)["estimate"].shape, mixture.shape)

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
