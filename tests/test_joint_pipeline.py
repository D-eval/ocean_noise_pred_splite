from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import tempfile
import torch
import unittest

from source.data import SpectrumNormaliser, eligible_origins
from source.dclde import _log_band_features, recording_start
from source.dclde_separation_experiment import InjectedUpcallDataset, render_marine_training_target, render_upcalls
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
from source.reference_free_eval import restoration_metrics
from source.synthesis import BROAD_FAMILIES, MARINE_FAMILIES, draw_injection, make_mixture, mix_with_real_ocean
from source.train import engineering_usability_gate
from source.windowed_audio import (
    BAND_CENTRES,
    WindowForecastDataset,
    _one_second_band_power,
    aggregate_features,
)


class JointPipelineTests(unittest.TestCase):
    def test_synthetic_upcalls_are_localised_and_reproducible(self):
        first, intervals = render_upcalls(16000, 91)
        second, second_intervals = render_upcalls(16000, 91)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(intervals, second_intervals)
        self.assertTrue(intervals)
        self.assertAlmostEqual(float(np.sqrt(np.mean(first * first))), 1.0, places=5)
        occupied = np.zeros(len(first), dtype=bool)
        for onset, offset in intervals:
            occupied[round(onset * 2000) : round(offset * 2000)] = True
        self.assertTrue(np.all(first[~occupied] == 0.0))

    def test_marine_training_targets_cover_whales_vessels_and_lfo(self):
        families = set()
        for seed in range(256):
            waveform, parameters = render_marine_training_target(10000, seed)
            self.assertTrue(np.isfinite(waveform).all())
            self.assertAlmostEqual(float(np.sqrt(np.mean(waveform * waveform))), 1.0, places=4)
            families.add(parameters["family"])
            if parameters["family"].startswith("vessel_"):
                self.assertGreaterEqual(parameters["modulation_hz"], 5.0)
                self.assertLessEqual(parameters["modulation_hz"], 20.0)
        self.assertEqual(families, set(MARINE_FAMILIES))

    def test_vessel_lfo_is_frequency_selective_metadata(self):
        found = None
        for seed in range(500):
            parameters = draw_injection(seed, profile="marine", family="vessel_tonal")
            if parameters.lfo_enabled:
                found = parameters
                break
        self.assertIsNotNone(found)
        self.assertGreaterEqual(found.modulation_hz, 5.0)
        self.assertLessEqual(found.modulation_hz, 20.0)

    def test_dclde_filename_timestamp_uses_documented_est_offset(self):
        stamp = recording_start("NOPP6_EST_20090330_001500_CH10.wav")
        self.assertEqual(stamp.isoformat(), "2009-03-30T05:15:00+00:00")

    def test_reference_free_curve_checks_semantics_and_partial_gt(self):
        background = torch.zeros(1, 16)
        background[:, :4] = 0.2
        event = torch.zeros(1, 16)
        event[:, 4:8] = 1.0
        mixture = background + event
        mask = torch.zeros_like(mixture)
        mask[:, 4:8] = 1.0

        def classifier(waveform):
            return 8.0 * waveform[:, 4:8].abs().mean(dim=1) - 2.0

        metrics, curve, examples = restoration_metrics(
            classifier, mixture, event, mask, return_per_example=True
        )
        self.assertGreater(metrics["removal_logit"], 5.0)
        self.assertGreater(metrics["monotonic_fraction"], 0.99)
        self.assertLess(metrics["leakage_outside_event"], 1e-8)
        self.assertLess(metrics["off_event_relative_error"], 1e-8)
        self.assertLess(metrics["reconstruction_relative_error"], 1e-8)
        self.assertGreater(curve[-1], curve[0])
        self.assertEqual(len(examples["removal_logit"]), 1)

    def test_dclde_causal_features_retain_absolute_level(self):
        waveform = np.random.default_rng(7).normal(size=16000).astype(np.float32)
        quiet = _log_band_features(waveform)
        loud = _log_band_features(2.0 * waveform)
        np.testing.assert_allclose(loud - quiet, 20.0 * np.log10(2.0), atol=1e-4)

    def test_oracle_is_extracted_from_current_clean_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocean.wav"
            audio = np.random.default_rng(9).normal(0.0, 0.01, 20000).astype(np.float32)
            sf.write(path, audio, 2000, subtype="FLOAT")
            frame = pd.DataFrame([
                {"path": str(path), "split": "train", "label": 0, "day": "day-1",
                 "start_seconds": start, "duration_seconds": 5.0, "event_intervals": "[]"}
                for start in (0.0, 5.0)
            ])

            class Store:
                mean = np.zeros(30, dtype=np.float32)
                scale = np.ones(30, dtype=np.float32)
                bands = 30

                def __init__(self):
                    self.frame = frame

                def indices(self, split, label):
                    return np.array([0, 1])

                def normalized_history(self, index):
                    return np.zeros((12, 30), dtype=np.float32)

                def normalized_target(self, index):
                    return _log_band_features(audio[index * 10000:(index + 1) * 10000])[0]

            dataset = InjectedUpcallDataset(Store(), "train", 2, seed=23,
                                            snr_range=(-12.0, -12.0))
            item = dataset[0]
            np.testing.assert_allclose(
                item["target_spectrum"].numpy(),
                _log_band_features(item["ocean"].numpy())[0], atol=1e-5,
            )
            np.testing.assert_allclose(
                item["mixture"].numpy(),
                item["ocean"].numpy() + item["source"].numpy(), atol=1e-6,
            )
            self.assertGreater(float((item["target_spectrum"] - item["shuffled_spectrum"]).abs().mean()), 0.01)

    def test_true_bsr_and_zero_film_start_from_the_same_waveform_mapping(self):
        kwargs = dict(spectrum_dim=30, latent_dim=16, initialization_seed=19,
                      model_dim=32, depth=2, heads=4, dim_head=8,
                      n_fft=256, hop_length=64, bands=20)
        plain = ForecastConditionedSeparator(variant="bsr", **kwargs).eval()
        zero = ForecastConditionedSeparator(variant="none_film", **kwargs).eval()
        exact_zero = ForecastConditionedSeparator(variant="zero_film", **kwargs).eval()
        self.assertIsNotNone(exact_zero.forecaster)
        audio = torch.from_numpy(np.random.default_rng(3).normal(0, .01, 10000).astype(np.float32))[None]
        history = torch.zeros(1, 12, 30)
        with torch.inference_mode():
            np.testing.assert_allclose(
                plain(audio, history)["estimate"].numpy(),
                zero(audio, history)["estimate"].numpy(), atol=1e-6,
            )
            np.testing.assert_allclose(
                zero(audio, history)["estimate"].numpy(),
                exact_zero(audio, history)["estimate"].numpy(), atol=1e-6,
            )

    def test_subhour_band_power_finds_tone_and_aggregates_windows(self):
        rate = 8000
        time = np.arange(rate * 10) / rate
        audio = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
        one_second = _one_second_band_power(audio, rate)
        peak = int(np.argmax(one_second.mean(axis=0)))
        self.assertEqual(float(BAND_CENTRES[peak]), 1000.0)
        five_second = one_second.reshape(2, 5, -1).mean(axis=1)
        thirty_second_input = np.tile(five_second, (3, 1))
        aggregated = aggregate_features(thirty_second_input, 30)
        self.assertEqual(aggregated.shape, (1, len(BAND_CENTRES)))
        self.assertTrue(np.isfinite(aggregated).all())

    def test_subhour_history_is_strictly_causal_for_all_resolutions(self):
        normaliser = SpectrumNormaliser(
            np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        )
        for window_seconds in (5, 30, 60, 600):
            with self.subTest(window_seconds=window_seconds):
                context = 3600 // window_seconds
                values = np.repeat(np.arange(context + 2, dtype=np.float32)[:, None], 3, axis=1)
                dataset = WindowForecastDataset(values, normaliser, window_seconds)
                item = dataset[0]
                self.assertEqual(item["history"].shape, (context, 3))
                self.assertEqual(float(item["history"][-1, 0]), context - 1)
                self.assertEqual(float(item["target_spectrum"][0]), context)

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
        engine.history_steps = 2
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
