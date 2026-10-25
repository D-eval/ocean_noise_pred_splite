"""Focused checks for the event-masked causal 128-band experiment."""
import numpy as np
import torch
import unittest
from types import SimpleNamespace

from source.dclde_causal_highres_experiment import (
    FRAMES, GROUPS, ConditionedInjectedDataset, HighResGRU, HighResolutionStore, log_band_map,
)
from source.dclde import CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE
from source.dclde_separation_experiment import CausalFeatureStore
from source.dclde_oracle_residual_experiment import OracleResidualBSR


class CausalHighResTests(unittest.TestCase):
    def test_log_map_grid_and_finite_values(self):
        maps = log_band_map(torch.randn(2, 10000))
        self.assertEqual(maps.shape, (2, FRAMES, GROUPS))
        self.assertTrue(torch.isfinite(maps).all())

    def test_map_is_identical_to_oracle_frontend(self):
        mixture = torch.randn(2, 10000)
        ocean = torch.randn(2, 10000)
        reference = OracleResidualBSR.oracle_map(
            SimpleNamespace(oracle_window=torch.hann_window(512)), mixture, ocean)
        expected = torch.cat((log_band_map(ocean),
                              log_band_map(mixture) - log_band_map(ocean)), dim=-1)
        torch.testing.assert_close(reference, expected, rtol=0, atol=0)

    def test_gru_initial_output_is_exact_persistence(self):
        model = HighResGRU(np.zeros(GROUPS, np.float32), np.ones(GROUPS, np.float32))
        history = torch.randn(2, 12, FRAMES, GROUPS)
        torch.testing.assert_close(model(history), history[:, -1], rtol=0, atol=0)

    def test_history_is_clean_and_strictly_prior(self):
        store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
        cache = HighResolutionStore(store)
        indices = np.flatnonzero(store.eligible)
        history = cache.history_indices[indices]
        self.assertEqual(history.shape, (len(indices), 12))
        self.assertTrue(np.all(history >= 0))
        self.assertTrue(np.all(history < indices[:, None]))
        self.assertTrue(np.all(store.frame.iloc[history.ravel()].split.to_numpy().reshape(-1, 12)
                            == store.frame.iloc[indices].split.to_numpy()[:, None]))
        self.assertTrue(np.all(store.frame.iloc[history.ravel()].label.to_numpy() == 0))

    def test_residual_controls_keep_samples_matched_and_shuffle_days(self):
        store = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
        cache = HighResolutionStore(store)
        for split, donor_split in (("train", "train"), ("validation", "train"),
                                   ("test", "test")):
            zero = ConditionedInjectedDataset(cache, split, 2, 20261022, "zero")
            shuffled = ConditionedInjectedDataset(cache, split, 2, 20261022, "shuffled")
            a, b = zero[0], shuffled[0]
            torch.testing.assert_close(a["mixture"], b["mixture"], rtol=0, atol=0)
            torch.testing.assert_close(a["source"], b["source"], rtol=0, atol=0)
            expected = torch.from_numpy(np.broadcast_to(cache.mean, (FRAMES, GROUPS)).copy())
            torch.testing.assert_close(a["condition_map"], expected, rtol=0, atol=0)
            self.assertNotEqual(a["example_id"].split(":", 1)[0], b["condition_day"])
            donor = store.frame.iloc[b["condition_manifest_index"]]
            self.assertEqual(donor["split"], donor_split)
            self.assertEqual(donor["day"], b["condition_day"])
            self.assertEqual(donor["label"], 0)


if __name__ == "__main__":
    unittest.main()
