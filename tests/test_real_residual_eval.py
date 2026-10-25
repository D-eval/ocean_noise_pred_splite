"""Causality audit for the 128-band real-event evaluation cohort."""
import unittest

import numpy as np

from source.dclde import (CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE,
                          EVENT_MASKED_FEATURE_CACHE, MANIFEST)
from source.dclde_causal_highres_experiment import HighResolutionStore
from source.dclde_real_residual_eval import RealResidualEventDataset
from source.dclde_separation_experiment import CausalFeatureStore


class RealResidualEventTests(unittest.TestCase):
    def test_same_103_calls_have_only_clean_strictly_past_maps(self):
        events = CausalFeatureStore(MANIFEST, EVENT_MASKED_FEATURE_CACHE)
        clean = CausalFeatureStore(CLEAN_BACKGROUND_MANIFEST, CLEAN_OCEAN_FEATURE_CACHE)
        dataset = RealResidualEventDataset(events, HighResolutionStore(clean))
        self.assertEqual(len(dataset), 103)
        self.assertEqual((dataset.imputed_examples, dataset.imputed_history_windows), (6, 6))
        for item, event_index in enumerate(dataset.base.rows):
            event = events.frame.iloc[int(event_index)]
            history = clean.frame.iloc[dataset.history_rows[item]]
            self.assertTrue(np.all(history.start_seconds.to_numpy() < event.start_seconds))
            self.assertTrue(np.all(history.label.to_numpy() == 0))
            self.assertTrue(np.all(history.path.to_numpy() == event.path))
            donor_index, donor_day = dataset.shuffled_donors[item]
            donor = clean.frame.iloc[donor_index]
            self.assertNotEqual(donor_day, event.day)
            self.assertEqual(donor.day, donor_day)
            self.assertEqual((donor.split, donor.label), ("test", 0))


if __name__ == "__main__":
    unittest.main()
