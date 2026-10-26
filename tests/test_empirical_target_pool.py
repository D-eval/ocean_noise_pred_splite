"""Acoustic extraction checks independent of the held-out event classifier."""

import unittest

import numpy as np

from source.empirical_target_pool import (SAMPLE_RATE, SAMPLES,
                                          extract_pseudo_and_profile)


class EmpiricalTargetPoolTest(unittest.TestCase):
    def test_soft_mask_keeps_annotated_upcall_and_suppresses_outside_audio(self):
        rng = np.random.default_rng(11)
        time = np.arange(SAMPLES) / SAMPLE_RATE
        background = .01*rng.standard_normal(SAMPLES)
        event = np.zeros(SAMPLES)
        local = time-1.0
        active = (local >= 0) & (local <= 2.0)
        event[active] = (.08*np.sin(2*np.pi*(80*local[active] +
                                            25*local[active]**2)) *
                         np.sin(np.pi*local[active]/2))
        pseudo, profile = extract_pseudo_and_profile(
            (background+event).astype(np.float32),
            background.astype(np.float32),
            background.astype(np.float32), [[1.0, 3.0]])
        inside = pseudo[SAMPLE_RATE:3*SAMPLE_RATE]
        self.assertGreater(np.sum(inside**2)/np.sum(pseudo**2), .98)
        self.assertLess(profile["f_min_hz"], profile["f_max_hz"])
        self.assertGreater(profile["duration_seconds"], 1.0)
        self.assertEqual(profile["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
