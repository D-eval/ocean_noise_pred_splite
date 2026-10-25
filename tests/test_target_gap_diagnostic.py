"""Check that coverage reporting does not reward a larger reference pool."""

import unittest

import numpy as np

from source.diagnose_target_gap import nearest_distance_audit


class TargetGapDiagnosticTest(unittest.TestCase):
    def test_equal_size_audit_uses_one_synthetic_per_real_event(self):
        # The first synthetic draw matches every real event exactly; the
        # second is shifted. Both repeats must appear in the equal-size audit.
        real = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        clean = np.zeros_like(real)
        synthetic = np.asarray([[0.0, 0.0], [10.0, 0.0],
                                [2.0, 0.0], [12.0, 0.0],
                                [4.0, 0.0], [14.0, 0.0]])
        owners = np.repeat(np.arange(3), 2)
        result = nearest_distance_audit(real, synthetic, clean, owners)
        self.assertEqual(result["equal_size_reference_count"], 3)
        self.assertEqual(result["equal_size_repeats"], 2)
        self.assertEqual(result["equal_size_nn_median_across_repeats"][0], 0.0)
        self.assertGreater(result["equal_size_nn_median_across_repeats"][1], 0.0)
        self.assertEqual(result["real_to_synthetic_nn_median"], 0.0)


if __name__ == "__main__":
    unittest.main()
