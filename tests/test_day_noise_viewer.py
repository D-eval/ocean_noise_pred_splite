import unittest

import numpy as np

from source.day_noise_viewer import _power_spectrogram, overview


class DayNoiseViewerTests(unittest.TestCase):
    def test_power_spectrogram_finds_tone_band(self):
        sample_rate = 8000
        samples = np.arange(4096) / sample_rate
        windows = np.stack([
            np.sin(2 * np.pi * 500 * samples),
            np.sin(2 * np.pi * 500 * samples),
        ]).astype(np.float32)
        values, edges, resolution = _power_spectrogram(
            windows, sample_rate, 20.0, 2000.0, 99
        )
        peak_band = int(np.argmax(values[0]))
        self.assertLessEqual(edges[peak_band], 500)
        self.assertGreaterEqual(edges[peak_band + 1], 500)
        self.assertEqual(values.shape, (2, 99))
        self.assertAlmostEqual(resolution, sample_rate / 4096)

    def test_overview_is_one_continuous_day(self):
        data = overview()
        self.assertEqual(data["time_edges_seconds"][0], 0)
        self.assertEqual(data["time_edges_seconds"][-1], 86400)
        self.assertEqual(len(data["values"]), 1440)
        self.assertEqual(len(data["frequency_edges_hz"]), 49)


if __name__ == "__main__":
    unittest.main()
