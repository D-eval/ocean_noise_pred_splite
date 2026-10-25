from __future__ import annotations

import unittest

import torch

from source.dclde_information_ladder import _time_pool, oracle_conditions


class InformationLadderTests(unittest.TestCase):
    def test_full_complex_stft_reconstructs_background_but_mixture_phase_does_not(self):
        torch.manual_seed(47)
        ocean = torch.randn(2, 9984) * 0.05
        event = torch.zeros_like(ocean)
        event[:, 1700:2700] = torch.sin(torch.arange(1000)[None] * 0.5) * 0.15
        conditions = oracle_conditions(ocean, ocean + event)
        self.assertLess((conditions["full_true"] - ocean).abs().max().item(), 1e-6)
        self.assertGreater((conditions["full_mix"] - ocean).square().mean().item(), 1e-5)
        self.assertGreater((conditions["b30_5s_true"] - ocean).square().mean().item(), 1e-5)

    def test_full_window_pool_includes_last_stft_frame(self):
        power = torch.arange(79, dtype=torch.float32)[None, None, :]
        pooled = _time_pool(power, 5.0)
        torch.testing.assert_close(pooled, torch.full_like(power, power.mean().item()))


if __name__ == "__main__":
    unittest.main()
