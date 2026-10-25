from __future__ import annotations

import unittest

import torch
from torch import nn

from source.dclde_oracle_residual_experiment import ResidualBandSplit


class OracleResidualTests(unittest.TestCase):
    def test_zero_initial_adapter_is_exact_bsr_and_receives_gradient(self):
        torch.manual_seed(73)
        wrapper = ResidualBandSplit(nn.Identity(), token_bands=20, token_dim=32)
        base_tokens = torch.randn(2, 157, 20, 32)
        wrapper.context = torch.randn(2, 79, 256)
        output = wrapper(base_tokens)
        torch.testing.assert_close(output, base_tokens, rtol=0, atol=0)
        output.square().mean().backward()
        self.assertGreater(wrapper.project.weight.grad.abs().sum().item(), 0)
        self.assertEqual(wrapper.project.bias, None)


if __name__ == "__main__":
    unittest.main()
