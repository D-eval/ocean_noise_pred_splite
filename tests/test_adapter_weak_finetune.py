import unittest
from pathlib import Path

import torch

from source.adapter_weak_finetune import (freeze_for_adapter_head,
    matched_clean_weak_loss, teacher_consistency)
from source.dclde_causal_highres_experiment import CausalResidualBSR


class AdapterWeakTests(unittest.TestCase):
    def test_only_residual_projection_and_mask_heads_update(self):
        model = CausalResidualBSR(30, 20261022, torch.device("cpu"),
                                  Path("models/practical_real.pt"))
        report = freeze_for_adapter_head(model)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertGreater(report["frozen_parameters"], 0)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue("band_split.project" in name or "mask_estimators" in name)

    def test_matched_clean_target_is_not_forced_to_zero(self):
        class Detector(torch.nn.Module):
            def forward(self, wave):
                return wave.reshape(len(wave), 80, 125).mean(-1)

        clean = torch.ones(1, 10000) * 0.2
        event = torch.zeros_like(clean)
        event[:, 2500:5000] = 0.4
        mixture = clean + event
        mask = torch.zeros_like(clean)
        mask[:, 2500:5000] = 1
        loss, parts = matched_clean_weak_loss(Detector(), mixture, event,
                                              clean, mask)
        self.assertAlmostEqual(parts["matched_logit_loss"], 0.0, places=6)
        self.assertTrue(torch.isfinite(loss))
        collapse, collapsed_parts = matched_clean_weak_loss(
            Detector(), mixture, mixture, clean, mask)
        self.assertGreater(collapsed_parts["matched_logit_loss"], 0.0)
        self.assertGreater(float(collapse), float(loss))

    def test_teacher_consistency_zero_for_identical_stems(self):
        mixture = torch.randn(2, 10000)
        stem = torch.randn(2, 9984)
        self.assertAlmostEqual(float(teacher_consistency(mixture, stem, stem)), 0.0)
        self.assertGreater(float(teacher_consistency(mixture, stem, stem + 0.1)), 0.0)
