import unittest

import torch

from source.weak_event_detector import FrameEventDetector, frame_targets
from source.weak_real_finetune import weak_loss


class WeakRealTests(unittest.TestCase):
    def test_framewise_annotation_alignment_and_detector_interface(self):
        mask = torch.zeros(1, 10000)
        mask[:, 1280:2560] = 1
        detector = FrameEventDetector()
        logits = detector(torch.randn(1, 10000))
        truth = frame_targets(mask, logits.shape[1])
        self.assertEqual(logits.shape, truth.shape)
        self.assertEqual(logits.shape, (1, 79))
        self.assertTrue(torch.equal(torch.nonzero(truth[0]).flatten(), torch.arange(10, 20)))


    def test_weak_objective_preserves_off_event_audio_and_backpropagates(self):
        class AmplitudeDetector(torch.nn.Module):
            def forward(self, wave):
                return wave.reshape(len(wave), 80, 125).abs().mean(-1)

        detector = AmplitudeDetector()
        mixture = torch.ones(2, 10000)
        mask = torch.zeros_like(mixture)
        mask[:, 2500:5000] = 1
        all_to_event = mixture.clone().requires_grad_()
        collapsed, pieces = weak_loss(detector, mixture, all_to_event, mask)
        self.assertGreater(pieces["off"], 0.99)
        self.assertGreater(collapsed.item(), 1.9)
        collapsed.backward()
        self.assertIsNotNone(all_to_event.grad)
        self.assertTrue(torch.isfinite(all_to_event.grad).all())
        self.assertGreater(float(all_to_event.grad.abs().sum()), 0)
