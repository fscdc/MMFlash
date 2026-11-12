from __future__ import annotations

import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.offline_capture import OfflineSGLangCapture


class OfflineCaptureLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = builtin_algorithm_registry()

    def test_duplicate_output_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate.*hidden_states"):
            OfflineCaptureLayout(
                capture_method="dflash",
                aux_feature="hidden_states",
                last_hidden_feature=None,
                passthrough=(("hidden_states", "input_ids"),),
            )

    def test_missing_mapped_source_reports_source_and_output_names(self):
        layout = OfflineCaptureLayout(
            capture_method="dflash",
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=(
                ("input_ids", "input_ids"),
                ("loss_mask", "loss_mask"),
            ),
        )
        sources = {
            "input_ids": torch.tensor([1]),
            "loss_mask": torch.tensor([1]),
            "aux_hidden_states": torch.zeros(1, 1, 1),
        }

        with self.assertRaises(KeyError) as raised:
            layout.materialize(sources)

        message = str(raised.exception)
        self.assertIn("last_hidden_states", message)
        self.assertIn("target_last_hidden_states", message)

    def test_local_capture_forwards_the_algorithm_capture_method(self):
        backend = mock.Mock()
        capture = OfflineSGLangCapture(backend)

        capture.set_capture_layers(
            [1, 9, 17, 25, 33],
            capture_method="dflash",
        )

        self.assertEqual("dflash", capture.capture_method)
        backend.set_capture_layers.assert_called_once_with(
            [1, 9, 17, 25, 33],
            capture_method="dflash",
        )


if __name__ == "__main__":
    unittest.main()
