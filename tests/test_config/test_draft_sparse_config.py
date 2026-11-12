# coding=utf-8
"""``training.draft_sparse``: opt-in, MMFlash-only, and inert when unset."""

from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from specforge.config import DraftSparseConfig, TrainingConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "train"
SPARSE = {"sink": 4, "text": True, "stride": 32, "window": 2048}


class DraftSparseConfigTest(unittest.TestCase):
    def test_default_is_dense(self):
        self.assertIsNone(TrainingConfig(strategy="mmflash").draft_sparse)
        self.assertIsNone(TrainingConfig().draft_sparse)

    def test_valid_block_and_env_string(self):
        training = TrainingConfig(strategy="mmflash", draft_sparse=SPARSE)
        self.assertIsInstance(training.draft_sparse, DraftSparseConfig)
        self.assertEqual("sink=4,text=1,stride=32,window=2048", training.draft_sparse.to_sglang_env())
        training = TrainingConfig(strategy="mmflash", attention_backend="sdpa", draft_sparse={"window": 8})
        self.assertEqual("sink=0,text=1,stride=0,window=8", training.draft_sparse.to_sglang_env())

    def test_window_is_required(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(strategy="mmflash", draft_sparse={"sink": 4, "text": True, "stride": 32})

    def test_rejects_negative_and_unknown_keys(self):
        for bad in ({**SPARSE, "sink": -1}, {**SPARSE, "window": -1}, {**SPARSE, "stride": -2},
                    {**SPARSE, "depth": 1}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                TrainingConfig(strategy="mmflash", draft_sparse=bad)

    def test_mmflash_only(self):
        for strategy in ("dflash", "eagle3", "domino"):
            with self.assertRaises(ValidationError, msg=strategy):
                TrainingConfig(strategy=strategy, draft_sparse=SPARSE)

    def test_needs_a_masking_backend(self):
        for backend in ("eager", "fa", "usp"):
            with self.assertRaises(ValidationError, msg=backend):
                TrainingConfig(strategy="mmflash", attention_backend=backend, draft_sparse=SPARSE)

    def test_video_recipes_load(self):
        for name in ("qwen3.5-4b-mmflash-video-sparse.yaml", "qwen3.5-9b-mmflash-video-sparse.yaml"):
            with self.subTest(recipe=name):
                config = load_config(str(CONFIGS / name))
                self.assertIsNotNone(config.training.draft_sparse)
                self.assertEqual("mmflash", config.training.strategy)
                # stage 2 continues the stage-1 draft
                self.assertIsNotNone(config.model.draft_checkpoint_path)
                self.assertIsNone(config.training.resume_from)
                # every 48-frame video row (42.4k-45.0k tokens) fits
                self.assertGreaterEqual(config.data.max_length, 45056)
