# coding=utf-8
"""E gate: draft-architecture registry drives the auto loaders.

Adding a draft architecture means adding a class with ``@register_draft``; both
auto loaders resolve it from that registry.
"""

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import LlamaConfig, Qwen3Config

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft import (
    DRAFT_REGISTRY,
    DFlashDraftModel,
    available_drafts,
    register_draft,
    resolve_draft,
)

TINY_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
}

def _write(cfg: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


class DraftRegistryTest(unittest.TestCase):
    def test_register_requires_config_class_and_honors_name(self):
        class NoConfig:
            pass

        with self.assertRaises(TypeError):
            register_draft(NoConfig)

        @register_draft(name="tiny_fake_draft")
        class FakeDraft:
            config_class = LlamaConfig

        self.addCleanup(DRAFT_REGISTRY.pop, "tiny_fake_draft", None)
        self.assertIs(resolve_draft("tiny_fake_draft"), FakeDraft)


class AutoLoaderRegistryTest(unittest.TestCase):
    def test_from_file_resolves_dflash_via_registry(self):
        # Not in the legacy _config_mapping: only the registry makes this load.
        path = _write(TINY_DFLASH)
        self.addCleanup(os.unlink, path)
        config = AutoDraftModelConfig.from_file(path)
        self.assertIsInstance(config, Qwen3Config)

if __name__ == "__main__":
    unittest.main(verbosity=2)
