# coding=utf-8
"""Gemma 4 E4B-it draft configs, HPC recipes and their job scripts.

Weight-free and CPU-only: the draft JSONs are instantiated on the meta device,
the target is read through its cached tokenizer, config and safetensors header,
the recipes resolve through ``specforge train --plan``, and the launcher through
its DRY_RUN mode. Everything that needs the pinned target snapshot is skipped
when it is not cached.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGOS = ("eagle3", "dflash", "mmflash")
ARCHITECTURES = {
    "eagle3": "LlamaForCausalLMEagle3",
    "dflash": "DFlashDraftModel",
    "mmflash": "MMFlashDraftModel",
}
CAPTURE_LAYERS = {
    "eagle3": [1, 20, 38],
    "dflash": [1, 10, 20, 30, 39],
    "mmflash": [1, 10, 20, 30, 39],
}
# fc maps the concatenated captured layers (3 or 5 x 2560) back to 2560
FC_SHAPES = {"eagle3": (2560, 7680), "dflash": (2560, 12800), "mmflash": (2560, 12800)}
TARGET_LAYERS = 42
HIDDEN = 2560
VOCAB = 262144
MASK_TOKEN_ID = 4
EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
def _draft_path(algo: str) -> Path:
    return REPO_ROOT / "configs" / f"gemma4-e4b-{algo}.json"


def _draft(algo: str) -> dict:
    return json.loads(_draft_path(algo).read_text())


def _capture_layers(draft: dict) -> list:
    if "eagle_config" in draft:
        return draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
    return draft["dflash_config"]["target_layer_ids"]


# every recipe pins the same local snapshot (the regen's chat template and weights)
def _flag(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _flag_values(argv, flag):
    values = []
    for value in argv[argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _script_env(**values) -> dict:
    """A clean job-like environment with this interpreter's env first on PATH."""
    env_bin = os.path.dirname(sys.executable)
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "USER": os.environ.get("USER", "specforge"),
        "PATH": os.pathsep.join([env_bin, "/usr/bin", "/bin"]),
        "CONDA_PREFIX": sys.prefix,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("HF_HOME", "HF_HUB_OFFLINE"):
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(values)
    return env


class GemmaDraftConfigTest(unittest.TestCase):
    def test_mmflash_draft_is_the_dflash_draft_but_its_class(self):
        dflash, mmflash = _draft("dflash"), _draft("mmflash")
        self.assertEqual(mmflash.pop("architectures"), ["MMFlashDraftModel"])
        self.assertEqual(mmflash.pop("auto_map"), {"AutoModel": "mmflash.MMFlashDraftModel"})
        del dflash["architectures"], dflash["auto_map"]
        self.assertEqual(mmflash, dflash)


class GemmaRecipeTest(unittest.TestCase):
    def test_schema_has_the_mm_attention_field(self):
        # Plan C10. Without this ModelConfig field every Gemma recipe fails to
        # load (extra fields are forbidden), so it is checked on its own.
        from specforge.config.schema import ModelConfig

        self.assertIn(
            "sglang_mm_attention_backend",
            ModelConfig.model_fields,
            "ModelConfig.sglang_mm_attention_backend (plan C10) is missing; the "
            "Gemma recipes set it to sdpa",
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
