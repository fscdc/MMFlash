# coding=utf-8
"""Draft config sources, target-derived defaults, and weights-only loading."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.model_loading import (
    load_draft_config_source,
    resolve_draft_config,
    warm_start_draft_model,
)


def _run_config(strategy: str, **model_overrides) -> Config:
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    if strategy == "eagle3":
        model["vocab_mapping_path"] = "/mapping.pt"
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"].update(
            {
                "attention_backend": "flex_attention",
                "batch_size": 1,
                "max_steps": 1,
            }
        )
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "/control",
                "backend": "mooncake",
                "server_urls": ["http://capture:30000"],
            },
        }
    return Config.model_validate(payload)


def _draft_config_provider(strategy: str):
    registration = builtin_algorithm_registry().resolve(strategy)
    return registration.providers.model.draft_config


def _target_config(*, layers: int = 12):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )


def _draft_payload(architecture: str, *, layers: int = 1, block_size=None):
    payload = {
        "architectures": [architecture],
        "model_type": "qwen3" if architecture == "DFlashDraftModel" else "llama",
        "vocab_size": 128,
        "draft_vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    if block_size is not None:
        payload.update(
            {
                "block_size": block_size,
                "num_target_layers": 12,
                "dflash_config": {},
            }
        )
    return payload


class DraftConfigResolutionTest(unittest.TestCase):
    def test_dflash_typed_overrides_rebuild_capture_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=5, block_size=16)
            payload["dflash_config"] = {"target_layer_ids": [1, 3, 5, 7, 9]}
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
                draft_block_size=8,
            )
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("dflash"),
            )
        self.assertEqual(resolved.num_hidden_layers, 2)
        self.assertEqual(resolved.block_size, 8)
        self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 2)

class _TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.proj = torch.nn.Linear(3, 2)


class WarmStartTest(unittest.TestCase):
    def _write_runtime_state(self, directory, state, *, strategy="dflash"):
        path = os.path.join(directory, "training_state.pt")
        torch.save(
            {
                "draft_state_dict": state,
                "strategy": strategy,
                # Warm start must ignore every field below.
                "global_step": 91,
                "epoch": 7,
                "backend": {
                    "optimizer": {"state": {1: {"step": torch.tensor(91)}}},
                    "rng": torch.tensor([123]),
                },
            },
            path,
        )
        return path

    def test_specforge_checkpoint_loads_only_draft_weights(self):
        torch.manual_seed(1)
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, source.state_dict())
            with mock.patch("torch.load", wraps=torch.load) as load:
                report = warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )
        self.assertEqual(report.checkpoint_format, "specforge")
        self.assertEqual(report.loaded_keys, len(source.state_dict()))
        self.assertTrue(report.loaded_embedding)
        self.assertTrue(
            all(
                torch.equal(destination.state_dict()[key], value)
                for key, value in source.state_dict().items()
            )
        )
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_eagle_checkpoint_may_omit_target_copied_embedding(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        original_embedding = destination.embed_tokens.weight.detach().clone()
        state = {
            key: value
            for key, value in source.state_dict().items()
            if "embed" not in key
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state, strategy="eagle3")
            report = warm_start_draft_model(
                destination,
                path,
                draft_config=object(),
                strategy="eagle3",
                allow_missing_embedding=True,
            )
        self.assertFalse(report.loaded_embedding)
        self.assertIn("embed_tokens.weight", report.missing_keys)
        self.assertTrue(
            torch.equal(destination.embed_tokens.weight, original_embedding)
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_missing_non_embedding_weights_fail_closed(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        state = {"embed_tokens.weight": source.embed_tokens.weight.detach().clone()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state)
            with self.assertRaisesRegex(ValueError, "missing draft weights"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )

    def test_runtime_checkpoint_strategy_must_match(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(
                directory, source.state_dict(), strategy="peagle"
            )
            with self.assertRaisesRegex(ValueError, "written by strategy"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="eagle3",
                )

    def test_pretrained_repo_uses_registered_hf_loader(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with mock.patch(
            "specforge.modeling.auto.AutoDraftModel.from_pretrained",
            return_value=(source, {"missing_keys": []}),
        ) as load:
            report = warm_start_draft_model(
                destination,
                "org/base-draft",
                draft_config=object(),
                strategy="dflash",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(report.checkpoint_format, "pretrained")
        load.assert_called_once_with(
            "org/base-draft",
            config=mock.ANY,
            cache_dir="/cache",
            trust_remote_code=True,
            output_loading_info=True,
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

if __name__ == "__main__":
    unittest.main(verbosity=2)
