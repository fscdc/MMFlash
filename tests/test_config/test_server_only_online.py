from __future__ import annotations

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def _online_payload(
    *,
    backend: str = "sglang",
    modality: str = "text",
    topology: str = "disaggregated",
) -> dict:
    payload = {
        "model": {
            "target_model_path": "target",
            "draft_model_config": "draft.json",
            "vocab_mapping_path": "mapping.pt",
            "target_backend": backend,
            "input_modality": modality,
        },
        "data": {"train_data_path": "train.jsonl"},
        "training": {"strategy": "eagle3", "max_steps": 1},
        "deployment": {"mode": topology},
    }
    if topology == "disaggregated":
        payload["deployment"]["disaggregated"] = {
            "control_dir": "outputs/test/control",
            "backend": "mooncake",
            "server_urls": ["http://127.0.0.1:30000"],
            "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
            "mooncake_master_server_addr": "127.0.0.1:35551",
        }
    return payload


class ServerOnlyOnlineConfigTest(unittest.TestCase):
    def test_colocated_online_is_rejected_at_the_config_boundary(self):
        with self.assertRaisesRegex(ValidationError, "colocated online training"):
            Config.model_validate(_online_payload(topology="local_colocated"))

    def test_online_requires_the_sglang_server_backend(self):
        # The retired in-process backends are no longer schema values at all,
        # so the rejection is the enum itself rather than the online validator.
        for backend in ("hf", "custom"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValidationError, "target_backend"),
            ):
                Config.model_validate(_online_payload(backend=backend))

    def test_offline_configs_reject_retired_backends_instead_of_ignoring_them(self):
        # Offline consumers never instantiate a target inference backend, but a
        # config naming a retired backend must fail at load rather than be
        # silently ignored — training.md promises retired options stay loud.
        def _offline_payload(backend):
            return {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                    "vocab_mapping_path": "mapping.pt",
                    "target_backend": backend,
                },
                "data": {"hidden_states_path": "features"},
            }

        config = Config.model_validate(_offline_payload("sglang"))
        self.assertEqual(config.mode, "offline")
        for backend in ("hf", "custom"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValidationError, "target_backend"),
            ):
                Config.model_validate(_offline_payload(backend))

if __name__ == "__main__":
    unittest.main(verbosity=2)
