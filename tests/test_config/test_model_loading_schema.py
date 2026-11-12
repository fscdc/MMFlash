# coding=utf-8
"""Typed contract for draft config resolution and weights-only warm starts."""

import unittest

from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config

ONLINE_DEPLOYMENT = {
    "mode": "disaggregated",
    "disaggregated": {
        "control_dir": "/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
    },
}


def _payload(strategy: str, **model_overrides):
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    if strategy in ("eagle3", "peagle"):
        model["vocab_mapping_path"] = "/mapping.pt"
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"]["max_steps"] = 1
        payload["deployment"] = ONLINE_DEPLOYMENT
    return payload


class ModelLoadingSchemaTest(unittest.TestCase):
    def test_warm_start_and_full_resume_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            Config.model_validate(
                {
                    **_payload("eagle3", draft_checkpoint_path="/base/checkpoint"),
                    "training": {
                        "strategy": "eagle3",
                        "resume_from": "/run/checkpoint",
                    },
                }
            )

    def test_disaggregated_producer_can_resolve_warm_start_metadata(self):
        payload = _payload(
            "dflash",
            draft_checkpoint_path="org/dflash-draft",
        )
        payload["model"]["target_backend"] = "sglang"
        payload["data"] = {"train_data_path": "/train.jsonl"}
        payload["training"] = {
            "strategy": "dflash",
            "role": "producer",
            "total_steps": 1,
        }
        payload["deployment"] = ONLINE_DEPLOYMENT
        cfg = Config.model_validate(payload)
        resolve_run(cfg)
        self.assertEqual(cfg.model.draft_checkpoint_path, "org/dflash-draft")


if __name__ == "__main__":
    unittest.main(verbosity=2)
