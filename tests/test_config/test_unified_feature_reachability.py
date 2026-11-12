# coding=utf-8
"""Config-to-assembly reachability for retained training capabilities."""

import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config
from specforge.training.assembly import (
    _configured_logger,
    _dataloader_num_workers,
    _logger,
    _profiling_options,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"

OFFLINE_EAGLE3 = {
    "model": {
        "target_model_path": "target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "mapping.pt",
    },
    "data": {"hidden_states_path": "features"},
}

EXPECTED_TRACKING = {
    "lfm2.5-1.2b-instruct-dflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-lfm2.5-1.2b-instruct-dflash",
        "wandb_name": "lfm2.5-1.2b-instruct-dflash-perfectblend-8layers",
    },
    "longcat-flash-dflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-longcat-flash-dflash",
        "wandb_name": "longcat-flash-dflash-sharegpt",
    },
    "qwen3-4b-dflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-4b-dflash",
        "wandb_name": "qwen3-4b-dflash-perfectblend",
    },
    "qwen3-8b-dflash-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen3-8b-dflash-disagg",
        "wandb_name": "qwen3-8b-dflash-disagg-dp4",
    },
    "qwen3-8b-dflash-1server-dp7-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen3-8b-dflash-disagg",
        "wandb_name": "qwen3-8b-dflash-1srv-dp7",
    },
    "qwen3-8b-dflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-8b-dflash",
        "wandb_name": "qwen3-8b-dflash-perfectblend",
    },
    "qwen3-8b-mmflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-8b-mmflash",
        "wandb_name": "qwen3-8b-mmflash-perfectblend",
    },
    "qwen3-8b-domino-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen3-8b-domino-disagg",
        "wandb_name": "qwen3-8b-domino-disagg-dp4",
    },
    "qwen3-8b-domino-1server-dp7-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen3-8b-domino-disagg",
        "wandb_name": "qwen3-8b-domino-1srv-dp7",
    },
    "qwen3-8b-domino-multiserver-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen3-8b-domino-disagg",
        "wandb_name": "qwen3-8b-domino-2srv-dp2",
    },
    "qwen3-8b-domino-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-8b-domino",
        "wandb_name": "qwen3-8b-domino_sharegpt",
    },
    "qwen3-8b-dpace-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "dpace-qwen3-8b",
        "wandb_name": "qwen3-8b-dpace",
    },
    "qwen3-8b-eagle3-disaggregated.yaml": {"report_to": "tensorboard"},
    "qwen3-8b-peagle-disaggregated.yaml": {"report_to": "wandb"},
    "qwen3-coder-30b-a3b-eagle3-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-coder",
        "wandb_name": "qwen3-coder-30b-eagle3-tp4-opc-regen",
    },
    "qwen3-coder-480b-a35b-eagle3-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "specforge-qwen3-480-coder-fp8",
        "wandb_name": "qwen3-coder-480b-a35b-eagle3-tp8-ep2-opc-regen",
    },
    "qwen3.5-35b-a3b-dflash-online.yaml": {"report_to": "tensorboard"},
    "qwen3.5-35b-a3b-eagle3-online.yaml": {"report_to": "tensorboard"},
    "qwen3.5-4b-dflash-online-npu.yaml": {"report_to": "tensorboard"},
    "qwen3.5-4b-domino-online-npu.yaml": {"report_to": "tensorboard"},
    "qwen3.6-27b-dflash-disaggregated.yaml": {
        "report_to": "wandb",
        "wandb_project": "qwen36-dflash-disagg",
        "wandb_name": "qwen36-27b-dflash-server-capture-dp2",
    },
    "qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen36-dflash-disagg",
        "wandb_name": "qwen36-27b-dflash-1srv-dp2",
    },
    "qwen3.6-27b-dflash-multiserver-disaggregated.yaml": {
        "report_to": "none",
        "wandb_project": "qwen36-dflash-disagg",
        "wandb_name": "qwen36-27b-dflash-2srv-dp2",
    },
    "qwen3.6-27b-dflash-online.yaml": {
        "report_to": "wandb",
        "wandb_project": "qwen36-dflash-pr645",
        "wandb_name": "qwen36-27b-dflash-nemotron-6ep",
    },
}

LEGACY_DIST_TIMEOUT_OVERRIDES = {
    "deepseek-v3-671b-eagle3-online.yaml": 60,
    "gpt-oss-120b-eagle3-online.yaml": 60,
    "gpt-oss-20b-eagle3-online.yaml": 60,
    "ling-flash-2.0-eagle3-online.yaml": 60,
    "qwen3-8b-peagle-disaggregated.yaml": 120,
    "qwen3-coder-30b-a3b-eagle3-online.yaml": 60,
}

LEGACY_EPOCH_OVERRIDES = {
    "qwen2.5-0.5b-dflash-online.yaml": 10,
    "qwen3.5-35b-a3b-dflash-online.yaml": 10,
    "qwen3.5-4b-dflash-online-npu.yaml": 10,
    "qwen3.5-4b-domino-online-npu.yaml": 10,
}


class UnifiedFeatureReachabilityTest(unittest.TestCase):
    def test_tracking_config_reaches_the_existing_tracker_adapter(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "tracking": {
                    "report_to": "wandb",
                    "wandb_project": "project",
                    "wandb_name": "experiment",
                    "wandb_offline": True,
                    "wandb_dir": "/tmp/wandb",
                },
                "run_id": "run",
                "output_dir": "/tmp/output",
            }
        )
        tracker_logger = object()

        with mock.patch(
            "specforge.training.tracking.create_tracker_logger",
            return_value=tracker_logger,
        ) as create:
            result = _configured_logger(cfg)

        self.assertIs(result, tracker_logger)
        args, output_dir = create.call_args.args
        self.assertEqual(args.report_to, "wandb")
        self.assertEqual(args.wandb_project, "project")
        self.assertEqual(args.wandb_name, "experiment")
        self.assertTrue(args.wandb_offline)
        self.assertEqual(args.wandb_dir, "/tmp/wandb")
        self.assertEqual(output_dir, "/tmp/output")
        self.assertIs(create.call_args.kwargs["console_logger"], _logger)

    def test_tracking_backend_is_strictly_typed(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {**OFFLINE_EAGLE3, "tracking": {"report_to": "unknown"}}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
