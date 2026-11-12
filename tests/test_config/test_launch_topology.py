from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from specforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"

EXPECTED_NPROC_PER_NODE = {
    "deepseek-v2-lite-eagle3-online.yaml": 8,
    "deepseek-v3-671b-eagle3-offline.yaml": 8,
    "deepseek-v3-671b-eagle3-online.yaml": 8,
    "gemma3-1b-eagle3-online.yaml": 1,
    "glm-5.2-dspark-disaggregated.yaml": 1,
    "gpt-oss-120b-eagle3-online.yaml": 8,
    "gpt-oss-20b-eagle3-online.yaml": 8,
    "lfm2.5-1.2b-instruct-dflash-online.yaml": 8,
    "inkling-dspark-disaggregated.yaml": 1,
    "ling-flash-2.0-eagle3-offline.yaml": 8,
    "ling-flash-2.0-eagle3-online.yaml": 8,
    "llama3.1-8b-eagle3-offline.yaml": 1,
    "llama3.1-8b-eagle3-online.yaml": 1,
    "llama3.3-70b-eagle3-online.yaml": 8,
    "llama4-scout-17b-16e-eagle3-online.yaml": 8,
    "longcat-flash-dflash-online.yaml": 1,
    "longcat-flash-eagle3-online.yaml": 1,
    "phi4-eagle3-online.yaml": 1,
    "qwen2.5-0.5b-dflash-online.yaml": 1,
    "qwen2.5-0.5b-eagle3-online.yaml": 1,
    "qwen2.5-7b-eagle3-offline.yaml": 1,
    "qwen2.5-7b-eagle3-offline-disaggregated.yaml": 1,
    "qwen3-235b-a22b-eagle3-online.yaml": 8,
    "qwen3-30b-a3b-eagle3.1-online.yaml": 4,
    "qwen3-30b-a3b-eagle3-online.yaml": 4,
    "qwen3-32b-eagle3-online.yaml": 4,
    "qwen3-4b-dflash-online.yaml": 8,
    "qwen3-4b-dspark-offline.yaml": 1,
    "qwen3-4b-dspark-disaggregated.yaml": 1,
    "qwen3-4b-eagle3-online.yaml": 1,
    "qwen3-8b-dspark-offline.yaml": 1,
    "qwen3-8b-dflash-disaggregated.yaml": 4,
    "qwen3-8b-dflash-1server-dp7-disaggregated.yaml": 7,
    "qwen3-8b-dflash-offline.yaml": 1,
    "qwen3-8b-mmflash-offline.yaml": 1,
    "qwen3-8b-dflash-online.yaml": 8,
    "qwen3-8b-domino-1server-dp7-disaggregated.yaml": 7,
    "qwen3-8b-domino-disaggregated.yaml": 4,
    "qwen3-8b-domino-multiserver-disaggregated.yaml": 2,
    "qwen3-8b-domino-offline.yaml": 1,
    "qwen3-8b-domino-online.yaml": 8,
    "qwen3-8b-dpace-online.yaml": 8,
    "qwen3-8b-mmflash-online.yaml": 8,
    "qwen3-8b-dspark-disaggregated.yaml": 1,
    "qwen3-8b-eagle3-offline-disaggregated.yaml": 1,
    "qwen3-8b-eagle3-offline.yaml": 1,
    "qwen3-8b-eagle3-disaggregated.yaml": 1,
    "qwen3-8b-peagle-disaggregated.yaml": 1,
    "qwen3-coder-30b-a3b-eagle3-online.yaml": 4,
    "qwen3-coder-480b-a35b-eagle3-offline.yaml": 8,
    "qwen3-coder-480b-a35b-eagle3-online.yaml": 8,
    "qwen3-coder-next-eagle3-online.yaml": 8,
    "qwen3-next-80b-a3b-eagle3-online.yaml": 8,
    "qwen3.5-35b-a3b-dflash-online.yaml": 4,
    "qwen3.5-35b-a3b-eagle3-offline.yaml": 4,
    "qwen3.5-35b-a3b-eagle3-online.yaml": 2,
    "qwen3.5-4b-dflash-online-npu.yaml": 8,
    "qwen3.5-4b-mmflash-online-npu.yaml": 8,
    "qwen3.5-4b-domino-online-npu.yaml": 8,
    "qwen3.6-27b-dflash-disaggregated.yaml": 2,
    "qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml": 2,
    "qwen3.6-27b-dflash-multiserver-disaggregated.yaml": 2,
    "qwen3.6-27b-dflash-online.yaml": 8,
    "qwen3.6-27b-domino-online.yaml": 8,
    "qwq-32b-eagle3-online.yaml": 4,
}

LOCAL_MOONCAKE_ENDPOINTS = {
    "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
    "mooncake_master_server_addr": "127.0.0.1:35551",
    "mooncake_protocol": "tcp",
}

EXPECTED_DISAGGREGATED = {
    "glm-5.2-dspark-disaggregated.yaml": {
        "control_dir": "outputs/glm-5.2-dspark-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "inkling-dspark-disaggregated.yaml": {
        "control_dir": "outputs/inkling-dspark-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen2.5-7b-eagle3-offline-disaggregated.yaml": {
        "control_dir": ("outputs/qwen2.5-7b-eagle3-offline-disaggregated/control"),
        "backend": "shared_dir",
        "store_root": ("outputs/qwen2.5-7b-eagle3-offline-disaggregated/features"),
    },
    "qwen3-4b-dspark-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-4b-dspark-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3-8b-dflash-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-8b-dflash-disaggregated/control",
        "consumer_state_dir": ("outputs/qwen3-8b-dflash-disaggregated/consumer-state"),
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3-8b-dspark-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-8b-dspark-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3-8b-dflash-1server-dp7-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3-8b-dflash-1server-dp7-disaggregated/control"),
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
            ],
            "mooncake": {
                "protocol": "tcp",
                "global_segment_size_bytes": 34359738368,
                "local_buffer_size_bytes": 1073741824,
            },
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0"],
                    "tp_size": 1,
                    "mem_fraction_static": 0.5,
                }
            ],
        },
    },
    "qwen3-8b-domino-1server-dp7-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3-8b-domino-1server-dp7-disaggregated/control"),
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
            ],
            "mooncake": {
                "protocol": "tcp",
                "global_segment_size_bytes": 34359738368,
                "local_buffer_size_bytes": 1073741824,
            },
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0"],
                    "tp_size": 1,
                    "mem_fraction_static": 0.5,
                }
            ],
        },
    },
    "qwen3-8b-domino-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-8b-domino-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3-8b-domino-multiserver-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3-8b-domino-multiserver-disaggregated/control"),
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": ["2", "3"],
            "mooncake": {
                "protocol": "tcp",
                "global_segment_size_bytes": 34359738368,
                "local_buffer_size_bytes": 1073741824,
            },
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0"],
                    "tp_size": 1,
                    "mem_fraction_static": 0.85,
                },
                {
                    "port": 30001,
                    "cuda_visible_devices": ["1"],
                    "tp_size": 1,
                    "mem_fraction_static": 0.85,
                },
            ],
        },
    },
    "qwen3-8b-eagle3-offline-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3-8b-eagle3-offline-disaggregated/control"),
        "backend": "shared_dir",
        "store_root": ("outputs/qwen3-8b-eagle3-offline-disaggregated/features"),
    },
    "qwen3-8b-eagle3-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-8b-eagle3-disaggregated/control",
        "consumer_state_dir": ("outputs/qwen3-8b-eagle3-disaggregated/consumer-state"),
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3-8b-peagle-disaggregated.yaml": {
        "control_dir": "outputs/qwen3-8b-peagle-disaggregated/control",
        "consumer_state_dir": ("outputs/qwen3-8b-peagle-disaggregated/consumer-state"),
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3.6-27b-dflash-disaggregated.yaml": {
        "control_dir": "outputs/qwen3.6-27b-dflash-disaggregated/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
        **LOCAL_MOONCAKE_ENDPOINTS,
    },
    "qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3.6-27b-dflash-1server-dp2-disaggregated/control"),
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": ["1", "2"],
            "mooncake": {
                "protocol": "tcp",
                "global_segment_size_bytes": 68719476736,
                "local_buffer_size_bytes": 1073741824,
            },
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0"],
                    "tp_size": 1,
                    "mem_fraction_static": 0.85,
                }
            ],
        },
    },
    "qwen3.6-27b-dflash-multiserver-disaggregated.yaml": {
        "control_dir": ("outputs/qwen3.6-27b-dflash-multiserver-disaggregated/control"),
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": ["4", "5"],
            "mooncake": {
                "protocol": "tcp",
                "global_segment_size_bytes": 51539607552,
                "local_buffer_size_bytes": 1073741824,
            },
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0", "1"],
                    "tp_size": 2,
                    "mem_fraction_static": 0.85,
                },
                {
                    "port": 30001,
                    "cuda_visible_devices": ["2", "3"],
                    "tp_size": 2,
                    "mem_fraction_static": 0.85,
                },
            ],
        },
    },
}


def _recipes() -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(EXAMPLE_CONFIG_DIR.glob("*.yaml"))
        if not path.name.startswith(".")
    }


class ExampleLaunchTopologyTest(unittest.TestCase):
    def test_golden_topologies_validate_for_their_declared_world_size(self):
        for filename, path in _recipes().items():
            with self.subTest(config=filename):
                config = Config.from_file(str(path))
                topology = config.deployment.trainer
                expected_nproc = EXPECTED_NPROC_PER_NODE[filename]
                self.assertEqual(topology.nnodes, 1)
                self.assertEqual(topology.nproc_per_node, expected_nproc)
                config.validate_world_size(topology.nnodes * expected_nproc)

    def test_every_recipe_keeps_trainer_tp_at_one(self):
        recipes = _recipes()
        for filename, path in recipes.items():
            with self.subTest(config=filename):
                config = Config.from_file(str(path))
                self.assertEqual(config.training.tp_size, 1)
                if config.deployment.mode != "disaggregated":
                    continue
                self.assertEqual(config.training.role, "auto")
                self.assertEqual(config.training.sp_ulysses_size, 1)
                self.assertEqual(config.training.sp_ring_size, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
