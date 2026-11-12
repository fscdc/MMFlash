from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import Enum

from specforge.algorithms import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)


def _offline(modality: str = "text") -> FeatureContract:
    return FeatureContract(
        mode=FeatureMode.OFFLINE,
        modality=modality,
        required_tensors={"input_ids", "hidden_states", "loss_mask"},
        allowed_target_representations={"hidden_state"},
        default_target_representation="hidden_state",
        storage=OfflineStorageContract(
            format="manifest_v1",
            required_tensors={"input_ids", "hidden_states", "loss_mask"},
            normalizer="eagle3_offline_v1",
        ),
    )


def _streaming(modality: str = "text") -> FeatureContract:
    required = {"input_ids", "hidden_state", "target", "loss_mask"}
    if modality == "vision_language":
        required.add("position_ids")
    return FeatureContract(
        mode=FeatureMode.STREAMING,
        modality=modality,
        required_tensors=required,
        allowed_target_representations={"hidden_state"},
        default_target_representation="hidden_state",
    )


def _spec(*contracts: FeatureContract) -> AlgorithmSpec:
    return AlgorithmSpec(
        name="eagle3",
        draft=DraftRequirement(
            compatible_architectures={"LlamaForCausalLMEagle3"},
            default_architecture="LlamaForCausalLMEagle3",
            supported_overrides={"attention_layout", "num_hidden_layers"},
        ),
        feature_contracts=contracts or (_offline(), _streaming()),
        capabilities=AlgorithmCapabilities(
            attention_backends={"sdpa", "flex_attention"},
            supports_compact_teacher=True,
            supports_vocab_mapping=True,
        ),
    )


def _assert_contract_is_pure(value: object) -> None:
    if isinstance(value, type) or callable(value):
        raise AssertionError(f"executable value leaked into contract: {value!r}")
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _assert_contract_is_pure(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_contract_is_pure(getattr(value, field.name))
        return
    raise AssertionError(f"opaque value leaked into contract: {type(value).__name__}")


class AlgorithmContractTest(unittest.TestCase):
    def test_offline_requires_storage_contract(self):
        with self.assertRaisesRegex(ValueError, "require a storage contract"):
            FeatureContract(
                mode="offline",
                modality="text",
                required_tensors={"input_ids"},
            )

    def test_streaming_rejects_offline_storage_contract(self):
        with self.assertRaisesRegex(ValueError, "cannot define storage"):
            FeatureContract(
                mode="streaming",
                modality="text",
                required_tensors={"input_ids"},
                storage=OfflineStorageContract(
                    format="manifest_v1",
                    required_tensors={"input_ids"},
                    normalizer="text_v1",
                ),
            )

if __name__ == "__main__":
    unittest.main()
