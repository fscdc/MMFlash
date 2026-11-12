from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import (
    MODEL_PROVENANCE_CONTRACT_KEY,
    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
    STEP_OPTIONS_CONTRACT_KEY,
    AlgorithmProviders,
    DraftConfigProvider,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepRuntimeConfig,
)
from specforge.algorithms.contracts import AlgorithmSpec, FeatureMode

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTINS = ("dflash", "domino", "dspark", "eagle3", "mmdomino", "mmflash", "peagle")


class BuiltinProviderContractTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

    def test_every_registration_pairs_contract_and_providers(self):
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertIsInstance(registration.spec, AlgorithmSpec)
                self.assertIsInstance(registration.providers, AlgorithmProviders)
                self.assertEqual(
                    registration.name,
                    registration.providers.algorithm_name,
                )
                contract_keys = {c.key for c in registration.spec.feature_contracts}
                provider_keys = {
                    *(
                        (FeatureMode.OFFLINE, p.modality)
                        for p in registration.providers.offline
                    ),
                    *(
                        (FeatureMode.STREAMING, p.modality)
                        for p in registration.providers.server_streaming
                    ),
                }
                self.assertEqual(contract_keys, provider_keys)

    def test_algorithm_metadata_has_no_factories_or_topology_flags(self):
        field_names = {field.name for field in fields(AlgorithmSpec)}
        self.assertEqual(
            {"name", "draft", "feature_contracts", "capabilities"},
            field_names,
        )
        forbidden = {
            "supports_online",
            "supported_deployments",
            "supported_target_backends",
            "deployment_mode",
            "target_backend",
        }
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertTrue(forbidden.isdisjoint(vars(registration.spec)))
                self.assertEqual(
                    registration.spec.supports_online,
                    bool(registration.providers.server_streaming),
                )

    def test_builtin_online_providers_are_server_streaming_only(self):
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                providers = registration.providers
                self.assertGreaterEqual(len(providers.server_streaming), 1)
                self.assertFalse(hasattr(providers, "colocated"))
                self.assertFalse(hasattr(providers, "target_backend"))
                self.assertFalse(hasattr(providers, "deployment"))

    def test_offline_consumer_provider_does_not_require_a_capture_layout(self):
        provider = OfflineDataProvider(
            modality="custom",
            normalizer_id="custom_v1",
            build_reader=lambda *args, **kwargs: None,
            build_normalizer=lambda *args, **kwargs: None,
            build_collator=lambda *args, **kwargs: None,
        )

        self.assertIsNone(provider.capture_layout)

    def test_draft_resolution_stays_outside_algorithm_spec(self):
        provider_fields = {field.name for field in fields(DraftConfigProvider)}
        self.assertEqual(
            {
                "architecture",
                "target_defaults",
                "expected_auto_map_model",
                "apply_overrides",
            },
            provider_fields,
        )
        forbidden = {
            "config_class",
            "model_class",
            "load_config",
            "resolve_config",
            "resolve_model",
        }
        self.assertTrue(forbidden.isdisjoint(provider_fields))
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertFalse(hasattr(registration.spec, "draft_config_resolver"))

    def test_vlm_is_not_registered_as_a_builtin(self):
        for registration in self.registry:
            modalities = {
                contract.modality for contract in registration.spec.feature_contracts
            }
            self.assertEqual({"text"}, modalities, registration.name)

    def test_missing_checkpoint_policy_requires_a_runtime_fingerprint(self):
        with self.assertRaisesRegex(
            ValueError,
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
        ):
            StepRuntimeConfig(
                options={},
                resume_contract={},
                allowed_missing_checkpoint_keys={"embed_tokens.weight"},
            )

    def test_step_options_are_canonical_resume_metadata(self):
        runtime = StepRuntimeConfig(
            options={"objective_scale": 0.5},
            resume_contract={},
            allowed_missing_checkpoint_keys=frozenset(),
        )
        self.assertEqual(
            runtime.resume_contract[STEP_OPTIONS_CONTRACT_KEY],
            (("objective_scale", 0.5),),
        )

        with self.assertRaisesRegex(ValueError, STEP_OPTIONS_CONTRACT_KEY):
            StepRuntimeConfig(
                options={"objective_scale": 0.5},
                resume_contract={
                    STEP_OPTIONS_CONTRACT_KEY: (("objective_scale", 1.0),),
                },
                allowed_missing_checkpoint_keys=frozenset(),
            )

    def test_streaming_provider_constructs_a_generic_input_adapter(self):
        config = object()
        calls = []

        class GenericInputAdapter:
            def load_input_tools(self, received_config):
                return ("tools", received_config)

            def prepare_prompts(
                self,
                received_config,
                input_tools,
                *,
                draft_config,
            ):
                return [
                    {
                        "config": received_config,
                        "input_tools": input_tools,
                        "draft_config": draft_config,
                    }
                ]

            def build_request_inputs(self, tasks):
                return {"generic_model_inputs": list(tasks)}

        adapter = GenericInputAdapter()

        def build_input_adapter(received_config):
            calls.append(received_config)
            return adapter

        provider = ServerStreamingProvider(
            modality="vision_language",
            capture_method="eagle3",
            target_representation="hidden_state",
            layout=ServerCaptureLayout(
                aux_feature="hidden_state",
                last_hidden_feature="target",
                passthrough=(("position_ids", "position_ids", (3,)),),
            ),
            build_collator=lambda: None,
            build_input_adapter=build_input_adapter,
        )

        self.assertEqual("vision_language", provider.modality)
        self.assertIs(adapter, provider.create_input_adapter(config))
        self.assertEqual([config], calls)
        self.assertEqual(
            {"generic_model_inputs": ["task"]},
            adapter.build_request_inputs(["task"]),
        )

    def test_streaming_provider_rejects_incomplete_input_adapters(self):
        required_methods = (
            "load_input_tools",
            "prepare_prompts",
            "build_request_inputs",
        )
        for invalid_method in required_methods:
            methods = {
                name: (lambda *args, **kwargs: None) for name in required_methods
            }
            methods[invalid_method] = object()
            incomplete = SimpleNamespace(**methods)
            provider = ServerStreamingProvider(
                modality="plugin_modality",
                capture_method="eagle3",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_state",
                    last_hidden_feature="target",
                    passthrough=(),
                ),
                build_collator=lambda: None,
                build_input_adapter=lambda _config: incomplete,
            )

            with (
                self.subTest(method=invalid_method),
                self.assertRaisesRegex(
                    TypeError,
                    f"missing callable methods: \\['{invalid_method}'\\]",
                ),
            ):
                provider.create_input_adapter(object())

if __name__ == "__main__":
    unittest.main()
