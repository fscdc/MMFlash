import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scripts.regenerate_train_data import call_sglang
from scripts.regenerate_train_data import main as regenerate_main
from scripts.regenerate_train_data import set_skipped, validate_regen_input
from scripts.validate_regenerated_data import validate_dataset, validate_row


def make_row(row_id="row-1", content="answer"):
    return {
        "id": row_id,
        "status": "success",
        "conversations": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content},
        ],
    }


class TestValidateRegeneratedData(TestCase):
    def test_valid_non_reasoning_dataset_allows_duplicate_ids_with_warning(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                "".join(json.dumps(make_row("same")) + "\n" for _ in range(2)),
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = validate_dataset(
                    path,
                    expect_non_reasoning=True,
                    strict_think_markers=True,
                )

            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.assistant_messages, 2)
            self.assertEqual(summary.duplicate_rows, 1)
            self.assertEqual(len(caught), 1)
            self.assertIn("duplicates are allowed", str(caught[0].message))

    def test_rejects_incomplete_training_conversation(self):
        row = make_row()
        row["conversations"].append({"role": "user", "content": "follow-up"})

        with self.assertRaisesRegex(ValueError, "must end with an assistant"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=True,
            )

    def test_rejects_nonempty_reasoning_content(self):
        row = make_row()
        row["conversations"][-1]["reasoning_content"] = "hidden reasoning"

        with self.assertRaisesRegex(ValueError, "reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=False,
            )

    def test_reasoning_mode_requires_reasoning_content(self):
        row = make_row()

        with self.assertRaisesRegex(ValueError, "empty reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            )

        row["conversations"][-1]["reasoning_content"] = "structured reasoning"
        self.assertEqual(
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            ),
            1,
        )

    def test_strict_mode_rejects_thinking_markers_in_reasoning(self):
        row = make_row()
        row["conversations"][-1]["reasoning_content"] = "<think>hidden</think>"

        with self.assertRaisesRegex(ValueError, "reasoning_content"):
            validate_row(
                row,
                expect_non_reasoning=False,
                expect_reasoning=True,
                strict_think_markers=True,
            )

    def test_strict_mode_rejects_thinking_markers(self):
        row = make_row(content="<THINK>hidden</THINK> visible")

        with self.assertRaisesRegex(ValueError, "thinking marker"):
            validate_row(
                row,
                expect_non_reasoning=True,
                strict_think_markers=True,
            )

    def test_rejects_non_success_rows(self):
        row = make_row()
        row["status"] = "error"

        with self.assertRaisesRegex(ValueError, "status must be 'success'"):
            validate_row(
                row,
                expect_non_reasoning=False,
                strict_think_markers=False,
            )


class TestRegenerationGuards(TestCase):
    def test_missing_openai_client_has_actionable_error(self):
        with (
            patch("scripts.regenerate_train_data.OpenAI", None),
            self.assertRaisesRegex(ModuleNotFoundError, "specforge\\[data\\]"),
        ):
            call_sglang(
                SimpleNamespace(),
                "localhost:30000",
                {"conversations": [{"role": "user", "content": "question"}]},
            )

    def test_input_precheck_rejects_bad_role_order_and_content(self):
        bad_order = {
            "conversations": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]
        }
        empty_content = {"conversations": [{"role": "user", "content": "  "}]}

        self.assertIn("role order", validate_regen_input(bad_order))
        self.assertIn("non-empty string", validate_regen_input(empty_content))

    def test_non_object_input_can_be_recorded_as_skipped(self):
        skipped = set_skipped([], "Expected a JSON object")

        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["data"], [])

    def test_disable_reasoning_skips_residual_think_output(self):
        args = SimpleNamespace(
            model="Qwen/Qwen3-8B",
            max_tokens=32,
            temperature=0,
            top_p=None,
            repetition_penalty=None,
            top_k=None,
            reasoning="disable",
            is_gpt_oss=False,
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<think>hidden</think>answer")
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )

        with patch("scripts.regenerate_train_data.OpenAI", return_value=client):
            result = call_sglang(
                args,
                "localhost:30000",
                {"conversations": [{"role": "user", "content": "question"}]},
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("thinking marker", result["error"])

    def test_save_reasoning_sets_contract_and_strips_history_reasoning(self):
        args = SimpleNamespace(
            model="Qwen/Qwen3.6-27B",
            max_tokens=32,
            temperature=0,
            top_p=None,
            repetition_penalty=None,
            top_k=None,
            reasoning="save",
            is_gpt_oss=False,
        )
        calls = []
        responses = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="first answer",
                                reasoning_content="first reasoning",
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="second answer",
                                reasoning_content="second reasoning",
                            )
                        )
                    ]
                ),
            ]
        )

        def create(**kwargs):
            calls.append(kwargs)
            return next(responses)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        data = {
            "conversations": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "old first answer"},
                {"role": "user", "content": "second question"},
            ]
        }

        with patch("scripts.regenerate_train_data.OpenAI", return_value=client):
            result = call_sglang(args, "localhost:30000", data)

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            True,
        )
        history_assistant = calls[1]["messages"][1]
        self.assertEqual(history_assistant["content"], "first answer")
        self.assertNotIn("reasoning_content", history_assistant)
        self.assertEqual(
            result["conversations"][-1]["reasoning_content"], "second reasoning"
        )

    def test_save_reasoning_rejects_missing_reasoning(self):
        args = SimpleNamespace(
            model="Qwen/Qwen3.6-27B",
            max_tokens=32,
            temperature=0,
            top_p=None,
            repetition_penalty=None,
            top_k=None,
            reasoning="save",
            is_gpt_oss=False,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )

        with patch("scripts.regenerate_train_data.OpenAI", return_value=client):
            result = call_sglang(
                args,
                "localhost:30000",
                {"conversations": [{"role": "user", "content": "question"}]},
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("reasoning_content", result["error"])

    def test_num_samples_limits_submitted_jobs_under_concurrency(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            rows = [[]] + [
                {
                    "id": str(index),
                    "conversations": [{"role": "user", "content": f"question {index}"}],
                }
                for index in range(4)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            submitted = 0

            def fake_call(args, server_address, data, max_tokens=None):
                nonlocal submitted
                if max_tokens is not None:
                    return {"status": "success", "conversations": []}
                submitted += 1
                return {
                    **data,
                    "status": "success",
                    "conversations": [
                        *data["conversations"],
                        {"role": "assistant", "content": "answer"},
                    ],
                }

            argv = [
                "regenerate_train_data.py",
                "--model",
                "Qwen/Qwen3-8B",
                "--server-address",
                "localhost:30000",
                "--input-file-path",
                str(input_path),
                "--output-file-path",
                str(output_path),
                "--num-samples",
                "2",
                "--concurrency",
                "8",
            ]
            with (
                patch("sys.argv", argv),
                patch("scripts.regenerate_train_data.call_sglang", fake_call),
            ):
                regenerate_main()

            skipped_path = Path(str(output_path).replace(".jsonl", "_skipped.jsonl"))
            self.assertEqual(submitted, 2)
            self.assertEqual(len(output_path.read_text().splitlines()), 2)
            self.assertEqual(len(skipped_path.read_text().splitlines()), 1)

