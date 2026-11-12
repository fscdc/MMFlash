"""The gemma4-spec SGLang patch and every place that checks for it agree.

The patch (patches/sglang/v0.5.14/gemma4-spec.patch) leaves one sentinel line in
each of three sglang files. The training preflight, the patch scripts and the
gate scripts each carry their own copy of those strings, so a reworded patch
line would otherwise make some of them report a patched tree as unpatched (or
the reverse) without any test noticing.
"""

import re
import unittest
from pathlib import Path

from specforge import launch_plan

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "patches/sglang/v0.5.14/gemma4-spec.patch"
EXPECTED = {
    "sglang/" + path: sentinel for path, sentinel in launch_plan._GEMMA4_SGLANG_SENTINELS
}


class Gemma4PatchSentinelTest(unittest.TestCase):
    def test_preflight_names_the_three_patched_files(self):
        self.assertEqual(
            set(EXPECTED),
            {
                "sglang/srt/models/gemma4_causal.py",
                "sglang/srt/models/gemma4_mm.py",
                "sglang/srt/multimodal/processors/gemma4.py",
            },
        )

    def test_each_sentinel_is_added_by_the_patch_and_only_there(self):
        added, other, current = {}, {}, None
        for line in PATCH.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ "):
                current = re.sub(r"^\+\+\+ b/python/", "", line.split("\t")[0])
                added.setdefault(current, [])
                other.setdefault(current, [])
            elif line.startswith("--- ") or current is None:
                continue
            elif line.startswith("+"):
                added[current].append(line)
            else:
                other[current].append(line)
        self.assertEqual(set(added), set(EXPECTED))
        for path, sentinel in EXPECTED.items():
            self.assertTrue(any(sentinel in l for l in added[path]), f"{sentinel!r} not added for {path}")
            self.assertFalse(any(sentinel in l for l in other[path]), f"{sentinel!r} already in stock {path}")

    def test_bench_patch_helper_targets_both_files(self):
        text = (REPO / "scripts/apply_sglang_bench_patches.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"PATCH_NAMES=\([^)]*\bgemma4-spec\b")
        for path in EXPECTED:
            self.assertIn(path, text)


if __name__ == "__main__":
    unittest.main()
