# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.training.model_card_lint import lint_model_card, lint_model_card_path


COMPLETE_CARD = """---
base_model: unsloth/Llama-3.2-3B-Instruct
license: apache-2.0
datasets:
- tatsu-lab/alpaca
language:
- en
---

# Fine-tune

## Evaluation

| Benchmark | Score |
| --- | --- |
| MMLU | 62.4 |
| GSM8K | 48.1 |
"""


class ModelCardLintTests(unittest.TestCase):
    def test_complete_card_reports_all_four_checks_with_evidence(self):
        report = lint_model_card(COMPLETE_CARD)
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing"], [])
        self.assertTrue(all(check["status"] == "present" for check in report["checks"]))
        self.assertIn("unsloth/Llama-3.2-3B-Instruct", report["checks"][0]["evidence"])

    def test_empty_card_reports_each_missing_check_without_guessing(self):
        report = lint_model_card("# Fine-tune\n\nA short description.\n")
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["missing"], ["base_model", "license", "dataset", "evaluation"]
        )
        self.assertTrue(all(check["evidence"] is None for check in report["checks"]))

    def test_body_fields_are_accepted_when_front_matter_is_absent(self):
        report = lint_model_card(
            "# Model\n\n"
            "Base model: org/base\n"
            "License: mit\n"
            "Dataset: my cleaned rows\n"
            "## Results\n"
            "Accuracy: 81.2%\n"
        )
        self.assertTrue(report["ok"])

    def test_headings_supply_base_model_and_license_evidence(self):
        report = lint_model_card(
            "# Model\n\n"
            "## Base model\n\n"
            "org/base-model\n\n"
            "## License\n\n"
            "apache-2.0\n\n"
            "## Dataset\n\n"
            "org/training-rows\n\n"
            "## Evaluation\n\n"
            "MMLU 62.4\n"
        )
        self.assertTrue(report["ok"])
        self.assertIn("org/base-model", report["checks"][0]["evidence"])
        self.assertIn("apache-2.0", report["checks"][1]["evidence"])

    def test_heading_counts_only_when_it_has_readable_content(self):
        report = lint_model_card("# Model\n\n## Dataset\n\n## Evaluation\n")
        self.assertEqual(report["missing"], ["base_model", "license", "dataset", "evaluation"])

    def test_numbers_without_an_evaluation_metric_are_not_evidence(self):
        report = lint_model_card(
            "# Model\n\n## Results\n\nThe model has 3 billion parameters and 32 layers.\n"
        )
        self.assertIn("evaluation", report["missing"])

    def test_model_index_values_count_as_evaluation_evidence(self):
        report = lint_model_card(
            "---\n"
            "base_model: org/base\n"
            "license: apache-2.0\n"
            "datasets:\n"
            "- org/data\n"
            "model-index:\n"
            "- name: run\n"
            "  results:\n"
            "  - metrics:\n"
            "    - name: Accuracy\n"
            "      value: 0.81\n"
            "---\n"
            "# Model\n"
        )
        self.assertTrue(report["ok"])

    def test_missing_card_is_reported_honestly(self):
        with TemporaryDirectory() as directory:
            report = lint_model_card_path(Path(directory))
        self.assertFalse(report["card_found"])
        self.assertIn("No README", report["reason"])
        self.assertTrue(all(check["evidence"] is None for check in report["checks"]))

    def test_readme_on_disk_is_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(COMPLETE_CARD, encoding="utf-8")
            report = lint_model_card_path(root)
        self.assertTrue(report["card_found"])
        self.assertTrue(report["ok"])
        self.assertTrue(report["source_path"].endswith("README.md"))


if __name__ == "__main__":
    unittest.main()
