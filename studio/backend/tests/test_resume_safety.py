# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.training import dataset_fingerprint, resume_safety


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(content, encoding = "utf-8")


def _run_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "train.jsonl"
    _write(dataset, '{"text": "hello"}\n')
    model = root / "model"
    _write(model / "tokenizer.json", '{"vocab": 1}')
    _write(model / "tokenizer_config.json", "{}")
    output = root / "output"
    checkpoint = output / "checkpoint-10"
    _write(checkpoint / "trainer_state.json", '{"global_step": 10}')
    _write(checkpoint / "adapter_model.safetensors", "weights")
    _write(checkpoint / "optimizer.pt", "optimizer")
    _write(checkpoint / "scheduler.pt", "scheduler")
    _write(checkpoint / "tokenizer.json", '{"vocab": 1}')
    return dataset, model, output


class ResumeSafetyTests(unittest.TestCase):
    def test_matching_checkpoint_passes_all_four_checks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, output = _run_fixture(root)
            stored = {
                "dataset_fingerprint": dataset_fingerprint.fingerprint_source(
                    str(dataset)
                ),
                "tokenizer_fingerprint": dataset_fingerprint.fingerprint_tokenizer(
                    str(model)
                ),
            }
            report = resume_safety.evaluate_resume(
                str(output),
                stored,
                dataset_fingerprint.fingerprint_source(str(dataset)),
                dataset_fingerprint.fingerprint_tokenizer(str(model)),
            )
            self.assertTrue(report["ok"], report["blockers"])
            self.assertEqual(report["dataset"]["status"], "match")
            self.assertEqual(report["tokenizer"]["status"], "match")
            self.assertTrue(report["checks"]["checkpoint"]["adapter"])
            self.assertTrue(report["checks"]["checkpoint"]["optimizer"])

    def test_changed_dataset_blocks_with_a_named_reason(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, output = _run_fixture(root)
            stored = {
                "dataset_fingerprint": dataset_fingerprint.fingerprint_source(
                    str(dataset)
                ),
                "tokenizer_fingerprint": dataset_fingerprint.fingerprint_tokenizer(
                    str(model)
                ),
            }
            _write(dataset, '{"text": "changed"}\n')
            report = resume_safety.evaluate_resume(
                str(output),
                stored,
                dataset_fingerprint.fingerprint_source(str(dataset)),
                dataset_fingerprint.fingerprint_tokenizer(str(model)),
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["blockers"][0]["code"], "dataset_changed")
            self.assertIn("dataset content changed", report["blockers"][0]["message"])

    def test_changed_tokenizer_blocks_with_the_file_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, output = _run_fixture(root)
            stored = {
                "dataset_fingerprint": dataset_fingerprint.fingerprint_source(
                    str(dataset)
                ),
                "tokenizer_fingerprint": dataset_fingerprint.fingerprint_tokenizer(
                    str(model)
                ),
            }
            _write(output / "checkpoint-10" / "tokenizer.json", '{"vocab": 2}')
            report = resume_safety.evaluate_resume(
                str(output),
                stored,
                dataset_fingerprint.fingerprint_source(str(dataset)),
                dataset_fingerprint.fingerprint_tokenizer(str(model)),
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["blockers"][0]["code"], "tokenizer_changed")
            self.assertIn("tokenizer.json", report["blockers"][0]["message"])

    def test_missing_optimizer_names_the_missing_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, output = _run_fixture(root)
            (output / "checkpoint-10" / "optimizer.pt").unlink()
            report = resume_safety.evaluate_resume(
                str(output),
                {},
                dataset_fingerprint.fingerprint_source(str(dataset)),
                dataset_fingerprint.fingerprint_tokenizer(str(model)),
            )
            self.assertFalse(report["ok"])
            codes = {blocker["code"] for blocker in report["blockers"]}
            self.assertIn("optimizer_state_missing", codes)

    def test_legacy_run_without_fingerprints_warns_instead_of_blocking(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, output = _run_fixture(root)
            report = resume_safety.evaluate_resume(
                str(output),
                {},
                dataset_fingerprint.fingerprint_source(str(dataset)),
                dataset_fingerprint.fingerprint_tokenizer(str(model)),
            )
            self.assertTrue(report["ok"], report["blockers"])
            codes = {warning["code"] for warning in report["warnings"]}
            self.assertIn("dataset_unavailable", codes)
            self.assertIn("tokenizer_unavailable", codes)
            self.assertIn("predates", report["dataset"]["message"])


if __name__ == "__main__":
    unittest.main()
