# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.training import dataset_fingerprint


class DatasetFingerprintTests(unittest.TestCase):
    def test_file_fingerprint_is_stable_and_tracks_content(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.jsonl"
            dataset.write_text('{"text": "one"}\n', encoding = "utf-8")
            first = dataset_fingerprint.fingerprint_source(str(dataset))
            second = dataset_fingerprint.fingerprint_source(str(dataset))
            self.assertEqual(first["status"], "hashed")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["files"], 1)
            dataset.write_text('{"text": "two"}\n', encoding = "utf-8")
            changed = dataset_fingerprint.fingerprint_source(str(dataset))
            self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_directory_fingerprint_includes_relative_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            (root / "a.jsonl").write_text("a", encoding = "utf-8")
            report = dataset_fingerprint.fingerprint_source(str(root))
            self.assertEqual(report["status"], "hashed")
            self.assertEqual(report["files"], 1)
            (root / "b.jsonl").write_text("b", encoding = "utf-8")
            self.assertNotEqual(
                report["sha256"],
                dataset_fingerprint.fingerprint_source(str(root))["sha256"],
            )

    def test_streaming_and_missing_paths_report_why_not_a_hash(self):
        streaming = dataset_fingerprint.fingerprint_source(
            "org/dataset", streaming = True
        )
        self.assertEqual(streaming["status"], "unavailable")
        self.assertIsNone(streaming["sha256"])
        self.assertIn("Streaming", streaming["reason"])
        missing = dataset_fingerprint.fingerprint_source("/does/not/exist.jsonl")
        self.assertEqual(missing["status"], "unavailable")
        self.assertIsNone(missing["sha256"])

    def test_config_reads_local_dataset_before_hub_reference(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.jsonl"
            dataset.write_text("x", encoding = "utf-8")
            report = dataset_fingerprint.fingerprint_config(
                {
                    "local_datasets": [str(dataset)],
                    "hf_dataset": "org/other",
                }
            )
            self.assertEqual(report["status"], "hashed")
            self.assertEqual(report["source"], str(dataset))


if __name__ == "__main__":
    unittest.main()
