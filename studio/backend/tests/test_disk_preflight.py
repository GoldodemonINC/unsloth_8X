# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import unittest

from core.training import disk_preflight

GB = 1024**3


class DiskPreflightTests(unittest.TestCase):
    def test_small_run_fits_and_reports_every_component(self):
        report = disk_preflight.estimate(
            {
                "params": 1_000_000_000,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_layers": 24,
                "method": "qlora",
                "load_in_4bit": True,
                "max_steps": 1000,
                "save_steps": 250,
                "free_bytes": 100 * GB,
                "model_known_cached": True,
            }
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["checkpoints"]["count"], 4)
        self.assertEqual(report["breakdown"]["base_weights_gb"], 0.0)
        self.assertGreater(report["breakdown"]["checkpoints_gb"], 0)
        self.assertGreater(report["breakdown"]["optimizer_gb"], 0)
        self.assertGreater(report["breakdown"]["final_save_gb"], 0)
        self.assertAlmostEqual(
            report["breakdown"]["total_gb"], report["required_gb"], places = 2
        )

    def test_tiny_disk_is_over_budget_with_a_blocker_and_breakdown(self):
        report = disk_preflight.estimate(
            {
                "params": 8_000_000_000,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_layers": 32,
                "method": "full",
                "load_in_4bit": False,
                "max_steps": 2000,
                "save_steps": 100,
                "free_bytes": 10 * GB,
                "model_known_cached": False,
            }
        )
        self.assertEqual(report["status"], "over")
        self.assertTrue(report["blockers"])
        self.assertEqual(report["checkpoints"]["count"], 20)
        self.assertGreater(report["required_gb"], report["free_gb"])
        self.assertLess(report["headroom_gb"], 0)
        self.assertGreater(len(report["suggestions"]), 0)

    def test_missing_parameter_count_is_unknown_not_a_passing_verdict(self):
        report = disk_preflight.estimate({"params": 0, "free_bytes": 100 * GB})
        self.assertEqual(report["status"], "unknown")
        self.assertIsNone(report["required_gb"])
        self.assertIn("parameter count", report["reason"])

    def test_unreadable_free_space_says_why_it_is_unknown(self):
        report = disk_preflight.estimate(
            {
                "params": 1_000_000_000,
                "method": "lora",
                "model_known_cached": True,
            }
        )
        self.assertEqual(report["status"], "unknown")
        self.assertIn("Free disk space", report["reason"])

    def test_unset_steps_marks_checkpoints_unknown(self):
        report = disk_preflight.estimate(
            {
                "params": 1_000_000_000,
                "method": "lora",
                "max_steps": 0,
                "save_steps": 100,
                "free_bytes": 100 * GB,
                "model_known_cached": True,
            }
        )
        self.assertFalse(report["checkpoints"]["known"])
        self.assertEqual(report["checkpoints"]["count"], 0)
        self.assertTrue(
            any("unknown" in assumption for assumption in report["assumptions"])
        )


if __name__ == "__main__":
    unittest.main()
