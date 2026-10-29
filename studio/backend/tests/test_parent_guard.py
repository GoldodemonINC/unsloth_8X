# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import unittest

from core.training.parent_guard import (
    ParentLossGuard,
    ParentLossReference,
    parent_loss_points,
)


def _guard(points, patience = 5, min_delta = 0.0):
    return ParentLossGuard(
        ParentLossReference(
            run_id = "parent",
            label = "baseline",
            points = tuple(points),
            patience_steps = patience,
            min_delta = min_delta,
        )
    )


class ParentLossGuardTests(unittest.TestCase):
    def test_improvements_reset_the_patience_window(self):
        guard = _guard([(10, 1.0), (20, 0.9), (30, 0.8)])
        self.assertIsNone(guard.observe(10, 0.9))
        self.assertIsNone(guard.observe(15, 0.95))
        self.assertIsNone(guard.observe(16, 0.5))
        self.assertIsNone(guard.observe(20, 0.85))
        self.assertFalse(guard.triggered)

    def test_stops_once_the_configured_step_window_elapses(self):
        guard = _guard([(10, 1.0), (20, 0.9)], patience = 5)
        self.assertIsNone(guard.observe(10, 1.2))
        self.assertIsNone(guard.observe(14, 1.1))
        signal = guard.observe(15, 1.05)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["parent_run_id"], "parent")
        self.assertEqual(signal["parent_label"], "baseline")
        self.assertEqual(signal["parent_step"], 10)
        self.assertEqual(signal["parent_loss"], 1.0)
        self.assertEqual(signal["step"], 15)
        self.assertEqual(signal["elapsed_steps"], 5)
        self.assertEqual(guard.observe(16, 0.1), signal)

    def test_parent_reference_advances_with_recorded_steps(self):
        guard = _guard([(10, 1.0), (20, 0.9)], patience = 10)
        guard.observe(9, 0.1)
        self.assertIsNone(guard.last_parent_step)
        guard.observe(19, 0.95)
        self.assertEqual(guard.last_parent_step, 10)
        guard.observe(20, 0.95)
        self.assertEqual(guard.last_parent_step, 20)
        self.assertEqual(guard.last_parent_loss, 0.9)

    def test_min_delta_must_be_beaten(self):
        guard = _guard([(10, 1.0)], patience = 2, min_delta = 0.2)
        self.assertIsNone(guard.observe(10, 0.85))
        signal = guard.observe(12, 0.85)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["min_delta"], 0.2)

    def test_non_finite_child_values_are_ignored(self):
        guard = _guard([(10, 1.0)], patience = 1)
        self.assertIsNone(guard.observe(10, float("nan")))
        self.assertIsNone(guard.observe(11, float("inf")))
        self.assertFalse(guard.triggered)

    def test_payload_requires_points_and_valid_patience(self):
        with self.assertRaises(ValueError):
            ParentLossReference.from_payload(
                {"run_id": "parent", "patience_steps": 5, "points": []}
            )
        with self.assertRaises(ValueError):
            ParentLossReference.from_payload(
                {"run_id": "parent", "patience_steps": 0, "points": [[1, 1.0]]}
            )
        with self.assertRaises(ValueError):
            ParentLossReference.from_payload(
                {
                    "run_id": "parent",
                    "patience_steps": 5,
                    "min_delta": float("nan"),
                    "points": [[1, 1.0]],
                }
            )

    def test_observe_progress_drives_the_guard(self):
        from core.training import webhooks

        guard = _guard([(10, 1.0)], patience = 3)
        for step, loss in ((10, 1.2), (11, 1.1), (12, 1.05), (13, 1.0)):
            webhooks.observe_progress(
                run_id = "parent-guard-observe-progress",
                step = step,
                loss = loss,
                parent_guard = guard,
            )
        self.assertTrue(guard.triggered)
        self.assertEqual(guard.signal["step"], 13)
        self.assertEqual(guard.signal["elapsed_steps"], 3)

    def test_parent_loss_points_filters_unusable_rows(self):
        points = parent_loss_points(
            {
                "loss_step_history": [0, 1, 2, 3, 4],
                "loss_history": [1.0, 0.9, float("nan"), float("inf"), 0.8],
            }
        )
        self.assertEqual(points, [[1, 0.9], [4, 0.8]])


if __name__ == "__main__":
    unittest.main()
