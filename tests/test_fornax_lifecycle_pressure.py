from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fornax.lifecycle_pressure import main, run_lifecycle_pressure


class LifecyclePressureTest(unittest.TestCase):
    def test_fast_deterministic_unique_request_pressure(self) -> None:
        result = run_lifecycle_pressure(
            min_iterations=128,
            wall_seconds=0,
            defer_every=5,
            track_allocations=False,
        )
        self.assertTrue(result["ok"], result)
        summary = result["summary"]
        self.assertEqual(128, summary["unique_requests_completed"])
        self.assertEqual(
            summary["deferred_for_automatic_expiry"],
            summary["automatic_expiries"],
        )
        self.assertTrue(summary["bounds_ok"])
        self.assertTrue(summary["continuity_ok"])
        self.assertGreaterEqual(
            summary["calendar_elapsed_ns"], summary["max_inter_iteration_gap_ns"]
        )
        self.assertEqual(summary["wall_elapsed_ns"], summary["monotonic_elapsed_ns"])
        final = result["final_health"]
        self.assertEqual(0, final["live_requests"])
        self.assertEqual(0, final["completed_results"])
        self.assertEqual(0, final["native_buffer_bytes"])
        self.assertLessEqual(
            final["release_tombstones"], final["max_release_tombstones"]
        )
        self.assertGreater(final["native_buffer_copy_operations"], 0)
        self.assertFalse(result["physical_evidence"])
        self.assertEqual("reference", result["measurement_kind"])
        self.assertTrue(result["source_identity"]["unchanged_during_run"])
        self.assertIn(
            "fornax/stage_runtime.py",
            result["source_identity"]["before"]["files"],
        )

    def test_module_runner_writes_machine_readable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lifecycle-pressure.json"
            status = main(
                [
                    "--min-iterations",
                    "16",
                    "--defer-every",
                    "3",
                    "--no-tracemalloc",
                    "--out",
                    str(output),
                ]
            )
            self.assertEqual(0, status)
            payload = output.read_text(encoding="utf-8")
            self.assertIn('"record_kind": "fornax-lifecycle-pressure"', payload)
            self.assertIn('"physical_evidence": false', payload)

    def test_rejects_nonfinite_or_zero_continuity_limit(self) -> None:
        for value in (0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    run_lifecycle_pressure(
                        min_iterations=1,
                        max_pause_seconds=value,
                        track_allocations=False,
                    )


if __name__ == "__main__":
    unittest.main()
