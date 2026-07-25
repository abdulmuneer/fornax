from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fornax.bounded_subprocess import (
    SubprocessOutputLimitExceeded,
    run_bounded_subprocess,
    start_bounded_subprocess,
)


class BoundedSubprocessTest(unittest.TestCase):
    def test_concurrently_drains_stdout_and_stderr(self) -> None:
        size = 256 * 1024
        result = run_bounded_subprocess(
            (
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    f"sys.stdout.buffer.write(b'o' * {size})\n"
                    "sys.stdout.buffer.flush()\n"
                    f"sys.stderr.buffer.write(b'e' * {size})\n"
                    "sys.stderr.buffer.flush()\n"
                ),
            ),
            timeout_s=5.0,
            stdout_limit_bytes=size,
            stderr_limit_bytes=size,
        )

        self.assertEqual(0, result.returncode)
        self.assertEqual("o" * size, result.stdout)
        self.assertEqual("e" * size, result.stderr)

    def test_stdout_overflow_kills_and_reaps_child(self) -> None:
        process = start_bounded_subprocess(
            (
                sys.executable,
                "-c",
                (
                    "import os, time\n"
                    "while True:\n"
                    "    os.write(1, b'x' * 65536)\n"
                    "    time.sleep(0.001)\n"
                ),
            ),
            stdout_limit_bytes=1024,
            stderr_limit_bytes=1024,
        )

        with self.assertRaises(SubprocessOutputLimitExceeded) as caught:
            process.communicate(timeout=5.0)

        self.assertIn("stdout", caught.exception.streams)
        self.assertEqual(1024, len(caught.exception.stdout.encode("utf-8")))
        self.assertIsNotNone(caught.exception.returncode)
        self.assertIsNotNone(process.poll())

    def test_stderr_overflow_kills_and_reaps_child(self) -> None:
        process = start_bounded_subprocess(
            (
                sys.executable,
                "-c",
                (
                    "import os, time\n"
                    "while True:\n"
                    "    os.write(2, b'e' * 65536)\n"
                    "    time.sleep(0.001)\n"
                ),
            ),
            stdout_limit_bytes=1024,
            stderr_limit_bytes=2048,
        )

        with self.assertRaises(SubprocessOutputLimitExceeded) as caught:
            process.communicate(timeout=5.0)

        self.assertIn("stderr", caught.exception.streams)
        self.assertEqual(2048, len(caught.exception.stderr.encode("utf-8")))
        self.assertIsNotNone(process.poll())

    def test_timeout_kills_and_reaps_child(self) -> None:
        process = start_bounded_subprocess(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            stdout_limit_bytes=1024,
            stderr_limit_bytes=1024,
        )

        with self.assertRaises(subprocess.TimeoutExpired):
            process.communicate(timeout=0.05)

        self.assertIsNotNone(process.poll())

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_exited_leader_does_not_leave_forked_descendant_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "descendant-survived"
            script = (
                "import os, pathlib, time\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    os.close(1)\n"
                "    os.close(2)\n"
                "    time.sleep(0.5)\n"
                f"    pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "    time.sleep(60)\n"
                "    os._exit(0)\n"
                "print(pid, flush=True)\n"
            )
            process = start_bounded_subprocess(
                (sys.executable, "-c", script),
                stdout_limit_bytes=1024,
                stderr_limit_bytes=1024,
            )

            try:
                stdout, _ = process.communicate(timeout=5.0)
                self.assertGreater(int(stdout.strip()), 0)
                time.sleep(0.8)
                self.assertFalse(
                    marker.exists(),
                    "forked descendant survived its exited group leader",
                )
            finally:
                process.kill()
                deadline = time.monotonic() + 0.8
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if marker.exists():
                    descendant_pid = int(marker.read_text())
                    try:
                        if os.getpgid(descendant_pid) == process._process.pid:
                            os.kill(descendant_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
