"""Subprocess capture with hard per-stream byte limits.

The standard ``subprocess.run(capture_output=True)`` API buffers both streams
without a size limit.  Qualification commands inspect tools and model
runtimes that may be absent, buggy, or unexpectedly verbose, so their live
capture path must remain bounded even while stdout and stderr are written
concurrently.
"""

from __future__ import annotations

import errno
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from typing import BinaryIO


_READ_CHUNK_BYTES = 64 * 1024
_REAP_TIMEOUT_SECONDS = 5.0


class SubprocessOutputLimitExceeded(subprocess.SubprocessError):
    """Raised after an over-limit child has been killed and reaped."""

    def __init__(
        self,
        *,
        cmd: Sequence[str],
        streams: Sequence[str],
        stdout: str,
        stderr: str,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        returncode: int | None,
    ) -> None:
        self.cmd = tuple(cmd)
        self.streams = tuple(sorted(streams))
        self.stdout = stdout
        self.stderr = stderr
        self.output = stdout
        self.stdout_limit_bytes = stdout_limit_bytes
        self.stderr_limit_bytes = stderr_limit_bytes
        self.returncode = returncode
        limits = {
            "stdout": stdout_limit_bytes,
            "stderr": stderr_limit_bytes,
        }
        detail = ", ".join(
            f"{stream}>{limits[stream]} bytes" for stream in self.streams
        )
        super().__init__(f"subprocess output limit exceeded ({detail})")


class SubprocessCaptureError(subprocess.SubprocessError):
    """Raised after a pipe read/decode failure has killed and reaped a child."""


def _positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class BoundedPopen:
    """A small ``Popen`` facade whose stdout/stderr capture is byte-bounded."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a non-empty argument sequence")
        self._argv = tuple(str(part) for part in argv)
        self._stdout_limit = _positive_limit(
            "stdout_limit_bytes", stdout_limit_bytes
        )
        self._stderr_limit = _positive_limit(
            "stderr_limit_bytes", stderr_limit_bytes
        )
        self._buffers = {
            "stdout": bytearray(),
            "stderr": bytearray(),
        }
        self._overflow_streams: set[str] = set()
        self._reader_errors: list[str] = []
        self._state_lock = threading.Lock()
        self._signal_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._readers_finished = False
        self._teardown_started = False
        self._group_released = False
        self._leader_exit_observed = False
        self._exit_wait_kind: str | None = None
        self._exit_kqueue: object | None = None

        self._process = subprocess.Popen(
            list(self._argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=False,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            start_new_session=os.name == "posix",
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._streams: dict[str, BinaryIO] = {
            "stdout": self._process.stdout,
            "stderr": self._process.stderr,
        }
        try:
            self._initialize_exit_observer()
        except BaseException:
            self._signal_process_group(signal.SIGKILL, release=True)
            self._process.wait()
            for stream in self._streams.values():
                try:
                    stream.close()
                except OSError:
                    pass
            raise
        self._threads = [
            threading.Thread(
                target=self._drain,
                args=("stdout", self._process.stdout, self._stdout_limit),
                name=f"bounded-subprocess-stdout-{self._process.pid}",
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=("stderr", self._process.stderr, self._stderr_limit),
                name=f"bounded-subprocess-stderr-{self._process.pid}",
                daemon=True,
            ),
        ]
        started_threads: list[threading.Thread] = []
        try:
            for thread in self._threads:
                thread.start()
                started_threads.append(thread)
        except BaseException:
            self._signal_process_group(signal.SIGKILL, release=True)
            try:
                self._process.wait(timeout=_REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._close_exit_observer()
            with self._state_lock:
                self._teardown_started = True
            for stream in self._streams.values():
                try:
                    stream.close()
                except OSError:
                    pass
            for thread in started_threads:
                thread.join(_REAP_TIMEOUT_SECONDS)
            raise

    @property
    def args(self) -> tuple[str, ...]:
        return self._argv

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def poll(self) -> int | None:
        with self._lifecycle_lock:
            if self._process.returncode is not None:
                return self._process.returncode
            if os.name != "posix":
                return self._process.poll()
            if not self._wait_for_leader_exit(timeout=0):
                return None
            self._kill_and_reap_leader()
            return self._process.returncode

    def _initialize_exit_observer(self) -> None:
        """Observe POSIX leader exit without reaping its group-owning PID."""

        if os.name != "posix":
            return
        if callable(getattr(os, "waitid", None)) and all(
            hasattr(os, name)
            for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
        ):
            self._exit_wait_kind = "waitid"
            return
        if not all(
            hasattr(select, name)
            for name in (
                "kqueue",
                "kevent",
                "KQ_FILTER_PROC",
                "KQ_NOTE_EXIT",
                "KQ_EV_ADD",
                "KQ_EV_ENABLE",
                "KQ_EV_ONESHOT",
            )
        ):
            raise RuntimeError(
                "bounded subprocesses require waitid or kqueue on POSIX"
            )

        queue = select.kqueue()
        try:
            event = select.kevent(
                self._process.pid,
                filter=select.KQ_FILTER_PROC,
                flags=(
                    select.KQ_EV_ADD
                    | select.KQ_EV_ENABLE
                    | select.KQ_EV_ONESHOT
                ),
                fflags=select.KQ_NOTE_EXIT,
            )
            queue.control([event], 0, 0)
        except OSError as exc:
            queue.close()
            if exc.errno == errno.ESRCH:
                # The unreaped leader exited before registration. Its PID is
                # still retained, so process-group teardown remains safe.
                self._leader_exit_observed = True
                return
            raise
        self._exit_wait_kind = "kqueue"
        self._exit_kqueue = queue

    def _close_exit_observer(self) -> None:
        queue = self._exit_kqueue
        self._exit_kqueue = None
        self._exit_wait_kind = None
        if queue is not None:
            try:
                queue.close()
            except OSError:
                pass

    def _wait_for_leader_exit(self, timeout: float | None) -> bool:
        """Wait without reaping so the leader PID cannot be reused as a PGID."""

        if self._process.returncode is not None or self._leader_exit_observed:
            return True
        if os.name != "posix":
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            return True

        deadline: float | None = None
        if timeout is not None:
            timeout_value = float(timeout)
            if not math.isfinite(timeout_value):
                raise ValueError("timeout must be finite")
            deadline = time.monotonic() + max(0.0, timeout_value)

        if self._exit_wait_kind == "waitid":
            flags = os.WEXITED | os.WNOWAIT
            if deadline is not None:
                flags |= os.WNOHANG
            while True:
                try:
                    result = os.waitid(os.P_PID, self._process.pid, flags)
                except InterruptedError:
                    continue
                if result is not None:
                    self._leader_exit_observed = True
                    return True
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    time.sleep(min(0.01, remaining))
                else:
                    time.sleep(0.01)

        if self._exit_wait_kind == "kqueue":
            queue = self._exit_kqueue
            assert queue is not None
            while True:
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                try:
                    events = queue.control(None, 1, remaining)
                except InterruptedError:
                    continue
                if events:
                    self._leader_exit_observed = True
                    return True
                if deadline is not None:
                    return False

        if self._leader_exit_observed:
            return True
        raise SubprocessCaptureError("POSIX child-exit observer is unavailable")

    def _signal_process_group(
        self,
        sig: signal.Signals,
        *,
        release: bool = False,
    ) -> None:
        with self._signal_lock:
            if self._group_released:
                return
            try:
                if os.name == "posix":
                    os.killpg(self._process.pid, sig)
                elif sig == signal.SIGTERM:
                    if self._process.poll() is not None:
                        return
                    self._process.terminate()
                else:
                    if self._process.poll() is not None:
                        return
                    self._process.kill()
            except ProcessLookupError:
                pass
            except PermissionError:
                if sys.platform != "darwin":
                    raise
                # Darwin reports EPERM, rather than ESRCH, when an exited
                # group contains only its unreaped zombie leader. A group
                # with a live same-user descendant remains signalable.
                pass
            if release:
                # Set this before the leader is reaped. Once reaped, its
                # numeric PID could be reused as an unrelated process group.
                self._group_released = True

    def terminate(self) -> None:
        self._signal_process_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_process_group(signal.SIGKILL, release=True)

    def _drain(
        self,
        name: str,
        stream: BinaryIO,
        limit: int,
    ) -> None:
        try:
            while True:
                chunk = stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    return
                with self._state_lock:
                    buffer = self._buffers[name]
                    remaining = max(0, limit - len(buffer))
                    if remaining:
                        buffer.extend(chunk[:remaining])
                    overflowed = len(chunk) > remaining
                    if overflowed:
                        self._overflow_streams.add(name)
                if overflowed:
                    self.kill()
        except (OSError, ValueError) as exc:
            # Closing a pipe during forced teardown can wake a blocked reader.
            # Avoid Popen.poll() here: it would reap the group leader before
            # retained descendants can be killed safely.
            with self._state_lock:
                if not self._teardown_started:
                    self._reader_errors.append(
                        f"{name} capture failed: {type(exc).__name__}: {exc}"
                    )
                    should_kill = True
                else:
                    should_kill = False
            if should_kill:
                self.kill()

    def _captured_text(self, *, errors: str) -> tuple[str, str]:
        with self._state_lock:
            stdout_bytes = bytes(self._buffers["stdout"])
            stderr_bytes = bytes(self._buffers["stderr"])
        try:
            return (
                stdout_bytes.decode("utf-8", errors=errors),
                stderr_bytes.decode("utf-8", errors=errors),
            )
        except UnicodeDecodeError as exc:
            raise SubprocessCaptureError(
                f"subprocess output is not valid UTF-8: {exc}"
            ) from exc

    def _finish_readers(self) -> None:
        if self._readers_finished:
            return
        for thread in self._threads:
            thread.join(_REAP_TIMEOUT_SECONDS)
        if any(thread.is_alive() for thread in self._threads):
            self.kill()
            with self._state_lock:
                self._teardown_started = True
            for stream in self._streams.values():
                try:
                    stream.close()
                except OSError:
                    pass
            for thread in self._threads:
                thread.join(_REAP_TIMEOUT_SECONDS)
        for stream in self._streams.values():
            try:
                stream.close()
            except OSError:
                pass
        self._readers_finished = True
        if any(thread.is_alive() for thread in self._threads):
            raise SubprocessCaptureError(
                "subprocess capture threads did not finish after process teardown"
            )

    def _kill_and_reap_leader(self) -> None:
        self.kill()
        try:
            self._process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._close_exit_observer()

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        with self._lifecycle_lock:
            timed_out = not self._wait_for_leader_exit(timeout)
            self._kill_and_reap_leader()
            self._finish_readers()

        if timed_out:
            stdout, stderr = self._captured_text(errors="replace")
            raise subprocess.TimeoutExpired(
                self._argv,
                timeout,
                output=stdout,
                stderr=stderr,
            )

        with self._state_lock:
            overflow_streams = tuple(self._overflow_streams)
            reader_errors = tuple(self._reader_errors)
        if overflow_streams:
            stdout, stderr = self._captured_text(errors="replace")
            raise SubprocessOutputLimitExceeded(
                cmd=self._argv,
                streams=overflow_streams,
                stdout=stdout,
                stderr=stderr,
                stdout_limit_bytes=self._stdout_limit,
                stderr_limit_bytes=self._stderr_limit,
                returncode=self.returncode,
            )
        if reader_errors:
            raise SubprocessCaptureError("; ".join(reader_errors))
        return self._captured_text(errors="strict")


def start_bounded_subprocess(
    argv: Sequence[str],
    *,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> BoundedPopen:
    """Start a child and immediately drain both output streams."""

    return BoundedPopen(
        argv,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
        cwd=cwd,
        env=env,
    )


def run_bounded_subprocess(
    argv: Sequence[str],
    *,
    timeout_s: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child with concurrent, hard-bounded stdout/stderr capture."""

    process = start_bounded_subprocess(
        argv,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
        cwd=cwd,
        env=env,
    )
    stdout, stderr = process.communicate(timeout=timeout_s)
    assert process.returncode is not None
    return subprocess.CompletedProcess(
        args=list(process.args),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


__all__ = [
    "BoundedPopen",
    "SubprocessCaptureError",
    "SubprocessOutputLimitExceeded",
    "run_bounded_subprocess",
    "start_bounded_subprocess",
]
