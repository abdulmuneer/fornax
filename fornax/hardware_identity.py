"""Observed host identity collection and conservative platform-profile matching.

This module records identity fields reported by the operating system or
``nvidia-smi``.  It does not infer performance, runtime compatibility, or
hardware-support maturity from a successful identity match.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .bounded_subprocess import run_bounded_subprocess


CommandRunner = Callable[[Sequence[str]], Any]

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024
_GB = 1_000_000_000
_COMMAND_STDOUT_LIMIT_BYTES = 4 * _MIB
_COMMAND_STDERR_LIMIT_BYTES = 256 * 1024

SYSTEM_PROFILER_COMMAND = (
    "system_profiler",
    "SPHardwareDataType",
    "-json",
)
NVIDIA_SMI_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,pci.device_id",
    "--format=csv,noheader,nounits",
)
NVIDIA_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)

_PROFILE_SECTIONS = (
    "identity",
    "hardware_identity",
    "identity_match",
    "match",
    "requirements",
)


def _default_command_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return run_bounded_subprocess(
        list(argv),
        timeout_s=10,
        stdout_limit_bytes=_COMMAND_STDOUT_LIMIT_BYTES,
        stderr_limit_bytes=_COMMAND_STDERR_LIMIT_BYTES,
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _coerce_command_result(result: Any) -> tuple[int, str, str]:
    """Normalize common fixture-runner and subprocess result forms."""

    if isinstance(result, str) or isinstance(result, bytes):
        return 0, _text(result), ""
    if isinstance(result, Mapping):
        returncode = result.get("returncode", result.get("code", 0))
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        return int(returncode or 0), _text(stdout), _text(stderr)
    if isinstance(result, tuple):
        if len(result) == 2:
            return int(result[0]), _text(result[1]), ""
        if len(result) == 3:
            return int(result[0]), _text(result[1]), _text(result[2])
        raise TypeError("command-runner tuple results must have two or three items")
    if hasattr(result, "returncode"):
        return (
            int(getattr(result, "returncode", 0) or 0),
            _text(getattr(result, "stdout", "")),
            _text(getattr(result, "stderr", "")),
        )
    raise TypeError(
        "command runner must return text, a mapping, a (code, stdout[, stderr]) "
        "tuple, or a subprocess-style result"
    )


def _run(
    runner: CommandRunner,
    argv: Sequence[str],
) -> tuple[str | None, str | None]:
    try:
        result = runner(list(argv))
        returncode, stdout, stderr = _coerce_command_result(result)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    except Exception as exc:  # Fixture runners may use lookup errors for absent tools.
        return None, f"{type(exc).__name__}: {exc}"
    if returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit status {returncode}"
        return None, detail
    return stdout, None


def _first_hardware_mapping(value: Any) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            candidates.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if not candidates:
        return None
    keys = {
        "chip_type",
        "machine_model",
        "machine_name",
        "model_number",
        "physical_memory",
    }
    return max(candidates, key=lambda item: len(keys.intersection(item.keys())))


def _parse_system_profiler(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "machine_name": None,
        "model_identifier": None,
        "model_number": None,
        "chip": None,
        "memory_text": None,
    }
    stripped = text.strip()
    if not stripped:
        return result

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        item = _first_hardware_mapping(payload)
        if item is not None:
            result.update(
                {
                    "machine_name": _optional_string(
                        item.get("machine_name", item.get("_name"))
                    ),
                    "model_identifier": _optional_string(item.get("machine_model")),
                    "model_number": _optional_string(item.get("model_number")),
                    "chip": _optional_string(
                        item.get("chip_type", item.get("chip"))
                    ),
                    "memory_text": _optional_string(item.get("physical_memory")),
                }
            )
        return result

    labels = {
        "model name": "machine_name",
        "model identifier": "model_identifier",
        "model number": "model_number",
        "chip": "chip",
        "memory": "memory_text",
    }
    for line in stripped.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = labels.get(label.strip().casefold())
        if key is not None and result[key] is None:
            result[key] = _optional_string(value)
    return result


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _host_platform_identity() -> dict[str, Any]:
    """Collect the OS and architecture facts needed by platform profiles."""

    system = platform.system() or None
    os_release: dict[str, str | None] | None = None
    if isinstance(system, str) and system.casefold() == "linux":
        release_reader = getattr(platform, "freedesktop_os_release", None)
        if callable(release_reader):
            try:
                release = release_reader()
            except OSError:
                release = {}
            if isinstance(release, Mapping):
                os_release = {
                    "id": _optional_string(release.get("ID")),
                    "version_id": _optional_string(release.get("VERSION_ID")),
                    "name": _optional_string(release.get("NAME")),
                    "pretty_name": _optional_string(release.get("PRETTY_NAME")),
                }
    return {
        "system": system,
        "machine": platform.machine() or None,
        "node": platform.node() or None,
        "os_release": os_release,
    }


def _parse_positive_int(text: str) -> int | None:
    try:
        value = int(text.strip(), 10)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_nvidia_smi(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    gpus: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_uuids: set[str] = set()
    for row_number, row in enumerate(csv.reader(StringIO(text.strip())), start=1):
        if not row:
            continue
        if len(row) != 7:
            errors.append(
                f"nvidia-smi row {row_number}: expected 7 columns, got {len(row)}"
            )
            continue
        index, gpu_uuid, name, memory_mib, driver, pci_bus_id, pci_device_id = (
            cell.strip() for cell in row
        )
        try:
            parsed_index = int(index, 10)
        except ValueError:
            errors.append(
                f"nvidia-smi row {row_number}: invalid GPU index {index!r}"
            )
            continue
        try:
            memory_decimal = Decimal(memory_mib)
            memory_bytes_decimal = memory_decimal * _MIB
            if (
                memory_decimal <= 0
                or memory_decimal != memory_decimal.to_integral_value()
                or memory_bytes_decimal != memory_bytes_decimal.to_integral_value()
            ):
                raise ValueError
            parsed_memory_mib = int(memory_decimal)
            parsed_memory_bytes = int(memory_bytes_decimal)
        except (InvalidOperation, ValueError):
            errors.append(
                f"nvidia-smi row {row_number}: invalid memory.total {memory_mib!r}"
            )
            continue
        missing = [
            field
            for field, value in {
                "name": name,
                "uuid": gpu_uuid,
                "driver_version": driver,
                "pci_bus_id": pci_bus_id,
                "pci_device_id": pci_device_id,
            }.items()
            if not value or value.casefold() in {"n/a", "[not supported]"}
        ]
        if missing:
            errors.append(
                f"nvidia-smi row {row_number}: missing exact "
                + ", ".join(missing)
            )
            continue
        if NVIDIA_GPU_UUID_RE.fullmatch(gpu_uuid) is None:
            errors.append(
                f"nvidia-smi row {row_number}: invalid physical GPU UUID "
                f"{gpu_uuid!r}"
            )
            continue
        normalized_uuid = gpu_uuid.casefold()
        if normalized_uuid in seen_uuids:
            errors.append(
                f"nvidia-smi row {row_number}: duplicate physical GPU UUID "
                f"{gpu_uuid!r}"
            )
            continue
        seen_uuids.add(normalized_uuid)
        gpus.append(
            {
                "index": parsed_index,
                "uuid": gpu_uuid,
                "name": name,
                "memory_total_mib": parsed_memory_mib,
                "memory_total_bytes": parsed_memory_bytes,
                "driver_version": driver,
                "pci_bus_id": pci_bus_id,
                "pci_device_id": pci_device_id,
            }
        )
    return gpus, errors


def _collect_apple(
    runner: CommandRunner,
    system_profiler_text: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    parsed = _parse_system_profiler(system_profiler_text)
    if not any(parsed.values()):
        return None, ["system_profiler returned no recognizable hardware fields"]

    sysctl_values: dict[str, str | None] = {}
    for key in ("hw.memsize", "hw.model", "machdep.cpu.brand_string"):
        stdout, error = _run(runner, ("sysctl", "-n", key))
        sysctl_values[key] = stdout.strip() if stdout is not None else None
        if error is not None and key == "hw.memsize":
            errors.append(f"sysctl {key}: {error}")

    sw_vers: dict[str, str | None] = {}
    for flag, key in (
        ("-productName", "name"),
        ("-productVersion", "version"),
        ("-buildVersion", "build"),
    ):
        stdout, _error = _run(runner, ("sw_vers", flag))
        sw_vers[key] = stdout.strip() if stdout is not None else None

    chip = parsed["chip"]
    brand_string = _optional_string(sysctl_values["machdep.cpu.brand_string"])
    if chip is None and brand_string is not None:
        chip = brand_string
    model_identifier = (
        parsed["model_identifier"]
        or _optional_string(sysctl_values["hw.model"])
    )
    memory_bytes = (
        _parse_positive_int(sysctl_values["hw.memsize"])
        if sysctl_values["hw.memsize"] is not None
        else None
    )
    if chip is None:
        errors.append("Apple identity is missing the exact chip reported by the OS")
    if memory_bytes is None:
        errors.append("Apple identity is missing exact hw.memsize bytes")

    apple = {
        "machine_name": parsed["machine_name"],
        "model_identifier": model_identifier,
        "model_number": parsed["model_number"],
        "chip": chip,
        "memory_bytes": memory_bytes,
        "memory_text": parsed["memory_text"],
        "memory_source": "sysctl hw.memsize" if memory_bytes is not None else None,
        "os": {
            "name": sw_vers["name"],
            "version": sw_vers["version"],
            "build": sw_vers["build"],
        },
    }
    return apple, errors


def collect_host_identity(
    command_runner: CommandRunner | None = None,
    *,
    executable_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect exact Apple or NVIDIA identity fields exposed by local tools.

    ``command_runner`` receives an argument sequence and may return a
    ``subprocess.CompletedProcess``-like object, a mapping, text, or a
    ``(returncode, stdout[, stderr])`` tuple.  This makes the collector usable
    with deterministic command fixtures on hosts without the target hardware.
    """

    collection_mode = (
        "live_subprocess" if command_runner is None else "injected_fixture_runner"
    )
    if command_runner is not None and executable_paths is not None:
        raise ValueError(
            "executable_paths cannot be combined with an injected command runner"
        )
    if executable_paths is not None:
        normalized_paths: dict[str, str] = {}
        for name, path in executable_paths.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(path, str)
                or not path
                or "\x00" in path
                or not Path(path).is_absolute()
            ):
                raise ValueError(
                    "executable_paths must map command names to absolute "
                    "NUL-free executable paths"
                )
            normalized_paths[name] = path

        def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
            rebound = list(argv)
            if rebound and rebound[0] in normalized_paths:
                rebound[0] = normalized_paths[rebound[0]]
            return _default_command_runner(rebound)

    else:
        runner = command_runner or _default_command_runner
    diagnostics: dict[str, list[str]] = {"apple": [], "nvidia": []}
    collection_errors: list[str] = []

    profiler_stdout, profiler_error = _run(runner, SYSTEM_PROFILER_COMMAND)
    apple: dict[str, Any] | None = None
    if profiler_stdout is not None:
        apple, apple_errors = _collect_apple(runner, profiler_stdout)
        collection_errors.extend(f"apple: {error}" for error in apple_errors)
    elif profiler_error is not None:
        diagnostics["apple"].append(profiler_error)

    smi_stdout, smi_error = _run(runner, NVIDIA_SMI_COMMAND)
    nvidia: dict[str, Any] | None = None
    if smi_stdout is not None:
        gpus, gpu_errors = _parse_nvidia_smi(smi_stdout)
        collection_errors.extend(f"nvidia: {error}" for error in gpu_errors)
        if gpus:
            driver_versions = sorted({gpu["driver_version"] for gpu in gpus})
            nvidia = {
                "gpus": gpus,
                "gpu_count": len(gpus),
                "driver_versions": driver_versions,
            }
        elif not gpu_errors:
            collection_errors.append("nvidia: nvidia-smi reported no GPUs")
    elif smi_error is not None:
        diagnostics["nvidia"].append(smi_error)

    if apple is not None and nvidia is not None:
        kind = "mixed"
    elif apple is not None:
        kind = "apple"
    elif nvidia is not None:
        kind = "nvidia"
    else:
        kind = "unknown"
        collection_errors.append(
            "no Apple system_profiler identity or NVIDIA nvidia-smi identity was collected"
        )

    return {
        "version": 1,
        "kind": kind,
        "collection_provenance": {
            "mode": collection_mode,
            "physical_observation_eligible": command_runner is None,
            "authenticated": False,
        },
        "host": _host_platform_identity(),
        "apple": apple,
        "nvidia": nvidia,
        "collection_errors": collection_errors,
        "diagnostics": diagnostics,
        "interpretation": (
            "Observed identity only; this is not runtime-support, correctness, "
            "performance, or support-maturity evidence."
        ),
    }


def _profile_lookup(
    profile: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[bool, Any, str | None]:
    for key in keys:
        if key in profile:
            return True, profile[key], key
    for section_name in _PROFILE_SECTIONS:
        section = profile.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key in keys:
            if key in section:
                return True, section[key], f"{section_name}.{key}"
    return False, None, None


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _numeric_version(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)*", normalized) is None:
        return None
    return tuple(int(component, 10) for component in normalized.split("."))


def _version_at_least(observed: Any, minimum: Any) -> bool | None:
    observed_parts = _numeric_version(observed)
    minimum_parts = _numeric_version(minimum)
    if observed_parts is None or minimum_parts is None:
        return None
    width = max(len(observed_parts), len(minimum_parts))
    observed_padded = observed_parts + (0,) * (width - len(observed_parts))
    minimum_padded = minimum_parts + (0,) * (width - len(minimum_parts))
    return observed_padded >= minimum_padded


def _runtime_requirement(
    profile: Mapping[str, Any],
    field: str,
) -> Any:
    runtime = profile.get("runtime")
    return runtime.get(field) if isinstance(runtime, Mapping) else None


def _canonical_apple_chip(value: str) -> str:
    normalized = _normalized_text(value)
    if normalized.startswith("apple "):
        normalized = normalized[6:]
    return normalized


def _canonical_os_family(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalized_text(value)
    aliases = {
        "darwin": "macos",
        "mac os": "macos",
        "macos": "macos",
        "gnu/linux": "linux",
        "linux": "linux",
        "win32": "windows",
        "windows": "windows",
    }
    return aliases.get(normalized, normalized)


def _canonical_architecture(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalized_text(value).replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return aliases.get(normalized, normalized)


def _linux_release_requirement(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*([A-Za-z][A-Za-z0-9._-]*)\s+(\d+(?:\.\d+)*)\s*",
        value,
    )
    if match is None:
        return None
    return match.group(1).casefold(), match.group(2)


def _vendor_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalized_text(value)
    if normalized in {"apple", "apple silicon", "mac", "macos"}:
        return "apple"
    if normalized in {"nvidia", "cuda"}:
        return "nvidia"
    return None


def _relevant_collection_errors(
    identity: Mapping[str, Any],
    vendor: str | None,
) -> list[str]:
    raw_errors = identity.get("collection_errors")
    if raw_errors is None:
        return []
    if not isinstance(raw_errors, list):
        return ["identity.collection_errors must be a list"]

    relevant: list[str] = []
    for index, raw_error in enumerate(raw_errors):
        if not isinstance(raw_error, str) or not raw_error.strip():
            relevant.append(
                f"identity.collection_errors[{index}] must be a non-empty string"
            )
            continue
        error = raw_error.strip()
        normalized = error.casefold()
        if normalized.startswith("apple:"):
            error_vendor = "apple"
        elif normalized.startswith("nvidia:"):
            error_vendor = "nvidia"
        else:
            error_vendor = None
        if vendor is None or error_vendor is None or error_vendor == vendor:
            relevant.append(f"identity collection error: {error}")
    return relevant


def _positive_number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return parsed


def _memory_value_to_bytes(value: Any, multiplier: int) -> int | None:
    parsed = _positive_number(value)
    if parsed is None:
        return None
    byte_value = parsed * multiplier
    if byte_value != byte_value.to_integral_value():
        return None
    return int(byte_value)


def _parse_memory_string(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(bytes?|b|mib|mb|gib|gb)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    unit = match.group(2).casefold()
    multipliers = {
        "byte": 1,
        "bytes": 1,
        "b": 1,
        "mib": _MIB,
        "mb": 1_000_000,
        "gib": _GIB,
        "gb": _GB,
    }
    return _memory_value_to_bytes(match.group(1), multipliers[unit])


def _extract_minimum_memory(
    profile: Mapping[str, Any],
) -> tuple[int | None, str | None, str | None]:
    aliases = (
        (
            (
                "minimum_memory_bytes",
                "min_memory_bytes",
                "memory_bytes",
                "total_memory_bytes",
            ),
            1,
        ),
        (
            (
                "minimum_memory_mib",
                "min_memory_mib",
                "memory_mib",
                "memory_total_mib",
            ),
            _MIB,
        ),
        (
            (
                "minimum_memory_gib",
                "min_memory_gib",
                "memory_gib",
                "unified_memory_gib",
                "vram_gib",
            ),
            _GIB,
        ),
        (
            (
                "minimum_memory_gb",
                "min_memory_gb",
                "memory_gb",
                "memory_capacity_gb",
                "unified_memory_gb",
                "vram_gb",
            ),
            _GB,
        ),
    )
    for keys, multiplier in aliases:
        found, value, path = _profile_lookup(profile, keys)
        if found:
            parsed = _memory_value_to_bytes(value, multiplier)
            if parsed is None:
                return None, path, f"{path} must be a positive exact memory value"
            return parsed, path, None
    found, value, path = _profile_lookup(
        profile,
        ("minimum_memory", "min_memory", "memory"),
    )
    if found:
        parsed = _parse_memory_string(value)
        if parsed is None:
            return (
                None,
                path,
                f"{path} must include an explicit B, MB, MiB, GB, or GiB unit",
            )
        return parsed, path, None
    return None, None, "platform profile must supply a minimum memory requirement"


def _extract_units(
    profile: Mapping[str, Any],
) -> tuple[int | None, bool, str | None]:
    found, value, path = _profile_lookup(
        profile,
        ("units", "unit_count", "gpu_count", "count"),
    )
    if not found:
        return None, False, None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, True, f"{path} must be a positive integer"
    return value, True, None


def _identity_apple(identity: Mapping[str, Any]) -> Mapping[str, Any] | None:
    apple = identity.get("apple")
    if isinstance(apple, Mapping):
        return apple
    if _vendor_name(identity.get("vendor")) == "apple" or identity.get("kind") == "apple":
        return identity
    return None


def _identity_nvidia_gpus(identity: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nvidia = identity.get("nvidia")
    values: Any = nvidia.get("gpus") if isinstance(nvidia, Mapping) else None
    if values is None:
        values = identity.get("gpus", identity.get("devices"))
    if not isinstance(values, list):
        return []
    return [gpu for gpu in values if isinstance(gpu, Mapping)]


def _observed_memory_bytes(
    value: Mapping[str, Any],
    *,
    apple: bool = False,
) -> int | None:
    keys = (
        ("memory_bytes", "total_memory_bytes", "unified_memory_bytes")
        if apple
        else ("memory_total_bytes", "total_memory_bytes", "memory_bytes")
    )
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            return raw
    raw_mib = value.get("memory_total_mib", value.get("memory_mib"))
    if isinstance(raw_mib, int) and not isinstance(raw_mib, bool) and raw_mib > 0:
        return raw_mib * _MIB
    return None


def _apple_profile_mislabels(
    profile: Mapping[str, Any],
    expected_chip: str | None,
    minimum_memory_bytes: int | None,
) -> list[str]:
    errors: list[str] = []
    normalized_chip = (
        _canonical_apple_chip(expected_chip) if expected_chip is not None else ""
    )
    ceilings_gb = {
        "m3 max": 128,
        "m3 ultra": 512,
        "m5": 32,
        "m5 pro": 64,
        "m5 max": 128,
    }
    ceiling_gb = ceilings_gb.get(normalized_chip)
    if (
        ceiling_gb is not None
        and minimum_memory_bytes is not None
        and minimum_memory_bytes > ceiling_gb * _GIB
    ):
        errors.append(
            f"Apple {expected_chip} is mislabeled with memory above the selected "
            f"{ceiling_gb}GB configuration ceiling"
        )

    labels: list[str] = []
    for key in ("label", "display_name", "profile_name", "name", "sku"):
        found, value, _path = _profile_lookup(profile, (key,))
        if found and isinstance(value, str):
            labels.append(_normalized_text(value))
    if any(
        re.search(r"\bm5(?:\s+(?:pro|max|ultra))?\b", label)
        and re.search(r"\b512\s*(?:gib|gb)?\b", label)
        for label in labels
    ):
        errors.append(
            "Apple M5 512GB is not a valid selected platform identity; "
            "the 512GB target is Apple M3 Ultra"
        )
    return errors


def _expected_string(
    profile: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[str | None, str | None]:
    found, value, path = _profile_lookup(profile, keys)
    if not found:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, f"{path} must be a non-empty string"
    return value.strip(), None


def match_platform_identity(
    identity: Mapping[str, Any],
    platform_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Match observed identity against exact profile identity and minimum memory.

    For NVIDIA profiles, omitted ``units`` means that any one observed GPU may
    satisfy the profile.  An explicitly supplied ``units > 1`` additionally
    requires at least that many GPUs and requires every observed GPU to match the
    exact name and minimum-memory constraint.
    """

    errors: list[str] = []
    warnings: list[str] = []
    observed: dict[str, Any] = {}
    expected: dict[str, Any] = {}

    if not isinstance(identity, Mapping):
        return {
            "ok": False,
            "errors": ["identity must be a mapping"],
            "warnings": [],
            "observed": {},
            "expected": {},
        }
    if not isinstance(platform_profile, Mapping):
        return {
            "ok": False,
            "errors": ["platform_profile must be a mapping"],
            "warnings": [],
            "observed": {},
            "expected": {},
        }

    vendor_found, vendor_value, vendor_path = _profile_lookup(
        platform_profile,
        ("vendor", "accelerator_vendor"),
    )
    vendor = _vendor_name(vendor_value) if vendor_found else None

    expected_chip, chip_error = _expected_string(
        platform_profile,
        ("chip", "chip_name", "exact_chip", "chip_exact"),
    )
    expected_gpu_name, gpu_name_error = _expected_string(
        platform_profile,
        (
            "gpu_name",
            "exact_gpu_name",
            "device_name",
            "exact_name",
        ),
    )
    generic_model, generic_model_error = _expected_string(
        platform_profile,
        ("model",),
    )
    if expected_chip is None and generic_model is not None:
        if re.search(r"\b(?:apple\s+)?m\d", generic_model, re.IGNORECASE):
            expected_chip = generic_model
    if expected_gpu_name is None and generic_model is not None:
        if re.search(r"\b(?:nvidia|geforce|rtx|h100|b100|b200)\b", generic_model, re.IGNORECASE):
            expected_gpu_name = generic_model

    if vendor is None and not vendor_found:
        if expected_chip is not None and expected_gpu_name is None:
            vendor = "apple"
            warnings.append("platform-profile vendor inferred from its Apple chip field")
        elif expected_gpu_name is not None and expected_chip is None:
            vendor = "nvidia"
            warnings.append("platform-profile vendor inferred from its GPU-name field")
        else:
            errors.append("platform profile must supply vendor as apple or nvidia")
    elif vendor is None:
        errors.append(f"{vendor_path} must identify apple or nvidia")

    errors.extend(_relevant_collection_errors(identity, vendor))

    minimum_memory_bytes, memory_source, memory_error = _extract_minimum_memory(
        platform_profile
    )
    units, units_supplied, units_error = _extract_units(platform_profile)
    if memory_error is not None:
        errors.append(memory_error)
    if units_error is not None:
        errors.append(units_error)

    expected.update(
        {
            "vendor": vendor,
            "minimum_memory_bytes": minimum_memory_bytes,
            "memory_profile_field": memory_source,
            "units": units,
            "units_supplied": units_supplied,
        }
    )

    host = identity.get("host")
    if not isinstance(host, Mapping):
        host = {}
    observed_system = _optional_string(host.get("system"))
    observed_architecture = _optional_string(host.get("machine"))
    observed_os_family = _canonical_os_family(observed_system)
    observed_architecture_canonical = _canonical_architecture(
        observed_architecture
    )
    os_release = host.get("os_release")
    if not isinstance(os_release, Mapping):
        os_release = {}
    observed_release_id = _optional_string(os_release.get("id"))
    observed_release_version = _optional_string(os_release.get("version_id"))
    observed.update(
        {
            "host_os_family": observed_os_family,
            "host_system": observed_system,
            "host_architecture": observed_architecture_canonical,
            "host_architecture_reported": observed_architecture,
            "host_os_release_id": observed_release_id,
            "host_os_release_version": observed_release_version,
        }
    )

    vendor_label = {
        "apple": "Apple",
        "nvidia": "NVIDIA",
    }.get(vendor, "platform")
    expected_os_family = _runtime_requirement(
        platform_profile,
        "os_family",
    )
    if expected_os_family is not None:
        expected_os_family_canonical = _canonical_os_family(expected_os_family)
        expected["os_family"] = expected_os_family_canonical
        if expected_os_family_canonical is None:
            errors.append(
                "runtime.os_family must be a non-empty operating-system family"
            )
        elif observed_os_family is None:
            errors.append(
                f"observed {vendor_label} host is missing its "
                "operating-system family"
            )
        elif observed_os_family != expected_os_family_canonical:
            errors.append(
                f"{vendor_label} host operating-system mismatch: "
                f"expected {expected_os_family_canonical!r}, "
                f"observed {observed_os_family!r}"
            )

    expected_architecture = _runtime_requirement(
        platform_profile,
        "architecture",
    )
    if expected_architecture is not None:
        expected_architecture_canonical = _canonical_architecture(
            expected_architecture
        )
        expected["architecture"] = expected_architecture_canonical
        if expected_architecture_canonical is None:
            errors.append(
                "runtime.architecture must be a non-empty host architecture"
            )
        elif observed_architecture_canonical is None:
            errors.append(
                f"observed {vendor_label} host is missing its machine architecture"
            )
        elif (
            observed_architecture_canonical
            != expected_architecture_canonical
        ):
            errors.append(
                f"{vendor_label} host architecture mismatch: "
                f"expected {expected_architecture_canonical!r}, "
                f"observed {observed_architecture_canonical!r}"
            )

    if vendor == "apple":
        if chip_error is not None:
            errors.append(chip_error)
        if expected_chip is None:
            errors.append("Apple platform profile must supply an exact chip")
        expected["chip"] = expected_chip
        errors.extend(
            _apple_profile_mislabels(
                platform_profile,
                expected_chip,
                minimum_memory_bytes,
            )
        )

        apple_identity = _identity_apple(identity)
        if apple_identity is None:
            errors.append("observed identity contains no Apple hardware identity")
        else:
            observed_chip = _optional_string(
                apple_identity.get("chip", apple_identity.get("chip_type"))
            )
            observed_memory = _observed_memory_bytes(apple_identity, apple=True)
            observed_os = apple_identity.get("os")
            observed_os_version = (
                observed_os.get("version")
                if isinstance(observed_os, Mapping)
                else None
            )
            observed.update(
                {
                    "vendor": "apple",
                    "chip": observed_chip,
                    "memory_bytes": observed_memory,
                    "os_version": observed_os_version,
                    "machine_name": apple_identity.get("machine_name"),
                    "model_identifier": apple_identity.get("model_identifier"),
                    "model_number": apple_identity.get("model_number"),
                }
            )
            if observed_chip is None:
                errors.append("observed Apple identity is missing its exact chip")
            elif (
                expected_chip is not None
                and _canonical_apple_chip(observed_chip)
                != _canonical_apple_chip(expected_chip)
            ):
                errors.append(
                    f"Apple chip mismatch: expected {expected_chip!r}, "
                    f"observed {observed_chip!r}"
                )
            if observed_memory is None:
                errors.append("observed Apple identity is missing exact memory bytes")
            elif (
                minimum_memory_bytes is not None
                and observed_memory < minimum_memory_bytes
            ):
                errors.append(
                    "Apple memory is below the profile minimum: "
                    f"expected at least {minimum_memory_bytes} bytes, "
                    f"observed {observed_memory} bytes"
                )

            capacity_policy = platform_profile.get("capacity_policy")
            exact_memory_bytes = (
                capacity_policy.get("sizing_memory_bytes")
                if isinstance(capacity_policy, Mapping)
                else None
            )
            if exact_memory_bytes is not None:
                expected["exact_memory_bytes"] = exact_memory_bytes
                if (
                    isinstance(exact_memory_bytes, bool)
                    or not isinstance(exact_memory_bytes, int)
                    or exact_memory_bytes <= 0
                ):
                    errors.append(
                        "capacity_policy.sizing_memory_bytes must be a positive "
                        "integer for exact Apple configuration matching"
                    )
                elif observed_memory != exact_memory_bytes:
                    errors.append(
                        "Apple unified memory does not equal the selected "
                        "configuration: "
                        f"expected {exact_memory_bytes} bytes, "
                        f"observed {observed_memory!r}"
                    )

            minimum_os = _runtime_requirement(platform_profile, "minimum_os")
            if minimum_os is not None:
                expected["minimum_os"] = minimum_os
                minimum_os_match = (
                    re.fullmatch(
                        r"macOS\s+(\d+(?:\.\d+)*)",
                        minimum_os,
                        flags=re.IGNORECASE,
                    )
                    if isinstance(minimum_os, str)
                    else None
                )
                if minimum_os_match is None:
                    errors.append(
                        "runtime.minimum_os must use the form macOS "
                        "<numeric-version>"
                    )
                else:
                    os_compatible = _version_at_least(
                        observed_os_version,
                        minimum_os_match.group(1),
                    )
                    if os_compatible is None:
                        errors.append(
                            "observed Apple identity is missing a comparable "
                            "numeric macOS version"
                        )
                    elif not os_compatible:
                        errors.append(
                            "Apple macOS version is below the profile minimum: "
                            f"expected at least {minimum_os!r}, "
                            f"observed {observed_os_version!r}"
                        )

            model_checks = (
                (
                    "machine_name",
                    ("machine_name", "product_name", "computer_name"),
                ),
                (
                    "model_identifier",
                    ("model_identifier", "machine_model", "hardware_model"),
                ),
                ("model_number", ("model_number", "part_number")),
            )
            for observed_key, aliases in model_checks:
                expected_value, value_error = _expected_string(
                    platform_profile,
                    aliases,
                )
                if value_error is not None:
                    errors.append(value_error)
                if expected_value is None:
                    continue
                expected[observed_key] = expected_value
                observed_value = _optional_string(apple_identity.get(observed_key))
                if observed_value is None:
                    errors.append(
                        f"observed Apple identity is missing {observed_key}"
                    )
                elif _normalized_text(observed_value) != _normalized_text(
                    expected_value
                ):
                    errors.append(
                        f"Apple {observed_key} mismatch: expected "
                        f"{expected_value!r}, observed {observed_value!r}"
                    )
        if units_supplied and units is not None and units != 1:
            errors.append(
                "a local Apple host identity represents one SoC; Apple profile "
                "units must be 1"
            )

    elif vendor == "nvidia":
        if gpu_name_error is not None:
            errors.append(gpu_name_error)
        if expected_gpu_name is None:
            errors.append("NVIDIA platform profile must supply an exact GPU name")
        expected["gpu_name"] = expected_gpu_name
        minimum_driver = _runtime_requirement(
            platform_profile,
            "minimum_driver",
        )
        if minimum_driver is not None:
            expected["minimum_driver"] = minimum_driver
            if _numeric_version(minimum_driver) is None:
                errors.append(
                    "runtime.minimum_driver must be a numeric dotted version"
                )

        minimum_os = _runtime_requirement(platform_profile, "minimum_os")
        if minimum_os is not None:
            expected["minimum_os"] = minimum_os
            requirement = _linux_release_requirement(minimum_os)
            if requirement is None:
                errors.append(
                    "runtime.minimum_os must use the form "
                    "<distribution> <numeric-version>"
                )
            else:
                expected_release_id, expected_release_version = requirement
                release_names = {
                    value.casefold()
                    for value in (
                        observed_release_id,
                        _optional_string(os_release.get("name")),
                    )
                    if value is not None
                }
                release_matches = (
                    expected_release_id in release_names
                    or any(
                        name.startswith(expected_release_id + " ")
                        for name in release_names
                    )
                )
                if not release_matches:
                    errors.append(
                        "NVIDIA host distribution mismatch: "
                        f"expected {expected_release_id!r}, "
                        f"observed {observed_release_id!r}"
                    )
                release_compatible = _version_at_least(
                    observed_release_version,
                    expected_release_version,
                )
                if release_compatible is None:
                    errors.append(
                        "observed NVIDIA host is missing a comparable "
                        "distribution version"
                    )
                elif not release_compatible:
                    errors.append(
                        "NVIDIA host operating-system version is below the "
                        f"profile minimum: expected at least {minimum_os!r}, "
                        f"observed {observed_release_version!r}"
                    )

        gpu_identities = _identity_nvidia_gpus(identity)
        gpu_reports: list[dict[str, Any]] = []
        matching_gpus: list[dict[str, Any]] = []
        for gpu in gpu_identities:
            name = _optional_string(gpu.get("name", gpu.get("gpu_name")))
            memory_bytes = _observed_memory_bytes(gpu)
            name_matches = (
                name is not None
                and expected_gpu_name is not None
                and _normalized_text(name) == _normalized_text(expected_gpu_name)
            )
            memory_matches = (
                memory_bytes is not None
                and minimum_memory_bytes is not None
                and memory_bytes >= minimum_memory_bytes
            )
            driver_version = gpu.get("driver_version")
            driver_matches = (
                True
                if minimum_driver is None
                else _version_at_least(driver_version, minimum_driver)
            )
            report = {
                "index": gpu.get("index"),
                "uuid": gpu.get("uuid"),
                "name": name,
                "memory_total_bytes": memory_bytes,
                "driver_version": driver_version,
                "pci_bus_id": gpu.get("pci_bus_id"),
                "pci_device_id": gpu.get("pci_device_id"),
                "name_matches": name_matches,
                "memory_matches": memory_matches,
                "minimum_driver_matches": driver_matches,
            }
            gpu_reports.append(report)
            if name_matches and memory_matches and driver_matches is True:
                matching_gpus.append(report)
        observed.update(
            {
                "vendor": "nvidia",
                "gpu_count": len(gpu_identities),
                "gpus": gpu_reports,
                "matching_gpu_indices": [
                    gpu.get("index") for gpu in matching_gpus
                ],
                "matching_gpu_uuids": [
                    gpu.get("uuid") for gpu in matching_gpus
                ],
            }
        )
        if not gpu_identities:
            errors.append("observed identity contains no NVIDIA GPU identities")
        elif expected_gpu_name is not None and minimum_memory_bytes is not None:
            if units_supplied and units is not None and units > 1:
                if len(gpu_identities) < units:
                    errors.append(
                        f"NVIDIA profile requires at least {units} GPUs; "
                        f"observed {len(gpu_identities)}"
                    )
                nonmatching = [
                    report
                    for report in gpu_reports
                    if not (
                        report["name_matches"]
                        and report["memory_matches"]
                        and report["minimum_driver_matches"] is True
                    )
                ]
                if nonmatching:
                    indices = [report["index"] for report in nonmatching]
                    errors.append(
                        "all observed GPUs must match an explicitly multi-unit "
                        f"profile; nonmatching GPU indices: {indices}"
                    )
            elif not matching_gpus:
                matching_sku_memory = [
                    report
                    for report in gpu_reports
                    if report["name_matches"] and report["memory_matches"]
                ]
                if minimum_driver is not None and matching_sku_memory:
                    observed_drivers = sorted(
                        {
                            str(report["driver_version"])
                            for report in matching_sku_memory
                        }
                    )
                    errors.append(
                        "NVIDIA driver is below or incomparable with the "
                        f"profile minimum {minimum_driver!r}; observed "
                        + ", ".join(observed_drivers)
                    )
                errors.append(
                    "no observed NVIDIA GPU matches the exact name and "
                    "minimum-memory/driver profile"
                )
            elif len(gpu_identities) > len(matching_gpus):
                warnings.append(
                    "additional observed GPUs do not match this single-unit "
                    "profile and were not used"
                )

    if generic_model_error is not None:
        errors.append(generic_model_error)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "observed": observed,
        "expected": expected,
    }
