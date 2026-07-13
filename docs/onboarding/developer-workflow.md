# Fornax developer workflow

## Local checks

Run `make test`. The suite is CPU-only but binds localhost sockets for the
Engine v0 and HTTP smoke tests.

## Golden vectors

Run `make golden` for contract fixtures and `make unittest` for Python tests.
Changing a golden requires a corresponding contract/spec explanation.

## T1 Engine v0

Use the saved Phase 0.5 artifact for fast validation. Reproducing the full soak
requires the explicit 30-minute command in `docs/getting-started.md`.

## Review and evidence

Classify every result as T0, T1, proxy, or physical. Keep the MAX/Fornax seam,
single-stream latency caveat, and no-silent-fallback rules intact.
