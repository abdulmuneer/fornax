"""Small public text-generation contract.

The activation-tensor orchestrator does not yet own tokenization, sampling, or
detokenization.  This facade therefore requires an explicit text generator and
never substitutes the reference or simulated stage backends.
"""

from __future__ import annotations

from collections.abc import Callable


ENGINE_API_VERSION = 1


class EngineContractError(RuntimeError):
    """A supplied generation backend violated the public text contract."""


class Engine:
    """Stable string-in/string-out facade over an explicit generator.

    ``Engine`` does not load a model, start workers, or select a backend.  The
    caller supplies the generation function so the Python reference package
    cannot accidentally present simulation as model serving.
    """

    def __init__(self, generate: Callable[[str], str]) -> None:
        if not callable(generate):
            raise TypeError("generate must be callable")
        self._generate = generate

    def generate(self, prompt: str) -> str:
        """Pass one exact string to the supplied generator and return a string."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        result = self._generate(prompt)
        if not isinstance(result, str):
            raise EngineContractError("generation backend must return a string")
        return result
