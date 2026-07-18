from __future__ import annotations

import unittest

import fornax
from fornax import ENGINE_API_VERSION, Engine, EngineContractError


class PublicEngineApiTest(unittest.TestCase):
    def test_exact_string_round_trip(self) -> None:
        seen: list[str] = []

        def generate(prompt: str) -> str:
            seen.append(prompt)
            return prompt[::-1]

        engine = Engine(generate)
        self.assertEqual("olleh", engine.generate("hello"))
        self.assertEqual(["hello"], seen)

    def test_empty_and_unicode_strings_are_passed_through(self) -> None:
        engine = Engine(lambda prompt: f"[{prompt}]")
        self.assertEqual("[]", engine.generate(""))
        self.assertEqual("[مرحبا 🌌]", engine.generate("مرحبا 🌌"))

    def test_constructor_requires_callable(self) -> None:
        with self.assertRaisesRegex(TypeError, "generate must be callable"):
            Engine("not-callable")  # type: ignore[arg-type]

    def test_generate_requires_string_input(self) -> None:
        engine = Engine(lambda prompt: prompt)
        with self.assertRaisesRegex(TypeError, "prompt must be a string"):
            engine.generate(42)  # type: ignore[arg-type]

    def test_backend_must_return_string(self) -> None:
        engine = Engine(lambda prompt: 42)  # type: ignore[arg-type,return-value]
        with self.assertRaisesRegex(EngineContractError, "must return a string"):
            engine.generate("hello")

    def test_backend_exception_is_not_hidden(self) -> None:
        class BackendFailure(RuntimeError):
            pass

        def fail(prompt: str) -> str:
            raise BackendFailure(prompt)

        with self.assertRaisesRegex(BackendFailure, "boom"):
            Engine(fail).generate("boom")

    def test_root_exports_versions_and_api(self) -> None:
        self.assertEqual(1, ENGINE_API_VERSION)
        self.assertIs(Engine, fornax.Engine)
        self.assertRegex(fornax.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
