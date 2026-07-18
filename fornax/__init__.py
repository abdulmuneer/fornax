"""Fornax executable contracts, planner, and Engine v0 simulation tools.

The Python package is the hardware-free reference layer. Physical MAX backends
remain gated separately and never silently fall back to simulation.
"""

from .api import ENGINE_API_VERSION, Engine, EngineContractError

__version__ = "0.1.0"

__all__ = [
    "ENGINE_API_VERSION",
    "Engine",
    "EngineContractError",
    "__version__",
]
