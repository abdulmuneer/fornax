from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .planner import Inventory, ModelSpec, Target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_model_target(path: str | Path) -> tuple[ModelSpec, Target]:
    data = read_json(path)
    if "model" not in data or "target" not in data:
        raise ValueError("target JSON must contain top-level 'model' and 'target'")
    return ModelSpec.from_dict(data["model"]), Target.from_dict(data["target"])


def load_inventory(path: str | Path, links_path: str | Path | None = None) -> Inventory:
    data = read_json(path)
    if links_path is not None:
        links_data = read_json(links_path)
        data = dict(data)
        data["links"] = links_data["links"] if "links" in links_data else links_data
    return Inventory.from_dict(data)
