from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    paths = {
        "base": base,
        "logs": base / "logs",
        "pcd": base / "pcd",
        "debug": base / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_rgb_image(path: str | Path, image) -> None:
    Image.fromarray(image).save(path)


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
