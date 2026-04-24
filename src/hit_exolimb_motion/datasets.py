from __future__ import annotations

import json
from pathlib import Path


def load_dataset_registry(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_dataset_layout(root: Path, dataset_name: str) -> list[Path]:
    paths = [
        root / "data" / "raw_videos" / dataset_name,
        root / "data" / "pose_tracks" / dataset_name,
        root / "data" / "motions" / dataset_name,
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths

