from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .skeleton import PoseFrame


def read_pose_jsonl(path: Path) -> list[PoseFrame]:
    frames: list[PoseFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                frames.append(
                    PoseFrame(
                        frame=int(payload["frame"]),
                        time=float(payload["time"]),
                        joints=payload["joints"],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number}: missing field {exc}") from exc
    return frames


def write_pose_jsonl(path: Path, frames: Iterable[PoseFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(
                json.dumps(
                    {"frame": frame.frame, "time": frame.time, "joints": frame.joints},
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

