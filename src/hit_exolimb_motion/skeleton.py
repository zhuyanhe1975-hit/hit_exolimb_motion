from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Vector3 = tuple[float, float, float]


REQUIRED_OVERHEAD_JOINTS = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass(frozen=True)
class PoseFrame:
    frame: int
    time: float
    joints: Mapping[str, Vector3]

    def joint(self, name: str) -> Vector3:
        try:
            value = self.joints[name]
        except KeyError as exc:
            raise KeyError(f"missing required joint: {name}") from exc
        if len(value) != 3:
            raise ValueError(f"joint {name!r} must have 3 coordinates")
        return (float(value[0]), float(value[1]), float(value[2]))

    def validate_for_overhead(self) -> None:
        for name in REQUIRED_OVERHEAD_JOINTS:
            self.joint(name)


def midpoint(a: Vector3, b: Vector3) -> Vector3:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)

