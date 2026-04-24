from __future__ import annotations

from .skeleton import PoseFrame


def generate_overhead_demo(duration_s: float = 4.0, fps: int = 30) -> list[PoseFrame]:
    frames: list[PoseFrame] = []
    total = int(duration_s * fps)
    for i in range(total):
        t = i / fps
        raised = 1.0 <= t <= 3.0
        wrist_y = 1.95 if raised else 1.15
        elbow_y = 1.72 if raised else 1.2
        frames.append(
            PoseFrame(
                frame=i,
                time=t,
                joints={
                    "pelvis": (0.0, 0.95, 0.0),
                    "spine": (0.0, 1.25, 0.0),
                    "neck": (0.0, 1.52, 0.0),
                    "head": (0.0, 1.68, 0.0),
                    "left_shoulder": (-0.22, 1.48, 0.0),
                    "right_shoulder": (0.22, 1.48, 0.0),
                    "left_elbow": (-0.35, elbow_y, 0.04),
                    "right_elbow": (0.35, elbow_y, 0.04),
                    "left_wrist": (-0.28, wrist_y, 0.08),
                    "right_wrist": (0.28, wrist_y, 0.08),
                    "left_hip": (-0.14, 0.92, 0.0),
                    "right_hip": (0.14, 0.92, 0.0),
                    "left_knee": (-0.14, 0.5, 0.02),
                    "right_knee": (0.14, 0.5, 0.02),
                    "left_ankle": (-0.14, 0.08, 0.0),
                    "right_ankle": (0.14, 0.08, 0.0),
                },
            )
        )
    return frames

