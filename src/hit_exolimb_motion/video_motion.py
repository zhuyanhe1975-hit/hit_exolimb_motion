from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .skeleton import PoseFrame, Vector3


@dataclass(frozen=True)
class KeyPose:
    time: float
    pelvis: Vector3
    spine: Vector3
    neck: Vector3
    head: Vector3
    left_shoulder: Vector3
    right_shoulder: Vector3
    left_elbow: Vector3
    right_elbow: Vector3
    left_wrist: Vector3
    right_wrist: Vector3
    left_hip: Vector3 = (-0.12, 0.91, 0.0)
    right_hip: Vector3 = (0.12, 0.91, 0.0)
    left_knee: Vector3 = (-0.12, 0.48, 0.01)
    right_knee: Vector3 = (0.12, 0.48, 0.01)
    left_ankle: Vector3 = (-0.12, 0.05, 0.0)
    right_ankle: Vector3 = (0.12, 0.05, 0.0)


def probe_video_duration(video: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    payload = json.loads(output)
    return float(payload["format"]["duration"])


def generate_overhead_panel_motion(video: Path, fps: int = 30) -> list[PoseFrame]:
    duration = probe_video_duration(video)
    keyposes = _build_keyposes(duration)
    total = int(round(duration * fps))
    frames: list[PoseFrame] = []
    for i in range(total + 1):
        t = min(i / fps, duration)
        a, b = _surrounding_keyposes(keyposes, t)
        alpha = 0.0 if a.time == b.time else (t - a.time) / (b.time - a.time)
        alpha = _smoothstep(alpha)
        frames.append(_make_frame(i, t, _blend_keypose(a, b, alpha)))
    return frames


def video_overhead_segments(duration: float) -> list[dict[str, object]]:
    work_end = min(duration, 12.2)
    return [
        {
            "label": "overhead_panel_contact",
            "start_frame": int(1.0 * 30),
            "end_frame": int(3.0 * 30),
            "start_time": 1.0,
            "end_time": 3.0,
            "duration": 2.0,
            "side": "both",
        },
        {
            "label": "overhead_tool_operation",
            "start_frame": int(7.1 * 30),
            "end_frame": int(work_end * 30),
            "start_time": 7.1,
            "end_time": work_end,
            "duration": work_end - 7.1,
            "side": "both",
        },
    ]


def _build_keyposes(duration: float) -> list[KeyPose]:
    end_time = max(duration, 14.7)
    return [
        KeyPose(
            time=0.0,
            pelvis=(0.0, 0.95, 0.03),
            spine=(0.0, 1.23, 0.07),
            neck=(0.0, 1.50, 0.07),
            head=(0.0, 1.63, 0.10),
            left_shoulder=(-0.20, 1.45, 0.04),
            right_shoulder=(0.20, 1.45, 0.04),
            left_elbow=(-0.27, 1.08, 0.18),
            right_elbow=(0.27, 1.08, 0.18),
            left_wrist=(-0.25, 0.74, 0.26),
            right_wrist=(0.25, 0.74, 0.26),
        ),
        KeyPose(
            time=0.8,
            pelvis=(0.0, 0.95, 0.01),
            spine=(0.0, 1.24, 0.03),
            neck=(0.0, 1.52, 0.03),
            head=(0.0, 1.66, 0.04),
            left_shoulder=(-0.21, 1.47, 0.01),
            right_shoulder=(0.21, 1.47, 0.01),
            left_elbow=(-0.26, 1.32, 0.08),
            right_elbow=(0.26, 1.32, 0.08),
            left_wrist=(-0.20, 1.03, 0.06),
            right_wrist=(0.20, 1.03, 0.06),
        ),
        KeyPose(
            time=1.35,
            pelvis=(0.0, 0.96, -0.01),
            spine=(0.0, 1.28, -0.02),
            neck=(0.0, 1.56, -0.01),
            head=(0.0, 1.75, 0.0),
            left_shoulder=(-0.22, 1.50, -0.01),
            right_shoulder=(0.22, 1.50, -0.01),
            left_elbow=(-0.23, 1.74, 0.02),
            right_elbow=(0.23, 1.74, 0.02),
            left_wrist=(-0.16, 2.10, 0.02),
            right_wrist=(0.16, 2.10, 0.02),
        ),
        KeyPose(
            time=2.9,
            pelvis=(0.0, 0.96, -0.01),
            spine=(0.0, 1.28, -0.02),
            neck=(0.0, 1.56, -0.01),
            head=(0.0, 1.74, 0.02),
            left_shoulder=(-0.22, 1.50, -0.01),
            right_shoulder=(0.22, 1.50, -0.01),
            left_elbow=(-0.22, 1.75, 0.03),
            right_elbow=(0.22, 1.75, 0.03),
            left_wrist=(-0.14, 2.11, 0.03),
            right_wrist=(0.14, 2.11, 0.03),
        ),
        KeyPose(
            time=4.1,
            pelvis=(0.0, 0.95, 0.0),
            spine=(0.0, 1.24, 0.0),
            neck=(0.0, 1.51, 0.02),
            head=(0.0, 1.66, 0.04),
            left_shoulder=(-0.19, 1.44, 0.01),
            right_shoulder=(0.19, 1.44, 0.01),
            left_elbow=(-0.18, 1.12, 0.06),
            right_elbow=(0.18, 1.12, 0.06),
            left_wrist=(-0.13, 0.86, 0.08),
            right_wrist=(0.12, 0.84, 0.08),
        ),
        KeyPose(
            time=5.6,
            pelvis=(0.0, 0.95, 0.01),
            spine=(0.0, 1.23, 0.04),
            neck=(0.0, 1.52, 0.06),
            head=(0.0, 1.67, 0.08),
            left_shoulder=(-0.18, 1.44, 0.05),
            right_shoulder=(0.18, 1.44, 0.05),
            left_elbow=(-0.13, 1.33, 0.12),
            right_elbow=(0.15, 1.36, 0.12),
            left_wrist=(-0.05, 1.62, 0.14),
            right_wrist=(0.11, 1.76, 0.14),
        ),
        KeyPose(
            time=7.2,
            pelvis=(0.0, 0.96, -0.01),
            spine=(0.0, 1.28, 0.0),
            neck=(0.0, 1.56, 0.02),
            head=(0.0, 1.74, 0.04),
            left_shoulder=(-0.18, 1.48, 0.02),
            right_shoulder=(0.18, 1.48, 0.02),
            left_elbow=(-0.06, 1.62, 0.08),
            right_elbow=(0.17, 1.72, 0.08),
            left_wrist=(0.01, 1.93, 0.08),
            right_wrist=(0.13, 2.13, 0.06),
        ),
        KeyPose(
            time=9.8,
            pelvis=(0.0, 0.96, -0.01),
            spine=(0.0, 1.28, 0.0),
            neck=(0.0, 1.56, 0.02),
            head=(0.0, 1.74, 0.05),
            left_shoulder=(-0.18, 1.48, 0.02),
            right_shoulder=(0.18, 1.48, 0.02),
            left_elbow=(-0.05, 1.64, 0.08),
            right_elbow=(0.16, 1.73, 0.08),
            left_wrist=(0.02, 1.97, 0.07),
            right_wrist=(0.12, 2.14, 0.05),
        ),
        KeyPose(
            time=12.0,
            pelvis=(0.0, 0.95, 0.01),
            spine=(0.0, 1.23, 0.05),
            neck=(0.0, 1.51, 0.06),
            head=(0.0, 1.66, 0.08),
            left_shoulder=(-0.17, 1.43, 0.05),
            right_shoulder=(0.17, 1.43, 0.05),
            left_elbow=(-0.14, 1.14, 0.11),
            right_elbow=(0.14, 1.13, 0.11),
            left_wrist=(-0.11, 0.84, 0.12),
            right_wrist=(0.10, 0.82, 0.12),
        ),
        KeyPose(
            time=end_time,
            pelvis=(0.0, 0.95, 0.0),
            spine=(0.0, 1.22, 0.01),
            neck=(0.0, 1.50, 0.01),
            head=(0.0, 1.64, 0.02),
            left_shoulder=(-0.16, 1.41, 0.01),
            right_shoulder=(0.16, 1.41, 0.01),
            left_elbow=(-0.14, 0.99, 0.03),
            right_elbow=(0.14, 1.00, 0.03),
            left_wrist=(-0.13, 0.60, 0.03),
            right_wrist=(0.12, 0.61, 0.03),
        ),
    ]


def _surrounding_keyposes(keyposes: list[KeyPose], time: float) -> tuple[KeyPose, KeyPose]:
    previous = keyposes[0]
    for keypose in keyposes[1:]:
        if time <= keypose.time:
            return previous, keypose
        previous = keypose
    return keyposes[-1], keyposes[-1]


def _blend_keypose(a: KeyPose, b: KeyPose, alpha: float) -> KeyPose:
    return KeyPose(
        time=a.time + (b.time - a.time) * alpha,
        pelvis=_lerp3(a.pelvis, b.pelvis, alpha),
        spine=_lerp3(a.spine, b.spine, alpha),
        neck=_lerp3(a.neck, b.neck, alpha),
        head=_lerp3(a.head, b.head, alpha),
        left_shoulder=_lerp3(a.left_shoulder, b.left_shoulder, alpha),
        right_shoulder=_lerp3(a.right_shoulder, b.right_shoulder, alpha),
        left_elbow=_lerp3(a.left_elbow, b.left_elbow, alpha),
        right_elbow=_lerp3(a.right_elbow, b.right_elbow, alpha),
        left_wrist=_lerp3(a.left_wrist, b.left_wrist, alpha),
        right_wrist=_lerp3(a.right_wrist, b.right_wrist, alpha),
        left_hip=_lerp3(a.left_hip, b.left_hip, alpha),
        right_hip=_lerp3(a.right_hip, b.right_hip, alpha),
        left_knee=_lerp3(a.left_knee, b.left_knee, alpha),
        right_knee=_lerp3(a.right_knee, b.right_knee, alpha),
        left_ankle=_lerp3(a.left_ankle, b.left_ankle, alpha),
        right_ankle=_lerp3(a.right_ankle, b.right_ankle, alpha),
    )


def _make_frame(index: int, time: float, pose: KeyPose) -> PoseFrame:
    return PoseFrame(
        frame=index,
        time=time,
        joints={
            "pelvis": pose.pelvis,
            "spine": pose.spine,
            "neck": pose.neck,
            "head": pose.head,
            "left_shoulder": pose.left_shoulder,
            "right_shoulder": pose.right_shoulder,
            "left_elbow": pose.left_elbow,
            "right_elbow": pose.right_elbow,
            "left_wrist": pose.left_wrist,
            "right_wrist": pose.right_wrist,
            "left_hip": pose.left_hip,
            "right_hip": pose.right_hip,
            "left_knee": pose.left_knee,
            "right_knee": pose.right_knee,
            "left_ankle": pose.left_ankle,
            "right_ankle": pose.right_ankle,
        },
    )


def _lerp3(a: Vector3, b: Vector3, alpha: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)
