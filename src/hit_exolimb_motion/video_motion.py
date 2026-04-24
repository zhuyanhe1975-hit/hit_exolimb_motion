from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .skeleton import PoseFrame, Vector3


@dataclass(frozen=True)
class KeyPose:
    time: float
    left_wrist: Vector3
    right_wrist: Vector3
    left_elbow: Vector3
    right_elbow: Vector3
    head: Vector3 = (0.0, 1.68, 0.0)
    spine: Vector3 = (0.0, 1.25, 0.0)
    pelvis: Vector3 = (0.0, 0.95, 0.0)


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
    keyposes = [
        # Hands start high in the provided video and touch the overhead panel.
        KeyPose(
            time=0.0,
            left_wrist=(-0.28, 1.95, 0.10),
            right_wrist=(0.28, 1.95, 0.10),
            left_elbow=(-0.32, 1.73, 0.04),
            right_elbow=(0.32, 1.73, 0.04),
            head=(0.0, 1.68, -0.02),
        ),
        KeyPose(
            time=1.3,
            left_wrist=(-0.30, 2.03, 0.04),
            right_wrist=(0.30, 2.03, 0.04),
            left_elbow=(-0.34, 1.76, 0.02),
            right_elbow=(0.34, 1.76, 0.02),
            head=(0.0, 1.70, 0.0),
        ),
        # Exolimb takes support; human hands begin to leave the panel.
        KeyPose(
            time=2.7,
            left_wrist=(-0.26, 1.82, 0.08),
            right_wrist=(0.26, 1.82, 0.08),
            left_elbow=(-0.32, 1.58, 0.06),
            right_elbow=(0.32, 1.58, 0.06),
            head=(0.0, 1.68, 0.02),
        ),
        KeyPose(
            time=4.2,
            left_wrist=(-0.16, 1.38, 0.20),
            right_wrist=(0.16, 1.38, 0.20),
            left_elbow=(-0.24, 1.30, 0.11),
            right_elbow=(0.24, 1.30, 0.11),
            head=(0.0, 1.66, 0.04),
        ),
        # Hands work around chest/front area while the device remains overhead.
        KeyPose(
            time=7.0,
            left_wrist=(-0.14, 1.22, 0.24),
            right_wrist=(0.14, 1.28, 0.24),
            left_elbow=(-0.24, 1.22, 0.10),
            right_elbow=(0.24, 1.25, 0.10),
            head=(0.0, 1.62, 0.06),
        ),
        # User looks down and relaxes the arms.
        KeyPose(
            time=duration,
            left_wrist=(-0.16, 1.04, 0.10),
            right_wrist=(0.17, 1.10, 0.12),
            left_elbow=(-0.22, 1.18, 0.04),
            right_elbow=(0.22, 1.18, 0.04),
            head=(0.0, 1.55, 0.08),
            spine=(0.0, 1.22, 0.02),
        ),
    ]
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
    return [
        {
            "label": "overhead_panel_contact",
            "start_frame": 0,
            "end_frame": int(2.7 * 30),
            "start_time": 0.0,
            "end_time": 2.7,
            "duration": 2.7,
            "side": "both",
        },
        {
            "label": "exolimb_support_takeover",
            "start_frame": int(2.7 * 30),
            "end_frame": int(min(duration, 14.0) * 30),
            "start_time": 2.7,
            "end_time": min(duration, 14.0),
            "duration": min(duration, 14.0) - 2.7,
            "side": "both",
        },
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
        left_wrist=_lerp3(a.left_wrist, b.left_wrist, alpha),
        right_wrist=_lerp3(a.right_wrist, b.right_wrist, alpha),
        left_elbow=_lerp3(a.left_elbow, b.left_elbow, alpha),
        right_elbow=_lerp3(a.right_elbow, b.right_elbow, alpha),
        head=_lerp3(a.head, b.head, alpha),
        spine=_lerp3(a.spine, b.spine, alpha),
        pelvis=_lerp3(a.pelvis, b.pelvis, alpha),
    )


def _make_frame(index: int, time: float, pose: KeyPose) -> PoseFrame:
    shoulder_y = 1.48
    return PoseFrame(
        frame=index,
        time=time,
        joints={
            "pelvis": pose.pelvis,
            "spine": pose.spine,
            "neck": (0.0, 1.52, 0.0),
            "head": pose.head,
            "left_shoulder": (-0.22, shoulder_y, 0.0),
            "right_shoulder": (0.22, shoulder_y, 0.0),
            "left_elbow": pose.left_elbow,
            "right_elbow": pose.right_elbow,
            "left_wrist": pose.left_wrist,
            "right_wrist": pose.right_wrist,
            "left_hip": (-0.14, 0.92, 0.0),
            "right_hip": (0.14, 0.92, 0.0),
            "left_knee": (-0.14, 0.5, 0.02),
            "right_knee": (0.14, 0.5, 0.02),
            "left_ankle": (-0.14, 0.08, 0.0),
            "right_ankle": (0.14, 0.08, 0.0),
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

