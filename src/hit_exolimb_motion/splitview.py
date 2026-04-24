from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .pose_extractors.mediapipe import probe_video_stream


@dataclass(frozen=True)
class CropRect:
    x_frac: float
    y_frac: float
    w_frac: float
    h_frac: float


# Tuned for the current overhead.mp4 layout:
# large frontal view plus a lower-left inset side view.
OVERHEAD_INSET_RECT = CropRect(
    x_frac=0.02,
    y_frac=0.60,
    w_frac=0.36,
    h_frac=0.40,
)


def crop_video_region(video: Path, out: Path, rect: CropRect) -> None:
    stream = probe_video_stream(video)
    width = stream.width
    height = stream.height
    crop_w = _even(max(2, round(width * rect.w_frac)))
    crop_h = _even(max(2, round(height * rect.h_frac)))
    crop_x = _even(max(0, min(width - crop_w, round(width * rect.x_frac))))
    crop_y = _even(max(0, min(height - crop_h, round(height * rect.y_frac))))

    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-an",
        str(out),
    ]
    subprocess.run(command, check=True)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1
