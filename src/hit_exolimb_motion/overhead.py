from __future__ import annotations

from dataclasses import dataclass

from .skeleton import PoseFrame, midpoint


@dataclass(frozen=True)
class OverheadConfig:
    wrist_above_head_margin_m: float = 0.05
    wrist_above_shoulder_margin_m: float = 0.25
    min_duration_s: float = 0.4


@dataclass(frozen=True)
class OverheadState:
    frame: int
    time: float
    left_active: bool
    right_active: bool

    @property
    def active(self) -> bool:
        return self.left_active or self.right_active

    @property
    def side(self) -> str:
        if self.left_active and self.right_active:
            return "both"
        if self.left_active:
            return "left"
        if self.right_active:
            return "right"
        return "none"


def classify_overhead_frame(frame: PoseFrame, config: OverheadConfig) -> OverheadState:
    frame.validate_for_overhead()
    head_y = frame.joint("head")[1]
    shoulder_y = midpoint(frame.joint("left_shoulder"), frame.joint("right_shoulder"))[1]
    left_wrist_y = frame.joint("left_wrist")[1]
    right_wrist_y = frame.joint("right_wrist")[1]

    def active(wrist_y: float) -> bool:
        above_head = wrist_y >= head_y + config.wrist_above_head_margin_m
        above_shoulder = wrist_y >= shoulder_y + config.wrist_above_shoulder_margin_m
        return above_head and above_shoulder

    return OverheadState(
        frame=frame.frame,
        time=frame.time,
        left_active=active(left_wrist_y),
        right_active=active(right_wrist_y),
    )


def detect_overhead_segments(
    frames: list[PoseFrame], config: OverheadConfig
) -> list[dict[str, object]]:
    states = [classify_overhead_frame(frame, config) for frame in frames]
    segments: list[dict[str, object]] = []
    start: OverheadState | None = None
    last: OverheadState | None = None
    sides: set[str] = set()

    def close_segment(end_state: OverheadState) -> None:
        assert start is not None
        duration = end_state.time - start.time
        if duration >= config.min_duration_s:
            segments.append(
                {
                    "label": "overhead_work",
                    "start_frame": start.frame,
                    "end_frame": end_state.frame,
                    "start_time": start.time,
                    "end_time": end_state.time,
                    "duration": duration,
                    "side": "both" if "both" in sides or len(sides) > 1 else next(iter(sides), "unknown"),
                }
            )

    for state in states:
        if state.active:
            if start is None:
                start = state
                sides = set()
            sides.add(state.side)
        elif start is not None and last is not None:
            close_segment(last)
            start = None
            sides = set()
        last = state

    if start is not None and last is not None:
        close_segment(last)

    return segments

