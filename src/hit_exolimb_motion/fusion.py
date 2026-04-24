from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from .skeleton import PoseFrame, Vector3


def fuse_body_with_hand_overlays(
    body_frames: list[PoseFrame],
    overlay_frames: list[PoseFrame],
    *,
    hand_blend: float = 0.85,
    smooth_alpha: float = 0.55,
) -> list[PoseFrame]:
    overlays = {frame.frame: frame for frame in overlay_frames}
    fused: list[PoseFrame] = []
    previous_hands: dict[str, Vector3] = {}
    for body in body_frames:
        joints = {name: _vec(point) for name, point in body.joints.items()}
        overlay = overlays.get(body.frame)
        if overlay is not None:
            for side in ("left", "right"):
                _add_hand_from_overlay(joints, overlay.joints, side=side, hand_blend=hand_blend)
        for name in ("left_hand", "right_hand"):
            point = joints.get(name)
            prior = previous_hands.get(name)
            if point is not None and prior is not None:
                joints[name] = _lerp(prior, point, smooth_alpha)
        previous_hands = {
            name: joints[name]
            for name in ("left_hand", "right_hand")
            if name in joints
        }
        fused.append(PoseFrame(frame=body.frame, time=body.time, joints=joints))
    return fused


def _add_hand_from_overlay(
    body: dict[str, Vector3],
    overlay: dict[str, Vector3],
    *,
    side: str,
    hand_blend: float,
) -> None:
    wrist_name = f"{side}_wrist"
    body_wrist = body.get(wrist_name)
    overlay_wrist = overlay.get(wrist_name)
    if body_wrist is None or overlay_wrist is None:
        return

    for suffix, min_len, max_len in (
        ("palm", 0.02, 0.12),
        ("hand", 0.06, 0.18),
        ("thumb", 0.03, 0.16),
        ("index", 0.05, 0.2),
        ("middle", 0.05, 0.22),
        ("ring", 0.05, 0.2),
        ("pinky", 0.05, 0.18),
        ("index_base", 0.02, 0.1),
        ("pinky_base", 0.02, 0.1),
    ):
        name = f"{side}_{suffix}"
        overlay_point = overlay.get(name)
        if overlay_point is None:
            continue
        relative = _sub(overlay_point, overlay_wrist)
        direction = _unit(relative)
        if direction is None:
            continue
        length = _clamp(_norm(relative), min_len, max_len)
        projected = _add(body_wrist, _scale(direction, length))
        base_point = body.get(name, projected)
        body[name] = _lerp(base_point, projected, hand_blend)


def _vec(point: Iterable[float]) -> Vector3:
    x, y, z = point
    return (float(x), float(y), float(z))


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _norm(v: Vector3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: Vector3) -> Vector3 | None:
    length = _norm(v)
    if length <= 1e-6:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def _lerp(a: Vector3, b: Vector3, alpha: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
