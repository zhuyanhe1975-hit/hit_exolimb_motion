from __future__ import annotations

from collections.abc import Iterable
from statistics import median

from .skeleton import PoseFrame, Vector3, midpoint


ARM_UPPER = 0.30
ARM_LOWER = 0.28
LEG_UPPER = 0.43
LEG_LOWER = 0.43
NECK_LENGTH = 0.18
HEAD_LENGTH = 0.16
SPINE_RATIO = 0.52


def stabilize_pose_frames(frames: list[PoseFrame]) -> list[PoseFrame]:
    if not frames:
        return []

    lengths = _estimate_lengths(frames)
    smoothed = _temporal_smooth(frames)
    stabilized: list[PoseFrame] = []
    previous: dict[str, Vector3] | None = None

    for frame in smoothed:
        raw = dict(frame.joints)
        joints: dict[str, Vector3] = {}

        left_hip = raw.get("left_hip")
        right_hip = raw.get("right_hip")
        if left_hip and right_hip:
            pelvis = midpoint(left_hip, right_hip)
        else:
            pelvis = raw.get("pelvis") or previous.get("pelvis") if previous else (0.0, 0.95, 0.0)
        joints["pelvis"] = pelvis

        left_shoulder = raw.get("left_shoulder") or _fallback_offset(previous, "left_shoulder", pelvis, (-0.18, 0.48, 0.0))
        right_shoulder = raw.get("right_shoulder") or _fallback_offset(previous, "right_shoulder", pelvis, (0.18, 0.48, 0.0))
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        spine = _lerp(pelvis, shoulder_mid, SPINE_RATIO)
        joints["spine"] = spine

        neck_target = raw.get("neck") or _fallback_offset(previous, "neck", shoulder_mid, (0.0, 0.10, 0.0))
        joints["neck"] = _fixed_length(shoulder_mid, neck_target, lengths["neck"])

        head_target = raw.get("head") or _fallback_offset(previous, "head", joints["neck"], (0.0, 0.16, 0.0))
        joints["head"] = _fixed_length(joints["neck"], head_target, lengths["head"])

        joints["left_shoulder"] = left_shoulder
        joints["right_shoulder"] = right_shoulder

        joints["left_elbow"] = _fixed_length(
            joints["left_shoulder"],
            raw.get("left_elbow") or _fallback_offset(previous, "left_elbow", joints["left_shoulder"], (-0.10, -0.25, 0.0)),
            lengths["left_upper_arm"],
        )
        joints["right_elbow"] = _fixed_length(
            joints["right_shoulder"],
            raw.get("right_elbow") or _fallback_offset(previous, "right_elbow", joints["right_shoulder"], (0.10, -0.25, 0.0)),
            lengths["right_upper_arm"],
        )
        joints["left_wrist"] = _fixed_length(
            joints["left_elbow"],
            raw.get("left_wrist") or _fallback_offset(previous, "left_wrist", joints["left_elbow"], (-0.06, -0.22, 0.0)),
            lengths["left_lower_arm"],
        )
        joints["right_wrist"] = _fixed_length(
            joints["right_elbow"],
            raw.get("right_wrist") or _fallback_offset(previous, "right_wrist", joints["right_elbow"], (0.06, -0.22, 0.0)),
            lengths["right_lower_arm"],
        )

        left_hip_joint = raw.get("left_hip") or _fallback_offset(previous, "left_hip", pelvis, (-0.12, -0.03, 0.0))
        right_hip_joint = raw.get("right_hip") or _fallback_offset(previous, "right_hip", pelvis, (0.12, -0.03, 0.0))
        joints["left_hip"] = left_hip_joint
        joints["right_hip"] = right_hip_joint
        joints["left_knee"] = _fixed_length(
            joints["left_hip"],
            raw.get("left_knee") or _fallback_offset(previous, "left_knee", joints["left_hip"], (0.0, -0.43, 0.01)),
            lengths["left_upper_leg"],
        )
        joints["right_knee"] = _fixed_length(
            joints["right_hip"],
            raw.get("right_knee") or _fallback_offset(previous, "right_knee", joints["right_hip"], (0.0, -0.43, 0.01)),
            lengths["right_upper_leg"],
        )
        joints["left_ankle"] = _fixed_length(
            joints["left_knee"],
            raw.get("left_ankle") or _fallback_offset(previous, "left_ankle", joints["left_knee"], (0.0, -0.43, 0.0)),
            lengths["left_lower_leg"],
        )
        joints["right_ankle"] = _fixed_length(
            joints["right_knee"],
            raw.get("right_ankle") or _fallback_offset(previous, "right_ankle", joints["right_knee"], (0.0, -0.43, 0.0)),
            lengths["right_lower_leg"],
        )

        stabilized.append(PoseFrame(frame=frame.frame, time=frame.time, joints=joints))
        previous = joints

    return stabilized


def _temporal_smooth(frames: list[PoseFrame]) -> list[PoseFrame]:
    window = (-2, -1, 0, 1, 2)
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    smoothed: list[PoseFrame] = []
    for index, frame in enumerate(frames):
        joints: dict[str, Vector3] = {}
        for name in frame.joints:
            samples: list[Vector3] = []
            sample_weights: list[float] = []
            for offset, weight in zip(window, weights):
                neighbor_index = index + offset
                if 0 <= neighbor_index < len(frames):
                    value = frames[neighbor_index].joints.get(name)
                    if value is not None:
                        samples.append(value)
                        sample_weights.append(weight)
            joints[name] = _weighted_average(samples, sample_weights) if samples else frame.joints[name]
        smoothed.append(PoseFrame(frame=frame.frame, time=frame.time, joints=joints))
    return smoothed


def _estimate_lengths(frames: list[PoseFrame]) -> dict[str, float]:
    return {
        "left_upper_arm": _median_length(frames, "left_shoulder", "left_elbow", ARM_UPPER),
        "right_upper_arm": _median_length(frames, "right_shoulder", "right_elbow", ARM_UPPER),
        "left_lower_arm": _median_length(frames, "left_elbow", "left_wrist", ARM_LOWER),
        "right_lower_arm": _median_length(frames, "right_elbow", "right_wrist", ARM_LOWER),
        "left_upper_leg": _median_length(frames, "left_hip", "left_knee", LEG_UPPER, maximum=0.7),
        "right_upper_leg": _median_length(frames, "right_hip", "right_knee", LEG_UPPER, maximum=0.7),
        "left_lower_leg": _median_length(frames, "left_knee", "left_ankle", LEG_LOWER, maximum=0.7),
        "right_lower_leg": _median_length(frames, "right_knee", "right_ankle", LEG_LOWER, maximum=0.7),
        "neck": NECK_LENGTH,
        "head": HEAD_LENGTH,
    }


def _median_length(
    frames: Iterable[PoseFrame],
    parent: str,
    child: str,
    fallback: float,
    *,
    minimum: float = 0.08,
    maximum: float = 0.6,
) -> float:
    values: list[float] = []
    for frame in frames:
        a = frame.joints.get(parent)
        b = frame.joints.get(child)
        if a is None or b is None:
            continue
        length = _distance(a, b)
        if minimum <= length <= maximum:
            values.append(length)
    return float(median(values)) if values else fallback


def _weighted_average(points: list[Vector3], weights: list[float]) -> Vector3:
    total = sum(weights)
    x = sum(point[0] * weight for point, weight in zip(points, weights))
    y = sum(point[1] * weight for point, weight in zip(points, weights))
    z = sum(point[2] * weight for point, weight in zip(points, weights))
    return (x / total, y / total, z / total)


def _fixed_length(parent: Vector3, target: Vector3, length: float) -> Vector3:
    direction = _sub(target, parent)
    magnitude = _norm(direction)
    if magnitude < 1e-6:
        return (parent[0], parent[1] - length, parent[2])
    unit = (direction[0] / magnitude, direction[1] / magnitude, direction[2] / magnitude)
    return (
        parent[0] + unit[0] * length,
        parent[1] + unit[1] * length,
        parent[2] + unit[2] * length,
    )


def _fallback_offset(
    previous: dict[str, Vector3] | None, name: str, parent: Vector3, offset: Vector3
) -> Vector3:
    if previous and name in previous:
        return previous[name]
    return (parent[0] + offset[0], parent[1] + offset[1], parent[2] + offset[2])


def _lerp(a: Vector3, b: Vector3, alpha: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vector3) -> float:
    return _distance(v, (0.0, 0.0, 0.0))


def _distance(a: Vector3, b: Vector3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5
