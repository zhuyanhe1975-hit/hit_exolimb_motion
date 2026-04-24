from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..skeleton import PoseFrame, Vector3, midpoint


@dataclass
class BvhNode:
    name: str
    offset: Vector3 = (0.0, 0.0, 0.0)
    channels: list[str] = field(default_factory=list)
    children: list["BvhNode"] = field(default_factory=list)


@dataclass(frozen=True)
class BvhMotion:
    root: BvhNode
    frames: list[list[float]]
    frame_time: float


JOINT_NAME_MAP: dict[str, str] = {
    "hips_JNT": "pelvis",
    "neck_JNT": "neck",
    "head_JNT": "head",
    "l_arm_JNT": "left_shoulder",
    "r_arm_JNT": "right_shoulder",
    "l_forearm_JNT": "left_elbow",
    "r_forearm_JNT": "right_elbow",
    "l_hand_JNT": "left_wrist",
    "r_hand_JNT": "right_wrist",
    "l_upleg_JNT": "left_hip",
    "r_upleg_JNT": "right_hip",
    "l_leg_JNT": "left_knee",
    "r_leg_JNT": "right_knee",
    "l_foot_JNT": "left_ankle",
    "r_foot_JNT": "right_ankle",
}


def import_bvh_motion(
    path: Path,
    *,
    scale_m: float = 0.01,
    pelvis_height_m: float = 0.95,
    lock_root: bool = True,
) -> list[PoseFrame]:
    motion = _parse_bvh(path)
    world_frames = [_world_positions(motion.root, values) for values in motion.frames]
    if not world_frames:
        return []

    first_pelvis = world_frames[0]["hips_JNT"]
    origin = first_pelvis

    frames: list[PoseFrame] = []
    for frame_index, world in enumerate(world_frames):
        joints = _map_pose_joints(
            world,
            scale_m=scale_m,
            origin=origin,
            pelvis_height_m=pelvis_height_m,
        )
        if joints is None:
            continue
        if lock_root:
            joints = _lock_root_translation(joints, reference_pelvis=(0.0, pelvis_height_m, 0.0))
        frames.append(
            PoseFrame(
                frame=frame_index,
                time=frame_index * motion.frame_time,
                joints=joints,
            )
        )
    return frames


def _parse_bvh(path: Path) -> BvhMotion:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0] != "HIERARCHY":
        raise ValueError(f"{path} is not a BVH file")

    index = 1

    def parse_node() -> BvhNode:
        nonlocal index
        header = lines[index].split()
        if len(header) < 2 or header[0] not in {"ROOT", "JOINT"}:
            raise ValueError(f"{path}: expected ROOT/JOINT at line {index + 1}")
        node = BvhNode(name=header[1])
        index += 1
        if lines[index] != "{":
            raise ValueError(f"{path}: expected '{{' at line {index + 1}")
        index += 1
        while index < len(lines):
            line = lines[index]
            parts = line.split()
            token = parts[0]
            if token == "OFFSET":
                node.offset = (float(parts[1]), float(parts[2]), float(parts[3]))
                index += 1
            elif token == "CHANNELS":
                count = int(parts[1])
                node.channels = parts[2 : 2 + count]
                index += 1
            elif token == "JOINT":
                child = parse_node()
                node.children.append(child)
            elif token == "End":
                index += 1
                if lines[index] != "{":
                    raise ValueError(f"{path}: expected '{{' after End Site at line {index + 1}")
                index += 1
                while lines[index] != "}":
                    index += 1
                index += 1
            elif token == "}":
                index += 1
                return node
            else:
                raise ValueError(f"{path}: unexpected token {token!r} at line {index + 1}")
        raise ValueError(f"{path}: unexpected end of file while parsing hierarchy")

    root = parse_node()
    if index >= len(lines) or lines[index] != "MOTION":
        raise ValueError(f"{path}: missing MOTION section")
    index += 1

    if not lines[index].startswith("Frames:"):
        raise ValueError(f"{path}: missing frame count")
    frame_count = int(lines[index].split(":", 1)[1].strip())
    index += 1

    if not lines[index].startswith("Frame Time:"):
        raise ValueError(f"{path}: missing frame time")
    frame_time = float(lines[index].split(":", 1)[1].strip())
    index += 1

    channel_count = _count_channels(root)
    frames: list[list[float]] = []
    for line in lines[index:]:
        values = [float(value) for value in line.split()]
        if len(values) != channel_count:
            raise ValueError(f"{path}: expected {channel_count} channel values, got {len(values)}")
        frames.append(values)
    if len(frames) != frame_count:
        raise ValueError(f"{path}: expected {frame_count} frames, got {len(frames)}")

    return BvhMotion(root=root, frames=frames, frame_time=frame_time)


def _count_channels(node: BvhNode) -> int:
    return len(node.channels) + sum(_count_channels(child) for child in node.children)


def _world_positions(root: BvhNode, values: list[float]) -> dict[str, Vector3]:
    index = 0
    positions: dict[str, Vector3] = {}

    def walk(node: BvhNode, parent_position: Vector3, parent_rotation: tuple[Vector3, Vector3, Vector3]) -> None:
        nonlocal index
        local_translation = [0.0, 0.0, 0.0]
        local_rotation = _identity_matrix()
        for channel in node.channels:
            value = values[index]
            index += 1
            if channel.endswith("position"):
                axis = "XYZ".index(channel[0].upper())
                local_translation[axis] = value
            elif channel.endswith("rotation"):
                local_rotation = _mat_mul(local_rotation, _axis_rotation(channel[0], math.radians(value)))

        local_offset = (
            node.offset[0] + local_translation[0],
            node.offset[1] + local_translation[1],
            node.offset[2] + local_translation[2],
        )
        world_position = _vec_add(parent_position, _mat_vec(parent_rotation, local_offset))
        world_rotation = _mat_mul(parent_rotation, local_rotation)
        positions[node.name] = world_position
        for child in node.children:
            walk(child, world_position, world_rotation)

    walk(root, (0.0, 0.0, 0.0), _identity_matrix())
    return positions


def _map_pose_joints(
    world: dict[str, Vector3],
    *,
    scale_m: float,
    origin: Vector3,
    pelvis_height_m: float,
) -> dict[str, Vector3] | None:
    if "hips_JNT" not in world:
        return None

    def transform(point: Vector3) -> Vector3:
        return (
            -(point[0] - origin[0]) * scale_m,
            (point[1] - origin[1]) * scale_m + pelvis_height_m,
            (point[2] - origin[2]) * scale_m,
        )

    joints: dict[str, Vector3] = {}
    for source_name, target_name in JOINT_NAME_MAP.items():
        point = world.get(source_name)
        if point is None:
            continue
        joints[target_name] = transform(point)

    spine1 = world.get("spine1_JNT")
    spine2 = world.get("spine2_JNT")
    if spine1 is not None and spine2 is not None:
        spine = midpoint(transform(spine1), transform(spine2))
        joints["spine"] = spine

    if "neck" not in joints and "head" in joints and "spine" in joints:
        joints["neck"] = midpoint(joints["spine"], joints["head"])

    required = {
        "pelvis",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    }
    if not required.issubset(joints):
        return None
    return joints


def _lock_root_translation(
    joints: dict[str, Vector3],
    *,
    reference_pelvis: Vector3,
) -> dict[str, Vector3]:
    pelvis = joints["pelvis"]
    delta = (
        pelvis[0] - reference_pelvis[0],
        pelvis[1] - reference_pelvis[1],
        pelvis[2] - reference_pelvis[2],
    )
    if delta == (0.0, 0.0, 0.0):
        return joints
    return {
        name: (
            point[0] - delta[0],
            point[1] - delta[1],
            point[2] - delta[2],
        )
        for name, point in joints.items()
    }


def _identity_matrix() -> tuple[Vector3, Vector3, Vector3]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _axis_rotation(axis: str, angle: float) -> tuple[Vector3, Vector3, Vector3]:
    c = math.cos(angle)
    s = math.sin(angle)
    axis = axis.upper()
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "Y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    if axis == "Z":
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    raise ValueError(f"unknown rotation axis {axis!r}")


def _mat_mul(
    a: tuple[Vector3, Vector3, Vector3],
    b: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _mat_vec(matrix: tuple[Vector3, Vector3, Vector3], vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
