from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..skeleton import PoseFrame, Vector3, midpoint

# Official COCO body-17 order used by common MMPose human inferencer configs.
COCO17 = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def import_mmpose_predictions(
    path: Path,
    *,
    fps: float,
    person_index: int = 0,
    score_threshold: float = 0.15,
    body_height_m: float = 1.42,
) -> list[PoseFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_predictions = _extract_frame_predictions(payload)
    frames: list[PoseFrame] = []
    for frame_index, predictions in enumerate(frame_predictions):
        if not predictions:
            continue
        person = _select_person(predictions, person_index)
        pose = _prediction_to_pose_frame(
            frame_index=frame_index,
            time_s=frame_index / fps,
            prediction=person,
            score_threshold=score_threshold,
            body_height_m=body_height_m,
        )
        if pose is not None:
            frames.append(pose)
    return frames


def _extract_frame_predictions(payload: Any) -> list[list[dict[str, Any]]]:
    if isinstance(payload, dict) and "predictions" in payload:
        predictions = payload["predictions"]
    else:
        predictions = payload

    if not isinstance(predictions, list):
        raise ValueError("MMPose predictions JSON must contain a top-level list or a 'predictions' list")

    frames: list[list[dict[str, Any]]] = []
    for item in predictions:
        if isinstance(item, list):
            frames.append([candidate for candidate in item if isinstance(candidate, dict)])
        elif isinstance(item, dict):
            frames.append([item])
        else:
            frames.append([])
    return frames


def _select_person(predictions: list[dict[str, Any]], person_index: int) -> dict[str, Any]:
    if 0 <= person_index < len(predictions):
        return predictions[person_index]

    def rank_key(candidate: dict[str, Any]) -> tuple[float, float]:
        keypoints = candidate.get("keypoints") or []
        bbox = candidate.get("bbox")
        area = 0.0
        if isinstance(bbox, list) and bbox and isinstance(bbox[0], list) and len(bbox[0]) >= 4:
            x1, y1, x2, y2 = bbox[0][:4]
            area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        score = float(sum(candidate.get("keypoint_scores", []) or [0.0]))
        if not area and len(keypoints) >= 17:
            xs = [float(point[0]) for point in keypoints]
            ys = [float(point[1]) for point in keypoints]
            area = max(xs) - min(xs)
            area *= max(ys) - min(ys)
        return area, score

    return max(predictions, key=rank_key)


def _prediction_to_pose_frame(
    *,
    frame_index: int,
    time_s: float,
    prediction: dict[str, Any],
    score_threshold: float,
    body_height_m: float,
) -> PoseFrame | None:
    keypoints = prediction.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) < 17:
        return None

    scores = prediction.get("keypoint_scores") or [1.0] * len(keypoints)
    left_hip = _joint_2d(keypoints, scores, "left_hip", score_threshold)
    right_hip = _joint_2d(keypoints, scores, "right_hip", score_threshold)
    left_ankle = _joint_2d(keypoints, scores, "left_ankle", score_threshold)
    right_ankle = _joint_2d(keypoints, scores, "right_ankle", score_threshold)
    left_shoulder = _joint_2d(keypoints, scores, "left_shoulder", score_threshold)
    right_shoulder = _joint_2d(keypoints, scores, "right_shoulder", score_threshold)
    if not all((left_hip, right_hip, left_ankle, right_ankle, left_shoulder, right_shoulder)):
        return None

    hip_center_x = (left_hip[0] + right_hip[0]) * 0.5
    hip_center_y = (left_hip[1] + right_hip[1]) * 0.5
    ankle_center_y = (left_ankle[1] + right_ankle[1]) * 0.5
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
    body_height_px = max(ankle_center_y - shoulder_center_y, 1.0)
    scale = body_height_m / body_height_px

    def joint(name: str) -> Vector3 | None:
        point = _joint_2d(keypoints, scores, name, score_threshold)
        if point is None:
          return None
        return (
            float((hip_center_x - point[0]) * scale),
            float((hip_center_y - point[1]) * scale + 0.95),
            0.0,
        )

    joints: dict[str, Vector3] = {}
    for name in (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ):
        point = joint(name)
        if point is not None:
            joints[name] = point

    if "left_hip" not in joints or "right_hip" not in joints:
        return None
    pelvis = midpoint(joints["left_hip"], joints["right_hip"])
    joints["pelvis"] = pelvis

    if "left_shoulder" in joints and "right_shoulder" in joints:
        shoulder_mid = midpoint(joints["left_shoulder"], joints["right_shoulder"])
        joints["spine"] = midpoint(pelvis, shoulder_mid)
        head = _head_joint(keypoints, scores, hip_center_x, hip_center_y, scale, shoulder_mid, score_threshold)
        if head is not None:
            joints["head"] = head
            joints["neck"] = (
                (shoulder_mid[0] + head[0]) * 0.5,
                shoulder_mid[1] + (head[1] - shoulder_mid[1]) * 0.38,
                0.0,
            )

    required = {"head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
    if not required.issubset(joints):
        return None

    return PoseFrame(frame=frame_index, time=time_s, joints=joints)


def _joint_2d(
    keypoints: list[Any],
    scores: list[Any],
    name: str,
    score_threshold: float,
) -> tuple[float, float] | None:
    index = COCO17[name]
    if index >= len(keypoints):
        return None
    score = float(scores[index]) if index < len(scores) else 1.0
    if score < score_threshold:
        return None
    point = keypoints[index]
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def _head_joint(
    keypoints: list[Any],
    scores: list[Any],
    hip_center_x: float,
    hip_center_y: float,
    scale: float,
    shoulder_mid: Vector3,
    score_threshold: float,
) -> Vector3 | None:
    points = [
        _joint_2d(keypoints, scores, "nose", score_threshold),
        _joint_2d(keypoints, scores, "left_ear", score_threshold),
        _joint_2d(keypoints, scores, "right_ear", score_threshold),
    ]
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    x = sum(point[0] for point in valid) / len(valid)
    y = sum(point[1] for point in valid) / len(valid)
    return (
        float((hip_center_x - x) * scale),
        float(max((hip_center_y - y) * scale + 0.97, shoulder_mid[1] + 0.13)),
        0.0,
    )
