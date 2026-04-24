from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..skeleton import PoseFrame, Vector3, midpoint


@dataclass(frozen=True)
class VideoStreamInfo:
    width: int
    height: int
    fps: float
    duration: float


MEDIAPIPE_LANDMARK = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def extract_pose_video_mediapipe(
    video: Path,
    *,
    model: Path | None = None,
    hand_model: Path | None = None,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    include_hands: bool = False,
    min_hand_detection_confidence: float = 0.3,
    min_hand_presence_confidence: float = 0.3,
    min_hand_tracking_confidence: float = 0.3,
) -> list[PoseFrame]:
    try:
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe extraction requires `mediapipe` and `numpy`. "
            "Install them first, then rerun `extract-mediapipe-pose`."
        ) from exc

    if model is None:
        raise RuntimeError(
            "MediaPipe Pose Landmarker requires a `.task` model file. "
            "Pass it with `--model /path/to/pose_landmarker_full.task`."
        )
    if include_hands and hand_model is None:
        default_hand_model = Path("models/mediapipe/hand_landmarker.task")
        if default_hand_model.exists():
            hand_model = default_hand_model
        else:
            raise RuntimeError(
                "MediaPipe hand overlay requires a hand landmarker `.task` model file. "
                "Pass it with `--hand-model /path/to/hand_landmarker.task`."
            )

    stream = probe_video_stream(video)
    base_options = python.BaseOptions(model_asset_path=str(model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_bytes = stream.width * stream.height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    frames: list[PoseFrame] = []
    with vision.PoseLandmarker.create_from_options(options) as detector:
        hand_detector = None
        if include_hands:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_hand_tracking_confidence,
            )
            hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        try:
            if process.stdout is None:
                raise RuntimeError("ffmpeg stdout is unavailable")

            frame_index = 0
            while True:
                raw = process.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (stream.height, stream.width, 3)
                )
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestamp_ms = int(round(frame_index / stream.fps * 1000.0))
                result = detector.detect_for_video(mp_image, timestamp_ms)
                hand_result = (
                    hand_detector.detect_for_video(mp_image, timestamp_ms)
                    if hand_detector is not None
                    else None
                )
                pose = _result_to_pose_frame(
                    frame_index,
                    frame_index / stream.fps,
                    result,
                    hand_result,
                )
                if pose is not None:
                    frames.append(pose)
                frame_index += 1
        finally:
            if hand_detector is not None:
                hand_detector.close()
            if process.stdout is not None:
                process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")

    return frames


def probe_video_stream(video: Path) -> VideoStreamInfo:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    payload = json.loads(output)
    stream = payload["streams"][0]
    fps_num, fps_den = stream["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)
    duration = float(stream.get("duration") or 0.0)
    return VideoStreamInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration=duration,
    )


def _result_to_pose_frame(
    frame_index: int,
    time_s: float,
    result: object,
    hand_result: object | None = None,
) -> PoseFrame | None:
    pose_landmarks = getattr(result, "pose_landmarks", None)
    if not pose_landmarks:
        return None

    pose_world_landmarks = getattr(result, "pose_world_landmarks", None)
    landmarks_2d = pose_landmarks[0]
    landmarks_3d = pose_world_landmarks[0] if pose_world_landmarks else None

    left_hip_2d = landmarks_2d[MEDIAPIPE_LANDMARK["left_hip"]]
    right_hip_2d = landmarks_2d[MEDIAPIPE_LANDMARK["right_hip"]]
    left_ankle_2d = landmarks_2d[MEDIAPIPE_LANDMARK["left_ankle"]]
    right_ankle_2d = landmarks_2d[MEDIAPIPE_LANDMARK["right_ankle"]]
    left_shoulder_2d = landmarks_2d[MEDIAPIPE_LANDMARK["left_shoulder"]]
    right_shoulder_2d = landmarks_2d[MEDIAPIPE_LANDMARK["right_shoulder"]]

    hip_center_x = (left_hip_2d.x + right_hip_2d.x) * 0.5
    hip_center_y = (left_hip_2d.y + right_hip_2d.y) * 0.5
    ankle_center_y = (left_ankle_2d.y + right_ankle_2d.y) * 0.5
    shoulder_center_y = (left_shoulder_2d.y + right_shoulder_2d.y) * 0.5
    body_height = max(ankle_center_y - shoulder_center_y, 0.15)
    scale = 1.42 / body_height
    hip_world_z = 0.0
    if landmarks_3d:
        hip_world_z = float(
            (landmarks_3d[MEDIAPIPE_LANDMARK["left_hip"]].z + landmarks_3d[MEDIAPIPE_LANDMARK["right_hip"]].z)
            * 0.5
        )

    def joint(name: str) -> Vector3:
        idx = MEDIAPIPE_LANDMARK[name]
        lm2d = landmarks_2d[idx]
        z = 0.0
        if landmarks_3d:
            # Keep depth subtle; the reference video is viewed mostly from the front.
            z = float((hip_world_z - landmarks_3d[idx].z) * 0.28)
        return (
            float((hip_center_x - lm2d.x) * scale),
            float((hip_center_y - lm2d.y) * scale + 0.95),
            z,
        )

    left_hip = joint("left_hip")
    right_hip = joint("right_hip")
    pelvis = midpoint(left_hip, right_hip)
    left_shoulder = joint("left_shoulder")
    right_shoulder = joint("right_shoulder")
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    spine = midpoint(pelvis, shoulder_mid)
    head = _head_position(landmarks_2d, landmarks_3d, hip_center_x, hip_center_y, scale, shoulder_mid)
    neck = _neck_position(shoulder_mid, head)

    joints = {
        "pelvis": pelvis,
        "spine": spine,
        "neck": neck,
        "head": head,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": joint("left_elbow"),
        "right_elbow": joint("right_elbow"),
        "left_wrist": joint("left_wrist"),
        "right_wrist": joint("right_wrist"),
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": joint("left_knee"),
        "right_knee": joint("right_knee"),
        "left_ankle": joint("left_ankle"),
        "right_ankle": joint("right_ankle"),
    }

    if hand_result is not None:
        _apply_hands_to_joints(joints, hand_result, hip_center_x, hip_center_y, scale)

    return PoseFrame(
        frame=frame_index,
        time=time_s,
        joints=joints,
    )


def _apply_hands_to_joints(
    joints: dict[str, Vector3],
    hand_result: object,
    hip_center_x: float,
    hip_center_y: float,
    scale: float,
) -> None:
    landmarks_sets = getattr(hand_result, "multi_hand_landmarks", None)
    handedness_sets = getattr(hand_result, "multi_handedness", None)
    if landmarks_sets is None:
        landmarks_sets = getattr(hand_result, "hand_landmarks", None)
    if handedness_sets is None:
        handedness_sets = getattr(hand_result, "handedness", None)
    if not landmarks_sets or not handedness_sets:
        return

    for hand_landmarks, handedness in zip(landmarks_sets, handedness_sets):
        category = handedness[0] if isinstance(handedness, list) else handedness.classification[0]
        label = getattr(category, "category_name", None) or getattr(category, "label", "")
        label = str(label).lower()
        if label not in {"left", "right"}:
            continue

        def avg(indices: list[int]) -> Vector3:
            total_x = total_y = total_z = 0.0
            for index in indices:
                point = hand_landmarks[index] if isinstance(hand_landmarks, list) else hand_landmarks.landmark[index]
                total_x += point.x
                total_y += point.y
                total_z += point.z
            n = float(len(indices))
            return (
                float((hip_center_x - total_x / n) * scale),
                float((hip_center_y - total_y / n) * scale + 0.95),
                float((0.0 - total_z / n) * 0.18),
            )

        wrist = avg([0])
        palm = avg([0, 5, 9, 13, 17])
        hand = avg([8, 12, 16, 20])
        thumb = avg([4])
        index = avg([8])
        middle = avg([12])
        ring = avg([16])
        pinky = avg([20])
        index_base = avg([5])
        pinky_base = avg([17])
        joints[f"{label}_wrist"] = wrist
        joints[f"{label}_palm"] = palm
        joints[f"{label}_hand"] = hand
        joints[f"{label}_thumb"] = thumb
        joints[f"{label}_index"] = index
        joints[f"{label}_middle"] = middle
        joints[f"{label}_ring"] = ring
        joints[f"{label}_pinky"] = pinky
        joints[f"{label}_index_base"] = index_base
        joints[f"{label}_pinky_base"] = pinky_base


def _head_position(
    landmarks_2d: object,
    landmarks_3d: object,
    hip_center_x: float,
    hip_center_y: float,
    scale: float,
    shoulder_mid: Vector3,
) -> Vector3:
    nose = landmarks_2d[MEDIAPIPE_LANDMARK["nose"]]
    left_ear = landmarks_2d[MEDIAPIPE_LANDMARK["left_ear"]]
    right_ear = landmarks_2d[MEDIAPIPE_LANDMARK["right_ear"]]
    face_x = (nose.x + left_ear.x + right_ear.x) / 3.0
    face_y = (nose.y + left_ear.y + right_ear.y) / 3.0
    x = (hip_center_x - face_x) * scale
    y = (hip_center_y - face_y) * scale + 0.97
    z = shoulder_mid[2] + 0.02
    if landmarks_3d:
        nose_world = landmarks_3d[MEDIAPIPE_LANDMARK["nose"]]
        z = float(shoulder_mid[2] + (0.0 - nose_world.z) * 0.18)
    return (float(x), float(max(y, shoulder_mid[1] + 0.13)), z)


def _neck_position(shoulder_mid: Vector3, head: Vector3) -> Vector3:
    return (
        (shoulder_mid[0] + head[0]) * 0.5,
        shoulder_mid[1] + (head[1] - shoulder_mid[1]) * 0.38,
        (shoulder_mid[2] + head[2]) * 0.5,
    )
