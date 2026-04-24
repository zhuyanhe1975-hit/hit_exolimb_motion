from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .adapters.ai4animationpy import export_npz_if_numpy_available
from .adapters.bvh import import_bvh_motion
from .assist import plan_support_events
from .fusion import fuse_body_with_hand_overlays
from .datasets import create_dataset_layout
from .demo import generate_overhead_demo
from .io import read_pose_jsonl, write_json, write_pose_jsonl
from .overhead import OverheadConfig, detect_overhead_segments
from .pose_extractors import extract_pose_video_mediapipe, import_mmpose_predictions
from .splitview import OVERHEAD_INSET_RECT, crop_video_region
from .stabilize import stabilize_pose_frames
from .video_motion import generate_overhead_panel_motion, probe_video_duration, video_overhead_segments


def _generate_demo(args: argparse.Namespace) -> None:
    frames = generate_overhead_demo(duration_s=args.duration, fps=args.fps)
    write_pose_jsonl(Path(args.out), frames)


def _detect_overhead(args: argparse.Namespace) -> None:
    frames = read_pose_jsonl(Path(args.input))
    segments = detect_overhead_segments(
        frames,
        OverheadConfig(
            wrist_above_head_margin_m=args.head_margin,
            wrist_above_shoulder_margin_m=args.shoulder_margin,
            min_duration_s=args.min_duration,
        ),
    )
    write_json(Path(args.out), {"segments": segments})


def _plan_assist(args: argparse.Namespace) -> None:
    payload = __import__("json").loads(Path(args.segments).read_text(encoding="utf-8"))
    events = plan_support_events(
        payload.get("segments", []),
        support_lead_time_s=args.lead_time,
        release_delay_s=args.release_delay,
    )
    write_json(Path(args.out), {"events": events})


def _export_npz(args: argparse.Namespace) -> None:
    frames = read_pose_jsonl(Path(args.input))
    exported = export_npz_if_numpy_available(Path(args.out), frames)
    if not exported:
        raise SystemExit("NumPy is not installed; install numpy or keep JSONL output.")


def _import_bvh(args: argparse.Namespace) -> None:
    frames = import_bvh_motion(
        Path(args.input),
        scale_m=args.scale,
        pelvis_height_m=args.pelvis_height,
        lock_root=not args.keep_root_translation,
    )
    write_pose_jsonl(Path(args.out), frames)


def _stabilize_pose(args: argparse.Namespace) -> None:
    frames = read_pose_jsonl(Path(args.input))
    stabilized = stabilize_pose_frames(frames)
    write_pose_jsonl(Path(args.out), stabilized)


def _init_dataset(args: argparse.Namespace) -> None:
    paths = create_dataset_layout(Path(args.root), args.name)
    write_json(Path(args.out), {"dataset": args.name, "paths": [str(path) for path in paths]})


def _generate_video_motion(args: argparse.Namespace) -> None:
    video = Path(args.video)
    frames = generate_overhead_panel_motion(video, fps=args.fps)
    write_pose_jsonl(Path(args.out), frames)
    duration = probe_video_duration(video)
    write_json(Path(args.segments_out), {"segments": video_overhead_segments(duration)})


def _extract_mediapipe_pose(args: argparse.Namespace) -> None:
    frames = extract_pose_video_mediapipe(
        Path(args.video),
        model=Path(args.model) if args.model else None,
        hand_model=Path(args.hand_model) if args.hand_model else None,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        include_hands=args.include_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_hand_tracking_confidence=args.min_hand_tracking_confidence,
    )
    write_pose_jsonl(Path(args.out), frames)


def _import_mmpose_predictions(args: argparse.Namespace) -> None:
    frames = import_mmpose_predictions(
        Path(args.input),
        fps=args.fps,
        person_index=args.person_index,
        score_threshold=args.score_threshold,
        body_height_m=args.body_height,
    )
    write_pose_jsonl(Path(args.out), frames)


def _fuse_pose_tracks(args: argparse.Namespace) -> None:
    body_frames = read_pose_jsonl(Path(args.body))
    overlay_frames = read_pose_jsonl(Path(args.overlay))
    fused = fuse_body_with_hand_overlays(
        body_frames,
        overlay_frames,
        hand_blend=args.hand_blend,
        smooth_alpha=args.smooth_alpha,
    )
    write_pose_jsonl(Path(args.out), fused)


def _analyze_overhead_splitview_hands(args: argparse.Namespace) -> None:
    video = Path(args.video)
    body = Path(args.body)
    inset_video = Path(args.inset_video)
    overlay_out = Path(args.overlay_out)
    fused_out = Path(args.fused_out)
    segments_out = Path(args.segments_out)
    assist_out = Path(args.assist_out)

    crop_video_region(video, inset_video, OVERHEAD_INSET_RECT)
    overlay_frames = extract_pose_video_mediapipe(
        inset_video,
        model=Path(args.model) if args.model else None,
        hand_model=Path(args.hand_model) if args.hand_model else None,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        include_hands=True,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_hand_tracking_confidence=args.min_hand_tracking_confidence,
    )
    write_pose_jsonl(overlay_out, overlay_frames)

    fused = fuse_body_with_hand_overlays(
        read_pose_jsonl(body),
        overlay_frames,
        hand_blend=args.hand_blend,
        smooth_alpha=args.smooth_alpha,
    )
    write_pose_jsonl(fused_out, fused)

    segments = detect_overhead_segments(
        fused,
        OverheadConfig(
            wrist_above_head_margin_m=args.head_margin,
            wrist_above_shoulder_margin_m=args.shoulder_margin,
            min_duration_s=args.min_duration,
        ),
    )
    write_json(segments_out, {"segments": segments})
    events = plan_support_events(
        segments,
        support_lead_time_s=args.lead_time,
        release_delay_s=args.release_delay,
    )
    write_json(assist_out, {"events": events})


def _extract_apple_vision_pose(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "apple_vision_pose.swift"
    command = [
        "swift",
        str(script),
        "--video",
        args.video,
        "--out",
        args.out,
        "--confidence",
        str(args.confidence),
    ]
    subprocess.run(command, check=True)


def _analyze_apple_vision_video(args: argparse.Namespace) -> None:
    pose_out = Path(args.pose_out)
    segments_out = Path(args.segments_out)
    assist_out = Path(args.assist_out)
    stabilized_out = Path(args.stabilized_out) if args.stabilized_out else None

    _extract_apple_vision_pose(
        argparse.Namespace(
            video=args.video,
            out=str(pose_out),
            confidence=args.confidence,
        )
    )

    frames = read_pose_jsonl(pose_out)
    if stabilized_out is not None:
        frames = stabilize_pose_frames(frames)
        write_pose_jsonl(stabilized_out, frames)
    segments = detect_overhead_segments(
        frames,
        OverheadConfig(
            wrist_above_head_margin_m=args.head_margin,
            wrist_above_shoulder_margin_m=args.shoulder_margin,
            min_duration_s=args.min_duration,
        ),
    )
    write_json(segments_out, {"segments": segments})

    events = plan_support_events(
        segments,
        support_lead_time_s=args.lead_time,
        release_delay_s=args.release_delay,
    )
    write_json(assist_out, {"events": events})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hit-exolimb-motion")
    sub = parser.add_subparsers(required=True)

    demo = sub.add_parser("generate-demo", help="Generate a synthetic overhead-work pose sequence.")
    demo.add_argument("--out", required=True)
    demo.add_argument("--duration", type=float, default=4.0)
    demo.add_argument("--fps", type=int, default=30)
    demo.set_defaults(func=_generate_demo)

    detect = sub.add_parser("detect-overhead", help="Detect overhead-work segments from pose JSONL.")
    detect.add_argument("--input", required=True)
    detect.add_argument("--out", required=True)
    detect.add_argument("--head-margin", type=float, default=0.05)
    detect.add_argument("--shoulder-margin", type=float, default=0.25)
    detect.add_argument("--min-duration", type=float, default=0.4)
    detect.set_defaults(func=_detect_overhead)

    plan = sub.add_parser("plan-assist", help="Generate exolimb support events for overhead segments.")
    plan.add_argument("--segments", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--lead-time", type=float, default=0.2)
    plan.add_argument("--release-delay", type=float, default=0.3)
    plan.set_defaults(func=_plan_assist)

    export = sub.add_parser("export-npz", help="Export a minimal NPZ motion file if NumPy is available.")
    export.add_argument("--input", required=True)
    export.add_argument("--out", required=True)
    export.set_defaults(func=_export_npz)

    bvh_import = sub.add_parser(
        "import-bvh-motion",
        help="Convert a BVH mocap file into this repo's pose JSONL format.",
    )
    bvh_import.add_argument("--input", required=True)
    bvh_import.add_argument("--out", required=True)
    bvh_import.add_argument("--scale", type=float, default=0.01)
    bvh_import.add_argument("--pelvis-height", type=float, default=0.95)
    bvh_import.add_argument(
        "--keep-root-translation",
        action="store_true",
        help="Keep BVH root motion instead of locking the pelvis in place.",
    )
    bvh_import.set_defaults(func=_import_bvh)

    stabilize = sub.add_parser("stabilize-pose", help="Apply temporal smoothing and fixed bone-length kinematic constraints to pose JSONL.")
    stabilize.add_argument("--input", required=True)
    stabilize.add_argument("--out", required=True)
    stabilize.set_defaults(func=_stabilize_pose)

    dataset = sub.add_parser("init-dataset", help="Create local directories for an external video dataset.")
    dataset.add_argument("--name", required=True)
    dataset.add_argument("--root", default=".")
    dataset.add_argument("--out", default="outputs/dataset_layout.json")
    dataset.set_defaults(func=_init_dataset)

    video = sub.add_parser(
        "generate-video-motion",
        help="Generate an AI4-compatible overhead-work pose sequence from the local reference video phases.",
    )
    video.add_argument("--video", required=True)
    video.add_argument("--out", required=True)
    video.add_argument("--segments-out", required=True)
    video.add_argument("--fps", type=int, default=30)
    video.set_defaults(func=_generate_video_motion)

    mediapipe_pose = sub.add_parser(
        "extract-mediapipe-pose",
        help="Extract pose tracks from a video using the official MediaPipe Pose Landmarker.",
    )
    mediapipe_pose.add_argument("--video", required=True)
    mediapipe_pose.add_argument("--out", required=True)
    mediapipe_pose.add_argument("--model", help="Path to a MediaPipe pose_landmarker .task bundle.")
    mediapipe_pose.add_argument("--hand-model", help="Path to a MediaPipe hand_landmarker .task bundle.")
    mediapipe_pose.add_argument("--min-detection-confidence", type=float, default=0.5)
    mediapipe_pose.add_argument("--min-presence-confidence", type=float, default=0.5)
    mediapipe_pose.add_argument("--min-tracking-confidence", type=float, default=0.5)
    mediapipe_pose.add_argument("--include-hands", action="store_true")
    mediapipe_pose.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    mediapipe_pose.add_argument("--min-hand-presence-confidence", type=float, default=0.3)
    mediapipe_pose.add_argument("--min-hand-tracking-confidence", type=float, default=0.3)
    mediapipe_pose.set_defaults(func=_extract_mediapipe_pose)

    mmpose_import = sub.add_parser(
        "import-mmpose-predictions",
        help="Convert official MMPose inferencer JSON predictions into this repo's pose JSONL format.",
    )
    mmpose_import.add_argument("--input", required=True)
    mmpose_import.add_argument("--out", required=True)
    mmpose_import.add_argument("--fps", type=float, required=True)
    mmpose_import.add_argument("--person-index", type=int, default=-1)
    mmpose_import.add_argument("--score-threshold", type=float, default=0.15)
    mmpose_import.add_argument("--body-height", type=float, default=1.42)
    mmpose_import.set_defaults(func=_import_mmpose_predictions)

    fuse_tracks = sub.add_parser(
        "fuse-pose-tracks",
        help="Fuse a stable body track with an overlay track, keeping the body and replacing arm/hand detail where available.",
    )
    fuse_tracks.add_argument("--body", required=True)
    fuse_tracks.add_argument("--overlay", required=True)
    fuse_tracks.add_argument("--out", required=True)
    fuse_tracks.add_argument("--hand-blend", type=float, default=0.85)
    fuse_tracks.add_argument("--smooth-alpha", type=float, default=0.55)
    fuse_tracks.set_defaults(func=_fuse_pose_tracks)

    splitview = sub.add_parser(
        "analyze-overhead-splitview-hands",
        help="Crop the lower-left inset view from overhead.mp4, extract MediaPipe hand detail there, and fuse it onto a stable body track.",
    )
    splitview.add_argument("--video", required=True)
    splitview.add_argument("--body", required=True)
    splitview.add_argument("--model", required=True)
    splitview.add_argument("--hand-model", required=True)
    splitview.add_argument("--inset-video", required=True)
    splitview.add_argument("--overlay-out", required=True)
    splitview.add_argument("--fused-out", required=True)
    splitview.add_argument("--segments-out", required=True)
    splitview.add_argument("--assist-out", required=True)
    splitview.add_argument("--min-detection-confidence", type=float, default=0.5)
    splitview.add_argument("--min-presence-confidence", type=float, default=0.5)
    splitview.add_argument("--min-tracking-confidence", type=float, default=0.5)
    splitview.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    splitview.add_argument("--min-hand-presence-confidence", type=float, default=0.3)
    splitview.add_argument("--min-hand-tracking-confidence", type=float, default=0.3)
    splitview.add_argument("--hand-blend", type=float, default=0.9)
    splitview.add_argument("--smooth-alpha", type=float, default=0.6)
    splitview.add_argument("--head-margin", type=float, default=-0.05)
    splitview.add_argument("--shoulder-margin", type=float, default=0.1)
    splitview.add_argument("--min-duration", type=float, default=0.4)
    splitview.add_argument("--lead-time", type=float, default=0.2)
    splitview.add_argument("--release-delay", type=float, default=0.3)
    splitview.set_defaults(func=_analyze_overhead_splitview_hands)

    apple_vision = sub.add_parser(
        "extract-apple-vision-pose",
        help="Extract pose tracks from a local video using Apple's Vision framework on macOS.",
    )
    apple_vision.add_argument("--video", required=True)
    apple_vision.add_argument("--out", required=True)
    apple_vision.add_argument("--confidence", type=float, default=0.0)
    apple_vision.set_defaults(func=_extract_apple_vision_pose)

    apple_vision_pipeline = sub.add_parser(
        "analyze-apple-vision-video",
        help="Run Apple's Vision extractor on a local MP4, then detect overhead segments and exolimb events.",
    )
    apple_vision_pipeline.add_argument("--video", required=True)
    apple_vision_pipeline.add_argument("--pose-out", required=True)
    apple_vision_pipeline.add_argument("--stabilized-out")
    apple_vision_pipeline.add_argument("--segments-out", required=True)
    apple_vision_pipeline.add_argument("--assist-out", required=True)
    apple_vision_pipeline.add_argument("--confidence", type=float, default=0.0)
    apple_vision_pipeline.add_argument("--head-margin", type=float, default=-0.05)
    apple_vision_pipeline.add_argument("--shoulder-margin", type=float, default=0.1)
    apple_vision_pipeline.add_argument("--min-duration", type=float, default=0.4)
    apple_vision_pipeline.add_argument("--lead-time", type=float, default=0.2)
    apple_vision_pipeline.add_argument("--release-delay", type=float, default=0.3)
    apple_vision_pipeline.set_defaults(func=_analyze_apple_vision_video)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
