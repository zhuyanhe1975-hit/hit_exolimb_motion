from __future__ import annotations

import argparse
from pathlib import Path

from .adapters.ai4animationpy import export_npz_if_numpy_available
from .assist import plan_support_events
from .datasets import create_dataset_layout
from .demo import generate_overhead_demo
from .io import read_pose_jsonl, write_json, write_pose_jsonl
from .overhead import OverheadConfig, detect_overhead_segments
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


def _init_dataset(args: argparse.Namespace) -> None:
    paths = create_dataset_layout(Path(args.root), args.name)
    write_json(Path(args.out), {"dataset": args.name, "paths": [str(path) for path in paths]})


def _generate_video_motion(args: argparse.Namespace) -> None:
    video = Path(args.video)
    frames = generate_overhead_panel_motion(video, fps=args.fps)
    write_pose_jsonl(Path(args.out), frames)
    duration = probe_video_duration(video)
    write_json(Path(args.segments_out), {"segments": video_overhead_segments(duration)})


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
