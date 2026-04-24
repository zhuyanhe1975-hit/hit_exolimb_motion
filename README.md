# HIT Exolimb Motion

Python-native research scaffold for overhead human work motion modeling and wearable exolimb assistance.

The project deliberately avoids Unity. It treats AI4Animation as a motion-modeling reference and targets the newer Python-native direction:

- video or mocap input
- 2D/3D human pose extraction
- overhead-work phase detection
- motion sequence generation
- exolimb assistance events such as support, hold, lift, and release

## Scope

This repository is not a physics simulator yet. It is the motion layer for an exolimb research stack:

```text
real operation video / mocap
        -> pose tracks
        -> overhead work segments
        -> human motion sequence
        -> exolimb assistance plan
        -> downstream simulation or robot policy
```

## Suggested Data Sources

Use only datasets and videos whose license permits research use.

- HA-ViD: human assembly video dataset with assembly taxonomy and annotations.
- Assembly101: multi-view procedural assembly dataset with hand-object interactions.
- OpenPack: logistics/packing work dataset with operation annotations.
- InHARD: industrial human action recognition dataset.
- Your own recordings: preferred for overhead work because public datasets rarely focus on sustained above-head tasks.

## Quick Start

```bash
cd ~/myworks/hit_exolimb_motion
python -m hit_exolimb_motion.cli generate-demo --out data/demo/overhead_pose.jsonl
python -m hit_exolimb_motion.cli detect-overhead --input data/demo/overhead_pose.jsonl --out outputs/demo_segments.json
python -m hit_exolimb_motion.cli plan-assist --segments outputs/demo_segments.json --out outputs/demo_assist_plan.json
```

## Frontend Viewer

The browser viewer is Python-free at runtime and visualizes the JSONL motion stream with Three.js. It also installs the official `@mujoco/mujoco` WebAssembly package so MJCF-based human or exolimb models can be integrated next.

```bash
cd ~/myworks/hit_exolimb_motion
npm install
npm run dev
```

Open the Vite URL and load the default demo sequence from:

- `data/demo/overhead_pose.jsonl`
- `outputs/demo_segments.json`
- `outputs/demo_assist_plan.json`

For local development:

```bash
python -m pip install -e .
hit-exolimb-motion generate-demo --out data/demo/overhead_pose.jsonl
```

## Pose Input Format

The initial pipeline expects JSONL, one frame per line:

```json
{"frame": 0, "time": 0.0, "joints": {"head": [0, 1.65, 0], "left_wrist": [-0.25, 1.95, 0.1]}}
```

Required joints for overhead detection:

- `head`
- `left_shoulder`, `right_shoulder`
- `left_elbow`, `right_elbow`
- `left_wrist`, `right_wrist`

Optional joints improve downstream modeling:

- `pelvis`, `spine`, `neck`
- `left_hip`, `right_hip`
- `left_knee`, `right_knee`
- `left_ankle`, `right_ankle`

## Recommended Pose Extractors

This scaffold does not vendor heavy pose models. Add one of these behind `src/hit_exolimb_motion/pose_extractors/`:

- MediaPipe Pose: fast baseline for 2D/weak 3D.
- MMPose: strong 2D/3D pose toolbox.
- WHAM / 4D-Humans / HybrIK: SMPL-style 3D body reconstruction.
- OpenPose: useful if legacy 2D keypoint JSON already exists.

## AI4Animation / AI4AnimationPy Integration

Use AI4AnimationPy as the motion processing backend when available:

- import BVH/FBX/GLB mocap
- store motion as `.npz`
- train or run neural motion models
- provide motion priors and smoothing for overhead-work sequences

This scaffold keeps the adapter boundary explicit so the project remains runnable without AI4AnimationPy installed.

## Project Layout

```text
configs/                    pipeline defaults
data/                       local datasets and generated demo data
outputs/                    generated segments and assistance plans
src/hit_exolimb_motion/     Python package
```
