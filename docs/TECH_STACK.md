# Technical Stack

## Required for the current scaffold

- Python 3.10+
- Standard library only for the demo pipeline

## Recommended additions

### Pose extraction

- MediaPipe Pose for quick 2D/weak-3D baselines.
- MMPose for configurable 2D/3D pose pipelines.
- WHAM, 4D-Humans, HybrIK, or VIBE for SMPL-like body reconstruction from video.

Preferred order for this repo:

1. MediaPipe Pose Landmarker for single-person baseline extraction from local videos.
2. MMPose RTMPose or MotionBERT when higher-quality 2D/3D inference is needed.
3. WHAM or 4D-Humans when we need temporally coherent 3D body motion instead of sparse joints.

### Motion modeling

- AI4AnimationPy for Python-native motion import, processing, training, and rendering.
- NumPy/PyTorch for sequence models and latent motion priors.
- BVH/FBX/GLB converters for mocap ingestion.

### Task and assistance modeling

- Behavior trees or finite-state machines for overhead work phases.
- Gymnasium + Stable-Baselines3 or skrl for assistance policy learning.
- MuJoCo or Isaac Sim if contact force and robot dynamics become important.

### Exolimb robot layer

- URDF/MJCF model for the wearable limb.
- Pinocchio, MuJoCo, or Mink for kinematics.
- MoveIt 2, OMPL, or cuRobo for motion planning when a full robot stack is needed.

## Near-term implementation order

1. Collect or select overhead-work videos.
2. Run pose extraction into JSONL.
3. Detect overhead segments and hand occupancy phases.
4. Generate exolimb support events.
5. Add an exolimb kinematic model and collision checks.
6. Train assistance timing and support-point policies.
