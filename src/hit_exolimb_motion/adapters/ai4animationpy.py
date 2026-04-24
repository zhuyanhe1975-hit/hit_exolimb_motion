from __future__ import annotations

from pathlib import Path

from ..skeleton import PoseFrame


def export_npz_if_numpy_available(path: Path, frames: list[PoseFrame]) -> bool:
    """Export a minimal AI4AnimationPy-style NPZ when NumPy is installed.

    Returns False instead of failing when NumPy is unavailable. The exact
    production schema should be aligned once AI4AnimationPy is added as a real
    dependency.
    """
    try:
        import numpy as np
    except ImportError:
        return False

    joint_names = sorted({name for frame in frames for name in frame.joints})
    positions = np.array(
        [[frame.joints[name] for name in joint_names] for frame in frames],
        dtype=np.float32,
    )
    times = np.array([frame.time for frame in frames], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, joint_names=np.array(joint_names), times=times, positions=positions)
    return True

