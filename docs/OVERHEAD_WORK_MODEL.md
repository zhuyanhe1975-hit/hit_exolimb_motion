# Overhead Work Motion Model

## Goal

Represent a human performing overhead work without Unity and convert the motion into exolimb assistance requirements.

## Human State

The first model uses a sparse kinematic body:

- head
- shoulders
- elbows
- wrists
- pelvis/spine when available
- lower body joints when available

For overhead work, the minimal signal is whether one or both wrists remain above the head and shoulders for a sustained interval.

## Work Phases

Initial phase taxonomy:

- `approach`: person moves toward the workstation.
- `reach_overhead`: one or both hands move above shoulder/head height.
- `overhead_work`: sustained overhead operation.
- `support_needed`: external support is useful for the tool, workpiece, or arm posture.
- `release`: overhead operation ends and the exolimb should clear the workspace.

## Exolimb Assistance Events

The assistance planner currently generates:

- `prepare_support`: move the exolimb before human overhead motion becomes sustained.
- `hold_support`: maintain support during the overhead segment.
- `release_support`: clear the support after the segment ends.

Later versions should add:

- support point selection
- tool/workpiece tracking
- collision checks against the human body
- comfort and ergonomic constraints
- force or impedance targets

## Data Flow

```text
video
  -> pose extractor
  -> pose JSONL
  -> overhead segment detector
  -> assistance event planner
  -> exolimb simulator or policy
```

## Why This Shape

The representation keeps the AI4Animation advantage: it treats motion as a temporal sequence and leaves room for learned motion priors. It avoids assuming a specific simulator until the exolimb model and task objective are stable.

