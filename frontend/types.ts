export type Vec3 = [number, number, number];

export type PoseFrame = {
  frame: number;
  time: number;
  joints: Record<string, Vec3>;
};

export type OverheadSegment = {
  label: string;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  duration: number;
  side: string;
};

export type AssistEvent = {
  event: string;
  segment_index: number;
  time: number;
  target: string;
  side: string;
};

