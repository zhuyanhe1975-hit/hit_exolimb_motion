import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { AssistEvent, OverheadSegment, PoseFrame, Vec3 } from "./types";

const MEDIAPIPE_WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const POSE_INDEX = {
  nose: 0,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_wrist: 15,
  right_wrist: 16,
  left_hip: 23,
  right_hip: 24,
  left_knee: 25,
  right_knee: 26,
  left_ankle: 27,
  right_ankle: 28,
} as const;

type AnalyzeProgress = {
  currentFrame: number;
  totalFrames: number;
  time: number;
};

let detectorBundlePromise:
  | Promise<{ pose: PoseLandmarker; hands: HandLandmarker }>
  | undefined;

function normalizeError(error: unknown) {
  if (error instanceof Error) return error;
  if (typeof error === "string") return new Error(error);
  if (error && typeof error === "object" && "type" in error) {
    return new Error(`MediaPipe event error: ${String((error as { type?: unknown }).type ?? "unknown")}`);
  }
  return new Error(String(error));
}

async function createDetectorBundle(delegate: "GPU" | "CPU") {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
  const [pose, hands] = await Promise.all([
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    }),
  ]);
  return { pose, hands };
}

function getDetectors() {
  if (!detectorBundlePromise) {
    detectorBundlePromise = (async () => {
      try {
        return await createDetectorBundle("GPU");
      } catch (gpuError) {
        console.warn("MediaPipe GPU delegate failed for MP4 analysis, retrying with CPU.", gpuError);
        return createDetectorBundle("CPU");
      }
    })();
    detectorBundlePromise.catch(() => {
      detectorBundlePromise = undefined;
    });
  }
  return detectorBundlePromise;
}

async function createAnalysisVideo(url: string) {
  const video = document.createElement("video");
  video.src = url;
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  await new Promise<void>((resolve, reject) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
    video.addEventListener(
      "error",
      () => reject(new Error(`Failed to load video for analysis: ${url}`)),
      { once: true },
    );
  });
  return video;
}

async function seekVideo(video: HTMLVideoElement, time: number) {
  if (Math.abs(video.currentTime - time) < 0.0005) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Video seek failed during MediaPipe analysis"));
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = time;
  });
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
}

function mapPoseResultToFrame(
  frame: number,
  time: number,
  pose: PoseLandmarkerResult,
  hands: HandLandmarkerResult,
): PoseFrame | undefined {
  const landmarks = pose.landmarks?.[0];
  if (!landmarks) return undefined;
  const world = pose.worldLandmarks?.[0];

  const leftShoulder2D = landmarks[POSE_INDEX.left_shoulder];
  const rightShoulder2D = landmarks[POSE_INDEX.right_shoulder];
  const leftHip2D = landmarks[POSE_INDEX.left_hip];
  const rightHip2D = landmarks[POSE_INDEX.right_hip];
  const leftAnkle2D = landmarks[POSE_INDEX.left_ankle];
  const rightAnkle2D = landmarks[POSE_INDEX.right_ankle];

  const hipCenterX = (leftHip2D.x + rightHip2D.x) * 0.5;
  const hipCenterY = (leftHip2D.y + rightHip2D.y) * 0.5;
  const shoulderCenterY = (leftShoulder2D.y + rightShoulder2D.y) * 0.5;
  const ankleCenterY = (leftAnkle2D.y + rightAnkle2D.y) * 0.5;
  const shoulderSpan = Math.max(Math.abs(rightShoulder2D.x - leftShoulder2D.x), 0.06);
  const bodyHeight = Math.max(ankleCenterY - shoulderCenterY, shoulderSpan * 3.0, 0.18);
  const scale = 1.42 / bodyHeight;

  const hipWorldZ = world
    ? (world[POSE_INDEX.left_hip].z + world[POSE_INDEX.right_hip].z) * 0.5
    : 0;

  const point3 = (index: number): Vec3 => {
    const lm = landmarks[index];
    const z = world ? (hipWorldZ - world[index].z) * 0.28 : 0;
    return [(hipCenterX - lm.x) * scale, (hipCenterY - lm.y) * scale + 0.95, z];
  };

  const joints: Record<string, Vec3> = {
    pelvis: midpoint(point3(POSE_INDEX.left_hip), point3(POSE_INDEX.right_hip)),
    left_hip: point3(POSE_INDEX.left_hip),
    right_hip: point3(POSE_INDEX.right_hip),
    left_knee: point3(POSE_INDEX.left_knee),
    right_knee: point3(POSE_INDEX.right_knee),
    left_ankle: point3(POSE_INDEX.left_ankle),
    right_ankle: point3(POSE_INDEX.right_ankle),
    left_shoulder: point3(POSE_INDEX.left_shoulder),
    right_shoulder: point3(POSE_INDEX.right_shoulder),
    left_elbow: point3(POSE_INDEX.left_elbow),
    right_elbow: point3(POSE_INDEX.right_elbow),
    left_wrist: point3(POSE_INDEX.left_wrist),
    right_wrist: point3(POSE_INDEX.right_wrist),
  };

  const shoulderMid = midpoint(joints.left_shoulder, joints.right_shoulder);
  joints.spine = midpoint(joints.pelvis, shoulderMid);
  const nose = point3(POSE_INDEX.nose);
  joints.head = [nose[0], Math.max(nose[1] + 0.1, shoulderMid[1] + 0.15), shoulderMid[2] + 0.02];
  joints.neck = [
    (shoulderMid[0] + joints.head[0]) * 0.5,
    shoulderMid[1] + (joints.head[1] - shoulderMid[1]) * 0.38,
    (shoulderMid[2] + joints.head[2]) * 0.5,
  ];

  if (hands.landmarks && hands.handedness) {
    hands.landmarks.forEach((handLandmarks, index) => {
      const handedness = hands.handedness?.[index]?.[0]?.categoryName?.toLowerCase();
      if (handedness !== "left" && handedness !== "right") return;
      const palmIndices = [0, 5, 9, 13, 17];
      const tipIndices = [8, 12, 16, 20];
      const avg = (indices: number[]): Vec3 => {
        const total = indices.reduce(
          (sum, current) => {
            const point = handLandmarks[current];
            return {
              x: sum.x + point.x,
              y: sum.y + point.y,
              z: sum.z + point.z,
            };
          },
          { x: 0, y: 0, z: 0 },
        );
        const n = indices.length;
        return [
          (hipCenterX - total.x / n) * scale,
          (hipCenterY - total.y / n) * scale + 0.95,
          ((0 - total.z / n) * 0.18),
        ];
      };
      joints[`${handedness}_wrist`] = avg([0]);
      joints[`${handedness}_hand`] = avg(tipIndices);
    });
  }

  return { frame, time, joints };
}

export async function analyzeMp4UpperBody(
  videoUrl: string,
  fps = 30,
  onProgress?: (progress: AnalyzeProgress) => void,
) {
  const [{ pose, hands }, video] = await Promise.all([getDetectors(), createAnalysisVideo(videoUrl)]);
  const totalFrames = Math.max(1, Math.round(video.duration * fps));
  const frames: PoseFrame[] = [];

  try {
    for (let index = 0; index < totalFrames; index += 1) {
      const time = Math.min(index / fps, Math.max(video.duration - 1 / fps, 0));
      await seekVideo(video, time);
      const timestampMs = Math.round(time * 1000);
      const poseResult = pose.detectForVideo(video, timestampMs);
      const handResult = hands.detectForVideo(video, timestampMs);
      const frame = mapPoseResultToFrame(index, time, poseResult, handResult);
      if (frame) frames.push(frame);
      onProgress?.({ currentFrame: index + 1, totalFrames, time });
    }
  } catch (error) {
    throw normalizeError(error);
  } finally {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }

  return {
    frames,
    segments: detectOverheadSegments(frames),
    events: planSupportEvents(detectOverheadSegments(frames)),
  };
}

function classifyOverhead(frame: PoseFrame) {
  const headY = frame.joints.head?.[1] ?? 0;
  const shoulderY = ((frame.joints.left_shoulder?.[1] ?? 0) + (frame.joints.right_shoulder?.[1] ?? 0)) * 0.5;
  const leftWristY = frame.joints.left_wrist?.[1] ?? -Infinity;
  const rightWristY = frame.joints.right_wrist?.[1] ?? -Infinity;
  const active = (wristY: number) => wristY >= headY && wristY >= shoulderY + 0.12;
  const leftActive = active(leftWristY);
  const rightActive = active(rightWristY);
  return {
    frame: frame.frame,
    time: frame.time,
    leftActive,
    rightActive,
    active: leftActive || rightActive,
    side: leftActive && rightActive ? "both" : leftActive ? "left" : rightActive ? "right" : "none",
  };
}

function detectOverheadSegments(frames: PoseFrame[]): OverheadSegment[] {
  const states = frames.map(classifyOverhead);
  const segments: OverheadSegment[] = [];
  let start = -1;
  let sides = new Set<string>();

  const closeSegment = (startIndex: number, endIndex: number) => {
    const first = states[startIndex];
    const last = states[endIndex];
    const duration = last.time - first.time;
    if (duration < 0.25) return;
    segments.push({
      label: "overhead_work",
      start_frame: first.frame,
      end_frame: last.frame,
      start_time: first.time,
      end_time: last.time,
      duration,
      side: sides.has("both") || sides.size > 1 ? "both" : [...sides][0] ?? "unknown",
    });
  };

  states.forEach((state, index) => {
    if (state.active) {
      if (start < 0) {
        start = index;
        sides = new Set<string>();
      }
      sides.add(state.side);
      return;
    }
    if (start >= 0) {
      closeSegment(start, index - 1);
      start = -1;
      sides = new Set<string>();
    }
  });

  if (start >= 0) closeSegment(start, states.length - 1);
  return segments;
}

function planSupportEvents(segments: OverheadSegment[]): AssistEvent[] {
  return segments.flatMap((segment, index) => [
    {
      event: "prepare_support",
      segment_index: index,
      time: Math.max(0, segment.start_time - 0.2),
      target: "overhead_tool_or_workpiece",
      side: segment.side,
    },
    {
      event: "hold_support",
      segment_index: index,
      time: segment.start_time,
      target: "overhead_tool_or_workpiece",
      side: segment.side,
    },
    {
      event: "release_support",
      segment_index: index,
      time: segment.end_time + 0.3,
      target: "overhead_tool_or_workpiece",
      side: segment.side,
    },
  ]);
}
