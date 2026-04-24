import "./styles.css";
import { loadAssistEvents, loadPoseJsonl, loadSegments } from "./data";
import { analyzeMp4UpperBody } from "./mediapipeVideo";
import { probeMujocoWasm } from "./mujoco";
import type { AssistEvent, OverheadSegment, PoseFrame } from "./types";
import { MotionViewer } from "./viewer";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app");

const DEFAULT_POSE_URL = "/data/overhead_video/pose_rokoko.jsonl";
const DEFAULT_SEGMENTS_URL = "/outputs/overhead_video_segments_rokoko.json";
const DEFAULT_ASSIST_URL = "/outputs/overhead_video_assist_plan_rokoko.json";
const DEFAULT_VIDEO_URL = "/assets/videos/overhead.mp4";

app.innerHTML = `
  <main class="app">
    <section class="viewport">
      <div class="overlay-title">
        <strong>HIT Exolimb Motion Viewer</strong>
        <span>Human overhead work sequence with exolimb assistance events</span>
      </div>
      <div id="viewer-root" style="width: 100%; height: 100%"></div>
    </section>
    <aside class="panel">
      <section class="section">
        <h2>Offline MP4 Source</h2>
        <div class="controls">
          <button id="analyze-mp4" class="button primary" type="button">Analyze MP4 In Browser</button>
        </div>
        <div id="analysis-status" class="status">Using local Rokoko Vision body motion with exported hand skeleton by default. Browser MediaPipe analysis remains optional and experimental.</div>
      </section>

      <section class="section">
        <h2>Playback</h2>
        <div class="controls">
          <button id="play-toggle" class="button primary" type="button">Pause</button>
          <button id="reset-view" class="button" type="button">Restart</button>
        </div>
        <div class="timeline">
          <input id="timeline" type="range" min="0" max="1000" value="0" />
          <div class="stats">
            <div class="stat"><span>Time</span><strong id="time-readout">0.00s</strong></div>
            <div class="stat"><span>Phase</span><strong id="phase-readout">loading</strong></div>
          </div>
        </div>
        <div class="field">
          <label for="speed">Playback speed</label>
          <select id="speed">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
          </select>
        </div>
      </section>

      <section class="section">
        <h2>Reference Video</h2>
        <video id="reference-video" src="${DEFAULT_VIDEO_URL}" muted loop playsinline controls style="width: 100%; border-radius: 8px; border: 1px solid #2d3943; background: #05070a"></video>
      </section>

      <section class="section">
        <h2>Sequence</h2>
        <div class="stats">
          <div class="stat"><span>Frames</span><strong id="frame-count">0</strong></div>
          <div class="stat"><span>Overhead Segments</span><strong id="segment-count">0</strong></div>
        </div>
      </section>

      <section class="section">
        <h2>Exolimb Events</h2>
        <div id="event-list" class="event-list"></div>
      </section>

      <section class="section">
        <h2>MuJoCo WASM</h2>
        <div id="mujoco-status" class="status">Checking @mujoco/mujoco...</div>
      </section>
    </aside>
  </main>
`;

function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`UI initialization failed: ${selector}`);
  return element;
}

const viewerRoot = requireElement<HTMLDivElement>("#viewer-root");
const playToggle = requireElement<HTMLButtonElement>("#play-toggle");
const resetView = requireElement<HTMLButtonElement>("#reset-view");
const timeline = requireElement<HTMLInputElement>("#timeline");
const speed = requireElement<HTMLSelectElement>("#speed");
const timeReadout = requireElement<HTMLElement>("#time-readout");
const phaseReadout = requireElement<HTMLElement>("#phase-readout");
const frameCount = requireElement<HTMLElement>("#frame-count");
const segmentCount = requireElement<HTMLElement>("#segment-count");
const eventList = requireElement<HTMLDivElement>("#event-list");
const mujocoStatus = requireElement<HTMLDivElement>("#mujoco-status");
const referenceVideo = requireElement<HTMLVideoElement>("#reference-video");
const analyzeMp4 = requireElement<HTMLButtonElement>("#analyze-mp4");
const analysisStatus = requireElement<HTMLDivElement>("#analysis-status");

const viewer = new MotionViewer(viewerRoot);
let userSeeking = false;
let syncRaf = 0;

function syncViewerToVideo() {
  viewer.setTime(referenceVideo.currentTime);
  if (!userSeeking && viewer.getDuration() > 0) {
    timeline.value = String(Math.round((viewer.getTime() / viewer.getDuration()) * 1000));
  }
}

function stopVideoSyncLoop() {
  if (syncRaf) {
    cancelAnimationFrame(syncRaf);
    syncRaf = 0;
  }
}

function startVideoSyncLoop() {
  stopVideoSyncLoop();
  const tick = () => {
    syncViewerToVideo();
    if (!referenceVideo.paused && !referenceVideo.ended) {
      syncRaf = requestAnimationFrame(tick);
    } else {
      syncRaf = 0;
    }
  };
  syncRaf = requestAnimationFrame(tick);
}

viewer.setFrameCallback((frame, activeSegment) => {
  timeReadout.textContent = `${frame.time.toFixed(2)}s`;
  phaseReadout.textContent = activeSegment ? activeSegment.label : "neutral";
});

playToggle.addEventListener("click", () => {
  viewer.setPlaying(!viewer.isPlaying());
  playToggle.textContent = viewer.isPlaying() ? "Pause" : "Play";
  if (viewer.isPlaying()) {
    void referenceVideo.play();
  } else {
    stopVideoSyncLoop();
    referenceVideo.pause();
  }
});

resetView.addEventListener("click", () => {
  referenceVideo.currentTime = 0;
  viewer.setTime(0);
});

speed.addEventListener("change", () => {
  const value = Number(speed.value);
  viewer.setSpeed(value);
  referenceVideo.playbackRate = value;
});

timeline.addEventListener("pointerdown", () => {
  userSeeking = true;
});

timeline.addEventListener("pointerup", () => {
  userSeeking = false;
});

timeline.addEventListener("input", () => {
  const value = Number(timeline.value) / 1000;
  const targetTime = viewer.getDuration() * value;
  referenceVideo.currentTime = targetTime;
  viewer.setTime(targetTime);
});

referenceVideo.addEventListener("play", () => {
  viewer.setPlaying(true);
  playToggle.textContent = "Pause";
  startVideoSyncLoop();
});

referenceVideo.addEventListener("pause", () => {
  viewer.setPlaying(false);
  playToggle.textContent = "Play";
  stopVideoSyncLoop();
  syncViewerToVideo();
});

referenceVideo.addEventListener("seeking", () => {
  syncViewerToVideo();
});

referenceVideo.addEventListener("seeked", () => {
  syncViewerToVideo();
});

referenceVideo.addEventListener("timeupdate", () => {
  if (referenceVideo.paused) {
    syncViewerToVideo();
  }
});

referenceVideo.addEventListener("ratechange", () => {
  const currentRate = referenceVideo.playbackRate;
  viewer.setSpeed(currentRate);
  speed.value = String(currentRate);
});

referenceVideo.addEventListener("loadedmetadata", () => {
  syncViewerToVideo();
});

function renderEvents(events: AssistEvent[]) {
  eventList.innerHTML =
    events
      .map(
        (event) => `
          <article class="event">
            <time>${event.time.toFixed(2)}s</time>
            <div>
              <strong>${event.event}</strong>
              <span>${event.side} · ${event.target}</span>
            </div>
          </article>
        `,
      )
      .join("") || `<div class="status warn">No exolimb assistance events found.</div>`;
}

async function loadFromJsonFallback() {
  const [frames, segments, events] = await Promise.all([
    loadPoseJsonl(DEFAULT_POSE_URL),
    loadSegments(DEFAULT_SEGMENTS_URL),
    loadAssistEvents(DEFAULT_ASSIST_URL),
  ]);
  return { frames, segments, events, source: "json" as const };
}

async function analyzeReferenceMp4() {
  analysisStatus.className = "status";
  analysisStatus.textContent = "Running browser-side MediaPipe analysis on overhead.mp4...";
  analyzeMp4.disabled = true;
  try {
    const result = await analyzeMp4UpperBody(DEFAULT_VIDEO_URL, 30, ({ currentFrame, totalFrames, time }) => {
      analysisStatus.textContent = `Analyzing MP4 frame ${currentFrame}/${totalFrames} at ${time.toFixed(2)}s...`;
    });
    analysisStatus.className = "status ready";
    analysisStatus.textContent = `MediaPipe finished: ${result.frames.length} frames, ${result.segments.length} overhead segments.`;
    return result;
  } finally {
    analyzeMp4.disabled = false;
  }
}

async function applySequence(frames: PoseFrame[], segments: OverheadSegment[], events: AssistEvent[]) {
  viewer.setData(frames, segments, events);
  frameCount.textContent = String(frames.length);
  segmentCount.textContent = String(segments.length);
  renderEvents(events);
  referenceVideo.playbackRate = Number(speed.value);
  syncViewerToVideo();
  void referenceVideo.play();
}

async function boot() {
  try {
    const offline = await loadFromJsonFallback();
    analysisStatus.className = "status ready";
    analysisStatus.textContent = `Loaded local Rokoko Vision motion: ${offline.frames.length} frames, ${offline.segments.length} overhead segments.`;
    await applySequence(offline.frames, offline.segments, offline.events);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    analysisStatus.className = "status warn";
    analysisStatus.textContent = `Failed to load local Rokoko Vision motion: ${message}`;
  }
  await updateMujocoStatus();
}

analyzeMp4.addEventListener("click", async () => {
  try {
    stopVideoSyncLoop();
    referenceVideo.pause();
    const result = await analyzeReferenceMp4();
    await applySequence(result.frames, result.segments, result.events);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    analysisStatus.className = "status warn";
    analysisStatus.textContent = `MediaPipe MP4 analysis failed: ${message}`;
  }
});

async function updateMujocoStatus() {
  const status = await probeMujocoWasm();
  mujocoStatus.classList.toggle("ready", status.available);
  mujocoStatus.classList.toggle("warn", !status.available);
  mujocoStatus.textContent = status.message;
}

boot().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  phaseReadout.textContent = "error";
  eventList.innerHTML = `<div class="status warn">${message}</div>`;
});
