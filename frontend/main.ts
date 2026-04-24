import "./styles.css";
import { loadAssistEvents, loadPoseJsonl, loadSegments } from "./data";
import { probeMujocoWasm } from "./mujoco";
import type { AssistEvent, OverheadSegment, PoseFrame } from "./types";
import { MotionViewer } from "./viewer";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app");

const DEFAULT_POSE_URL = "/data/overhead_video/pose.jsonl";
const DEFAULT_SEGMENTS_URL = "/outputs/overhead_video_segments.json";
const DEFAULT_ASSIST_URL = "/outputs/overhead_video_assist_plan.json";
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

const viewer = new MotionViewer(viewerRoot);
let userSeeking = false;

viewer.setFrameCallback((frame, activeSegment) => {
  timeReadout.textContent = `${frame.time.toFixed(2)}s`;
  phaseReadout.textContent = activeSegment ? activeSegment.label : "neutral";
  if (Number.isFinite(referenceVideo.duration)) {
    const target = Math.min(frame.time, referenceVideo.duration);
    if (Math.abs(referenceVideo.currentTime - target) > 0.18) {
      referenceVideo.currentTime = target;
    }
  }
  if (!userSeeking && viewer.getDuration() > 0) {
    timeline.value = String(Math.round((viewer.getTime() / viewer.getDuration()) * 1000));
  }
});

playToggle.addEventListener("click", () => {
  viewer.setPlaying(!viewer.isPlaying());
  playToggle.textContent = viewer.isPlaying() ? "Pause" : "Play";
  if (viewer.isPlaying()) {
    void referenceVideo.play();
  } else {
    referenceVideo.pause();
  }
});

resetView.addEventListener("click", () => {
  viewer.seek01(0);
  referenceVideo.currentTime = 0;
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
  viewer.seek01(value);
  if (Number.isFinite(referenceVideo.duration)) {
    referenceVideo.currentTime = referenceVideo.duration * value;
  }
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

async function boot() {
  const [frames, segments, events] = await Promise.all([
    loadPoseJsonl(DEFAULT_POSE_URL),
    loadSegments(DEFAULT_SEGMENTS_URL),
    loadAssistEvents(DEFAULT_ASSIST_URL),
  ]);
  viewer.setData(frames, segments, events);
  frameCount.textContent = String(frames.length);
  segmentCount.textContent = String(segments.length);
  renderEvents(events);
  referenceVideo.playbackRate = Number(speed.value);
  void referenceVideo.play();
  await updateMujocoStatus();
}

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
