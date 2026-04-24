import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { AI4BluemanAvatar } from "./ai4Avatar";
import type { AssistEvent, OverheadSegment, PoseFrame } from "./types";

const BONE_PAIRS = [
  ["pelvis", "spine"],
  ["spine", "neck"],
  ["neck", "head"],
  ["neck", "left_shoulder"],
  ["neck", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["pelvis", "left_hip"],
  ["pelvis", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
] as const;

const JOINT_NAMES = Array.from(new Set(BONE_PAIRS.flat()));
const Y_AXIS = new THREE.Vector3(0, 1, 0);

type CapsuleSpec = {
  a: string;
  b: string;
  radius: number;
};

const PRIMITIVE_CAPSULES: CapsuleSpec[] = [
  { a: "pelvis", b: "spine", radius: 0.155 },
  { a: "spine", b: "neck", radius: 0.135 },
  { a: "neck", b: "left_shoulder", radius: 0.07 },
  { a: "neck", b: "right_shoulder", radius: 0.07 },
  { a: "left_shoulder", b: "left_elbow", radius: 0.062 },
  { a: "left_elbow", b: "left_wrist", radius: 0.052 },
  { a: "right_shoulder", b: "right_elbow", radius: 0.062 },
  { a: "right_elbow", b: "right_wrist", radius: 0.052 },
  { a: "pelvis", b: "left_hip", radius: 0.09 },
  { a: "pelvis", b: "right_hip", radius: 0.09 },
  { a: "left_hip", b: "left_knee", radius: 0.082 },
  { a: "left_knee", b: "left_ankle", radius: 0.064 },
  { a: "right_hip", b: "right_knee", radius: 0.082 },
  { a: "right_knee", b: "right_ankle", radius: 0.064 },
];

export class MotionViewer {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly joints = new Map<string, THREE.Mesh>();
  private readonly bones: THREE.Line[] = [];
  private readonly avatar = new AI4BluemanAvatar();
  private readonly primitiveCapsules: Array<{ mesh: THREE.Mesh; spec: CapsuleSpec }> = [];
  private readonly headMesh: THREE.Mesh;
  private readonly pelvisMesh: THREE.Mesh;
  private readonly handMeshes = new Map<string, THREE.Mesh>();
  private readonly footMeshes = new Map<string, THREE.Mesh>();
  private readonly clock = new THREE.Clock();
  private readonly supportMarker: THREE.Mesh;
  private readonly overheadBand: THREE.Mesh;
  private frames: PoseFrame[] = [];
  private segments: OverheadSegment[] = [];
  private events: AssistEvent[] = [];
  private playing = true;
  private currentTime = 0;
  private speed = 1;
  private duration = 0;
  private onFrameChanged: ((frame: PoseFrame, activeSegment?: OverheadSegment) => void) | undefined;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x15191d);
    container.appendChild(this.renderer.domElement);

    this.camera.position.set(2.7, 1.7, 3.1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1.1, 0);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1d252b, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
    keyLight.position.set(2.4, 4, 2);
    this.scene.add(keyLight);

    const grid = new THREE.GridHelper(4, 20, 0x34414b, 0x25313a);
    this.scene.add(grid);
    this.scene.add(this.avatar.group);
    this.avatar.load().catch((error: unknown) => {
      console.error("Failed to load AI4Animation Blueman avatar", error);
    });

    this.createSkeleton();
    const primitiveParts = this.createPrimitiveCharacter();
    this.headMesh = primitiveParts.headMesh;
    this.pelvisMesh = primitiveParts.pelvisMesh;
    this.overheadBand = this.createOverheadBand();
    this.supportMarker = this.createSupportMarker();

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this.animate();
  }

  setData(frames: PoseFrame[], segments: OverheadSegment[], events: AssistEvent[]) {
    this.frames = frames;
    this.segments = segments;
    this.events = events;
    this.duration = frames.at(-1)?.time ?? 0;
    this.currentTime = 0;
    this.renderFrame(0);
  }

  setFrameCallback(callback: (frame: PoseFrame, activeSegment?: OverheadSegment) => void) {
    this.onFrameChanged = callback;
  }

  setPlaying(value: boolean) {
    this.playing = value;
  }

  isPlaying() {
    return this.playing;
  }

  setSpeed(value: number) {
    this.speed = value;
  }

  seek01(value: number) {
    this.currentTime = this.duration * Math.min(Math.max(value, 0), 1);
    this.renderFrame(this.currentTime);
  }

  getTime() {
    return this.currentTime;
  }

  getDuration() {
    return this.duration;
  }

  private createSkeleton() {
    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xeef4f8, roughness: 0.42 });
    const wristMaterial = new THREE.MeshStandardMaterial({ color: 0x5ce0d4, roughness: 0.36 });
    const sphere = new THREE.SphereGeometry(0.035, 18, 18);
    for (const name of JOINT_NAMES) {
      const mesh = new THREE.Mesh(sphere, name.includes("wrist") ? wristMaterial : jointMaterial);
      mesh.visible = false;
      this.joints.set(name, mesh);
      this.scene.add(mesh);
    }

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x89a2ad,
      linewidth: 1,
      transparent: true,
      opacity: 0.18,
    });
    for (const _pair of BONE_PAIRS) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const line = new THREE.Line(geometry, lineMaterial);
      line.visible = false;
      this.bones.push(line);
      this.scene.add(line);
    }
  }

  private createPrimitiveCharacter() {
    const material = new THREE.MeshStandardMaterial({
      color: 0xe7ecef,
      roughness: 0.46,
      metalness: 0.02,
    });
    const accent = new THREE.MeshStandardMaterial({
      color: 0x71ded6,
      roughness: 0.35,
      metalness: 0.02,
    });
    const capsuleGeometry = new THREE.CapsuleGeometry(0.5, 1, 10, 20);
    for (const spec of PRIMITIVE_CAPSULES) {
      const mesh = new THREE.Mesh(capsuleGeometry, material);
      mesh.visible = false;
      this.primitiveCapsules.push({ mesh, spec });
      this.scene.add(mesh);
    }

    const headMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), material);
    headMesh.scale.set(0.115, 0.16, 0.105);
    headMesh.visible = false;
    this.scene.add(headMesh);

    const pelvisMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 18), material);
    pelvisMesh.scale.set(0.18, 0.12, 0.12);
    pelvisMesh.visible = false;
    this.scene.add(pelvisMesh);

    const handGeometry = new THREE.SphereGeometry(0.06, 20, 16);
    for (const name of ["left_wrist", "right_wrist"]) {
      const mesh = new THREE.Mesh(handGeometry, accent);
      mesh.visible = false;
      this.handMeshes.set(name, mesh);
      this.scene.add(mesh);
    }

    const footGeometry = new THREE.BoxGeometry(0.09, 0.055, 0.18);
    for (const name of ["left_ankle", "right_ankle"]) {
      const mesh = new THREE.Mesh(footGeometry, material);
      mesh.visible = false;
      this.footMeshes.set(name, mesh);
      this.scene.add(mesh);
    }

    return { headMesh, pelvisMesh };
  }

  private createOverheadBand() {
    const geometry = new THREE.BoxGeometry(1.35, 0.018, 0.95);
    const material = new THREE.MeshStandardMaterial({
      color: 0x2ba6a0,
      transparent: true,
      opacity: 0.22,
      roughness: 0.7,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 1.86, 0.04);
    mesh.visible = false;
    this.scene.add(mesh);
    return mesh;
  }

  private createSupportMarker() {
    const geometry = new THREE.ConeGeometry(0.065, 0.18, 24);
    const material = new THREE.MeshStandardMaterial({ color: 0xf2c15b, roughness: 0.4 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = Math.PI;
    mesh.visible = false;
    this.scene.add(mesh);
    return mesh;
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    const delta = this.clock.getDelta();
    if (this.playing && this.duration > 0) {
      this.currentTime = (this.currentTime + delta * this.speed) % this.duration;
      this.renderFrame(this.currentTime);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  private renderFrame(time: number) {
    const frame = this.frameAt(time);
    if (!frame) return;
    const activeSegment = this.segments.find(
      (segment) => frame.time >= segment.start_time && frame.time <= segment.end_time,
    );
    this.overheadBand.visible = Boolean(activeSegment);

    for (const name of JOINT_NAMES) {
      const joint = frame.joints[name];
      const mesh = this.joints.get(name);
      if (!mesh) continue;
      mesh.visible = false;
      if (joint) mesh.position.set(joint[0], joint[1], joint[2]);
    }

    BONE_PAIRS.forEach(([a, b], index) => {
      const line = this.bones[index];
      const ja = frame.joints[a];
      const jb = frame.joints[b];
      line.visible = false;
      if (ja && jb) {
        const attr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
        attr.setXYZ(0, ja[0], ja[1], ja[2]);
        attr.setXYZ(1, jb[0], jb[1], jb[2]);
        attr.needsUpdate = true;
        line.geometry.computeBoundingSphere();
      }
    });

    this.avatar.update(frame);

    this.updateSupportMarker(frame);
    this.onFrameChanged?.(frame, activeSegment);
  }

  private updatePrimitiveCharacter(frame: PoseFrame) {
    for (const { mesh, spec } of this.primitiveCapsules) {
      const a = frame.joints[spec.a];
      const b = frame.joints[spec.b];
      mesh.visible = Boolean(a && b);
      if (a && b) this.placeCapsule(mesh, a, b, spec.radius);
    }

    const head = frame.joints.head;
    this.headMesh.visible = Boolean(head);
    if (head) {
      this.headMesh.position.set(head[0], head[1], head[2] + 0.015);
      this.headMesh.rotation.set(-0.18, 0, 0);
    }

    const pelvis = frame.joints.pelvis;
    this.pelvisMesh.visible = Boolean(pelvis);
    if (pelvis) {
      this.pelvisMesh.position.set(pelvis[0], pelvis[1], pelvis[2]);
      this.pelvisMesh.rotation.set(0.18, 0, 0);
    }

    for (const [name, mesh] of this.handMeshes) {
      const joint = frame.joints[name];
      mesh.visible = Boolean(joint);
      if (joint) mesh.position.set(joint[0], joint[1], joint[2]);
    }

    for (const [name, mesh] of this.footMeshes) {
      const joint = frame.joints[name];
      mesh.visible = Boolean(joint);
      if (joint) {
        const side = name.startsWith("left") ? -1 : 1;
        mesh.position.set(joint[0] + side * 0.012, joint[1] - 0.02, joint[2] + 0.055);
        mesh.rotation.set(0.08, 0, side * 0.05);
      }
    }
  }

  private placeCapsule(mesh: THREE.Mesh, a: [number, number, number], b: [number, number, number], radius: number) {
    const start = new THREE.Vector3(a[0], a[1], a[2]);
    const end = new THREE.Vector3(b[0], b[1], b[2]);
    const delta = end.clone().sub(start);
    const length = delta.length();
    if (length <= 0.0001) {
      mesh.visible = false;
      return;
    }

    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(Y_AXIS, delta.normalize());
    mesh.scale.set(radius / 0.5, length / 2, radius / 0.5);
  }

  private frameAt(time: number) {
    if (this.frames.length === 0) return undefined;
    let best = this.frames[0];
    for (const frame of this.frames) {
      if (frame.time > time) break;
      best = frame;
    }
    return best;
  }

  private updateSupportMarker(frame: PoseFrame) {
    const hold = this.events.find((event) => {
      if (event.event !== "hold_support") return false;
      const release = this.events.find(
        (candidate) =>
          candidate.segment_index === event.segment_index && candidate.event === "release_support",
      );
      return frame.time >= event.time && frame.time <= (release?.time ?? event.time);
    });
    this.supportMarker.visible = Boolean(hold);
    if (!hold) return;
    this.supportMarker.position.set(0, 2.13, 0.04);
  }

  private resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }
}
