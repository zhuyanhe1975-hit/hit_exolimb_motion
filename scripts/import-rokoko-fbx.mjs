#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { AnimationMixer, Vector3 } from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

const BODY_JOINTS = {
  Hips: "pelvis",
  Spine2: "spine",
  Neck: "neck",
  Head: "head",
  LeftArm: "left_shoulder",
  RightArm: "right_shoulder",
  LeftForeArm: "left_elbow",
  RightForeArm: "right_elbow",
  LeftHand: "left_wrist",
  RightHand: "right_wrist",
  LeftThigh: "left_hip",
  RightThigh: "right_hip",
  LeftShin: "left_knee",
  RightShin: "right_knee",
  LeftFoot: "left_ankle",
  RightFoot: "right_ankle",
};

const FINGERS = {
  thumb: 1,
  index: 2,
  middle: 3,
  ring: 4,
  pinky: 5,
};

const args = parseArgs(process.argv.slice(2));
if (!args.input || !args.out) {
  console.error("Usage: node scripts/import-rokoko-fbx.mjs --input file.fbx --out pose.jsonl [--fps 30] [--scale 0.01]");
  process.exit(2);
}

const fps = Number(args.fps ?? 30);
const scale = Number(args.scale ?? 0.01);
const pelvisHeight = Number(args.pelvisHeight ?? 0.95);
const lockRoot = args.keepRootTranslation !== "true";

const buffer = fs.readFileSync(args.input);
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const object = new FBXLoader().parse(arrayBuffer, path.dirname(args.input));
const clip = object.animations[0];
if (!clip) {
  throw new Error(`${args.input} does not contain an animation clip`);
}

const bones = new Map();
object.traverse((node) => {
  if (node.isBone && !bones.has(node.name)) {
    bones.set(node.name, node);
  }
});

const mixer = new AnimationMixer(object);
mixer.clipAction(clip).play();
mixer.setTime(0);
object.updateMatrixWorld(true);
const origin = worldPoint(bones.get("Hips"));

const frameCount = Math.floor(clip.duration * fps) + 1;
const frames = [];
for (let frame = 0; frame < frameCount; frame += 1) {
  const time = Math.min(frame / fps, clip.duration);
  mixer.setTime(time);
  object.updateMatrixWorld(true);
  const joints = {};

  for (const [source, target] of Object.entries(BODY_JOINTS)) {
    const bone = bones.get(source);
    if (bone) joints[target] = transform(worldPoint(bone), origin, scale, pelvisHeight);
  }

  addHand(joints, bones, "Left", "left", origin, scale, pelvisHeight);
  addHand(joints, bones, "Right", "right", origin, scale, pelvisHeight);

  if (lockRoot && joints.pelvis) {
    const delta = [
      joints.pelvis[0],
      joints.pelvis[1] - pelvisHeight,
      joints.pelvis[2],
    ];
    for (const [name, point] of Object.entries(joints)) {
      joints[name] = [point[0] - delta[0], point[1] - delta[1], point[2] - delta[2]];
    }
  }

  frames.push({ frame, time, joints });
}

fs.mkdirSync(path.dirname(args.out), { recursive: true });
fs.writeFileSync(args.out, frames.map((frame) => JSON.stringify(frame)).join("\n") + "\n");
console.log(`Wrote ${frames.length} frames from ${args.input} to ${args.out}`);

function addHand(joints, bones, sourceSide, targetSide, origin, scale, pelvisHeight) {
  const wrist = bones.get(`${sourceSide}Hand`);
  if (!wrist) return;

  const bases = {
    thumb: bones.get(`${sourceSide}Finger1Metacarpal`),
    index: bones.get(`${sourceSide}Finger2Metacarpal`),
    middle: bones.get(`${sourceSide}Finger3Metacarpal`),
    ring: bones.get(`${sourceSide}Finger4Metacarpal`),
    pinky: bones.get(`${sourceSide}Finger5Metacarpal`),
  };
  const tips = Object.fromEntries(
    Object.entries(FINGERS).map(([name, number]) => [
      name,
      bones.get(`${sourceSide}Finger${number}Tip`) ??
        bones.get(`${sourceSide}Finger${number}Distal`) ??
        bones.get(`${sourceSide}Finger${number}Proximal`),
    ]),
  );

  const basePoints = Object.values(bases).filter(Boolean).map((bone) => transform(worldPoint(bone), origin, scale, pelvisHeight));
  const tipPoints = Object.values(tips).filter(Boolean).map((bone) => transform(worldPoint(bone), origin, scale, pelvisHeight));
  if (!basePoints.length || !tipPoints.length) return;

  joints[`${targetSide}_palm`] = average([transform(worldPoint(wrist), origin, scale, pelvisHeight), ...basePoints]);
  joints[`${targetSide}_hand`] = average(tipPoints);
  joints[`${targetSide}_index_base`] = bases.index ? transform(worldPoint(bases.index), origin, scale, pelvisHeight) : joints[`${targetSide}_palm`];
  joints[`${targetSide}_pinky_base`] = bases.pinky ? transform(worldPoint(bases.pinky), origin, scale, pelvisHeight) : joints[`${targetSide}_palm`];

  for (const [name, bone] of Object.entries(tips)) {
    if (bone) joints[`${targetSide}_${name}`] = transform(worldPoint(bone), origin, scale, pelvisHeight);
  }
}

function worldPoint(bone) {
  if (!bone) throw new Error("missing required bone");
  const point = new Vector3();
  bone.getWorldPosition(point);
  return point;
}

function transform(point, origin, scale, pelvisHeight) {
  return [
    -(point.x - origin.x) * scale,
    (point.y - origin.y) * scale + pelvisHeight,
    (point.z - origin.z) * scale,
  ];
}

function average(points) {
  const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]], [0, 0, 0]);
  return [sum[0] / points.length, sum[1] / points.length, sum[2] / points.length];
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = "true";
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}
