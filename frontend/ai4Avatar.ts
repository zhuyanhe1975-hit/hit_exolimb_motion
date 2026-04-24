import * as THREE from "three";
import { CCDIKSolver, type IK } from "three/examples/jsm/animation/CCDIKSolver.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { PoseFrame, Vec3 } from "./types";

export class AI4BluemanAvatar {
  readonly group = new THREE.Group();
  private readonly bones = new Map<string, THREE.Bone>();
  private readonly restQuaternions = new Map<string, THREE.Quaternion>();
  private loaded = false;
  private rootBone: THREE.Bone | undefined;
  private skinnedMesh: THREE.SkinnedMesh | undefined;
  private ikSolver: CCDIKSolver | undefined;
  private leftTarget: THREE.Bone | undefined;
  private rightTarget: THREE.Bone | undefined;
  private readonly ikBoneNames = [
    "p_l_scap",
    "b_l_shoulder",
    "b_l_arm",
    "b_l_forearm",
    "b_l_wrist_twist",
    "b_l_wrist",
    "p_r_scap",
    "b_r_shoulder",
    "b_r_arm",
    "b_r_forearm",
    "b_r_wrist_twist",
    "b_r_wrist",
    "b_l_thumb0",
    "b_l_thumb1",
    "b_l_thumb2",
    "b_l_thumb3",
    "b_l_index1",
    "b_l_index2",
    "b_l_index3",
    "b_l_middle1",
    "b_l_middle2",
    "b_l_middle3",
    "b_l_ring1",
    "b_l_ring2",
    "b_l_ring3",
    "b_l_pinky1",
    "b_l_pinky2",
    "b_l_pinky3",
    "b_r_thumb0",
    "b_r_thumb1",
    "b_r_thumb2",
    "b_r_thumb3",
    "b_r_index1",
    "b_r_index2",
    "b_r_index3",
    "b_r_middle1",
    "b_r_middle2",
    "b_r_middle3",
    "b_r_ring1",
    "b_r_ring2",
    "b_r_ring3",
    "b_r_pinky1",
    "b_r_pinky2",
    "b_r_pinky3",
  ];

  async load(url = "/assets/human/ai4animation/worker.glb") {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    const object = gltf.scene;
    object.name = "AI4Animation Worker";
    object.rotation.set(0, 0, 0);
    object.position.set(0, 0, 0);

    const material = new THREE.MeshStandardMaterial({
      color: 0xf5f7f8,
      roughness: 0.52,
      metalness: 0.02,
    });

    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const meshName = child.name.toLowerCase();
        child.visible = meshName === "mesh" || meshName === "body_mesh" || meshName === "head_mesh";
        if (!child.visible) return;
        child.castShadow = false;
        child.receiveShadow = true;
        child.material = material;
        if (child instanceof THREE.SkinnedMesh && meshName === "mesh") {
          this.skinnedMesh = child;
        }
      }
      if (child instanceof THREE.Bone) {
        this.bones.set(child.name, child);
      }
    });

    this.group.add(object);
    this.rootBone = this.bones.get("b_root");
    this.captureRestPose();
    this.setupIk();
    this.loaded = true;
  }

  isLoaded() {
    return this.loaded;
  }

  update(frame: PoseFrame) {
    if (!this.loaded) return;
    this.positionRoot(frame);
    this.restoreIkBones();
    this.updateIkTarget(this.leftTarget, frame.joints.left_wrist);
    this.updateIkTarget(this.rightTarget, frame.joints.right_wrist);
    this.group.updateWorldMatrix(true, true);
    this.ikSolver?.update(1);
    this.applyHandPose("left", frame);
    this.applyHandPose("right", frame);
  }

  private captureRestPose() {
    this.group.updateWorldMatrix(true, true);
    for (const name of this.ikBoneNames) {
      const bone = this.bones.get(name);
      if (bone) {
        this.restQuaternions.set(name, bone.quaternion.clone());
      }
    }
  }

  private restoreIkBones() {
    for (const name of this.ikBoneNames) {
      const bone = this.bones.get(name);
      const rest = this.restQuaternions.get(name);
      if (bone && rest) {
        bone.quaternion.copy(rest);
      }
    }
  }

  private positionRoot(frame: PoseFrame) {
    const pelvis = frame.joints.pelvis;
    if (!pelvis || !this.rootBone) return;
    this.group.position.set(pelvis[0], 0, pelvis[2]);
  }

  private setupIk() {
    if (!this.skinnedMesh) return;
    const skeleton = this.skinnedMesh.skeleton;
    this.leftTarget = this.createIkTarget("left_wrist_target");
    this.rightTarget = this.createIkTarget("right_wrist_target");
    skeleton.bones.push(this.leftTarget, this.rightTarget);
    skeleton.calculateInverses();
    const index = (name: string) => skeleton.bones.findIndex((bone) => bone.name === name);
    const iks: IK[] = [
      {
        target: index("left_wrist_target"),
        effector: index("b_l_wrist"),
        iteration: 12,
        maxAngle: 0.28,
        links: [
          { index: index("b_l_wrist_twist") },
          { index: index("b_l_forearm") },
          { index: index("b_l_arm") },
          { index: index("p_l_scap") },
        ],
      },
      {
        target: index("right_wrist_target"),
        effector: index("b_r_wrist"),
        iteration: 12,
        maxAngle: 0.28,
        links: [
          { index: index("b_r_wrist_twist") },
          { index: index("b_r_forearm") },
          { index: index("b_r_arm") },
          { index: index("p_r_scap") },
        ],
      },
    ].filter((ik) => ik.target >= 0 && ik.effector >= 0 && ik.links.every((link) => link.index >= 0));
    this.ikSolver = new CCDIKSolver(this.skinnedMesh, iks);
  }

  private createIkTarget(name: string) {
    const bone = new THREE.Bone();
    bone.name = name;
    this.group.add(bone);
    this.bones.set(name, bone);
    return bone;
  }

  private updateIkTarget(target: THREE.Bone | undefined, joint: Vec3 | undefined) {
    if (!target || !joint) return;
    const worldTarget = new THREE.Vector3(-joint[0], joint[1], joint[2]);
    const localTarget = this.group.worldToLocal(worldTarget);
    target.position.copy(localTarget);
    target.updateMatrixWorld(true);
  }

  private applyHandPose(side: "left" | "right", frame: PoseFrame) {
    const sideCode = side === "left" ? "l" : "r";
    const wrist = frame.joints[`${side}_wrist`];
    const palm = frame.joints[`${side}_palm`];
    const thumb = frame.joints[`${side}_thumb`];
    const index = frame.joints[`${side}_index`];
    const middle = frame.joints[`${side}_middle`];
    const ring = frame.joints[`${side}_ring`];
    const pinky = frame.joints[`${side}_pinky`];
    const indexBase = frame.joints[`${side}_index_base`];
    const pinkyBase = frame.joints[`${side}_pinky_base`];
    if (!wrist || !palm || !thumb || !index || !middle || !ring || !pinky) return;

    const palmSize =
      this.distance(indexBase ?? index, pinkyBase ?? pinky) || this.distance(index, pinky) || 0.08;
    const scale = Math.max(palmSize, 0.06);
    this.applyWristPose(side, wrist, palm, middle, indexBase ?? index, pinkyBase ?? pinky, scale);

    const curls = {
      thumb: this.fingerCurl(palm, thumb, scale * 1.05),
      index: this.fingerCurl(palm, index, scale * 1.18),
      middle: this.fingerCurl(palm, middle, scale * 1.24),
      ring: this.fingerCurl(palm, ring, scale * 1.18),
      pinky: this.fingerCurl(palm, pinky, scale * 1.06),
    };

    this.applyFingerChain(`${sideCode}_thumb`, curls.thumb, side === "left" ? 1 : -1, true);
    this.applyFingerChain(`${sideCode}_index`, curls.index, side === "left" ? 1 : -1);
    this.applyFingerChain(`${sideCode}_middle`, curls.middle, side === "left" ? 1 : -1);
    this.applyFingerChain(`${sideCode}_ring`, curls.ring, side === "left" ? 1 : -1);
    this.applyFingerChain(`${sideCode}_pinky`, curls.pinky, side === "left" ? 1 : -1);
    this.applyFingerSpread(side, thumb, index, middle, ring, pinky, palm, scale);
  }

  private fingerCurl(wrist: Vec3, tip: Vec3, openDistance: number) {
    const distance = this.distance(wrist, tip);
    const ratio = THREE.MathUtils.clamp(distance / Math.max(openDistance, 1e-4), 0, 1);
    return 1 - ratio;
  }

  private applyFingerChain(prefix: string, curl: number, handSign: number, thumb = false) {
    const names = thumb
      ? [`b_${prefix}0`, `b_${prefix}1`, `b_${prefix}2`, `b_${prefix}3`]
      : [`b_${prefix}1`, `b_${prefix}2`, `b_${prefix}3`];
    const multipliers = thumb ? [0.12, 0.28, 0.4, 0.34] : [0.45, 0.72, 0.58];
    names.forEach((name, index) => {
      const bone = this.bones.get(name);
      const rest = this.restQuaternions.get(name);
      if (!bone || !rest) return;
      const angle = curl * multipliers[index];
      const euler = thumb
        ? new THREE.Euler(0.05 * handSign, handSign * angle * 0.7, -handSign * angle, "XYZ")
        : new THREE.Euler(angle, 0, 0, "XYZ");
      bone.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(euler));
    });
  }

  private applyWristPose(
    side: "left" | "right",
    wristPoint: Vec3,
    palm: Vec3,
    middle: Vec3,
    indexBase: Vec3,
    pinkyBase: Vec3,
    scale: number,
  ) {
    const wrist = this.bones.get(side === "left" ? "b_l_wrist" : "b_r_wrist");
    const rest = this.restQuaternions.get(side === "left" ? "b_l_wrist" : "b_r_wrist");
    if (!wrist || !rest) return;
    const handSign = side === "left" ? 1 : -1;
    const forward = this.subVec(middle, palm);
    const lateral = this.subVec(indexBase, pinkyBase);
    const bend = THREE.MathUtils.clamp((forward[1] - 0.06) / Math.max(scale, 1e-4), -1, 1);
    const yaw = THREE.MathUtils.clamp(forward[0] / Math.max(scale, 1e-4), -1, 1);
    const twist = THREE.MathUtils.clamp(lateral[2] / Math.max(scale, 1e-4), -1, 1);
    const euler = new THREE.Euler(
      -bend * 0.45,
      handSign * yaw * 0.35,
      handSign * twist * 0.55,
      "XYZ",
    );
    wrist.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(euler));

    const wristTwist = this.bones.get(side === "left" ? "b_l_wrist_twist" : "b_r_wrist_twist");
    const wristTwistRest = this.restQuaternions.get(side === "left" ? "b_l_wrist_twist" : "b_r_wrist_twist");
    if (wristTwist && wristTwistRest) {
      const lift = THREE.MathUtils.clamp((palm[2] - wristPoint[2]) / Math.max(scale, 1e-4), -1, 1);
      wristTwist.quaternion
        .copy(wristTwistRest)
        .multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -handSign * lift * 0.4)));
    }
  }

  private applyFingerSpread(
    side: "left" | "right",
    thumb: Vec3,
    index: Vec3,
    middle: Vec3,
    ring: Vec3,
    pinky: Vec3,
    palm: Vec3,
    scale: number,
  ) {
    const handSign = side === "left" ? 1 : -1;
    const spreadValues = {
      thumb: THREE.MathUtils.clamp((thumb[0] - palm[0]) / Math.max(scale, 1e-4), -1, 1),
      index: THREE.MathUtils.clamp((index[0] - middle[0]) / Math.max(scale, 1e-4), -1, 1),
      ring: THREE.MathUtils.clamp((ring[0] - middle[0]) / Math.max(scale, 1e-4), -1, 1),
      pinky: THREE.MathUtils.clamp((pinky[0] - ring[0]) / Math.max(scale, 1e-4), -1, 1),
    };
    this.applySpreadToBone(`b_${side === "left" ? "l" : "r"}_thumb0`, handSign * spreadValues.thumb * 0.35);
    this.applySpreadToBone(`b_${side === "left" ? "l" : "r"}_index1`, handSign * spreadValues.index * 0.18);
    this.applySpreadToBone(`b_${side === "left" ? "l" : "r"}_ring1`, handSign * spreadValues.ring * 0.12);
    this.applySpreadToBone(`b_${side === "left" ? "l" : "r"}_pinky1`, handSign * spreadValues.pinky * 0.22);
  }

  private applySpreadToBone(name: string, angle: number) {
    const bone = this.bones.get(name);
    const rest = this.restQuaternions.get(name);
    if (!bone || !rest) return;
    bone.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, angle, 0)));
  }

  private distance(a: Vec3, b: Vec3) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private subVec(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }
}
