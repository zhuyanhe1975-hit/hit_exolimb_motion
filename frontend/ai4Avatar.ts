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

  update(frame: PoseFrame) {
    if (!this.loaded) return;
    this.positionRoot(frame);
    this.restoreIkBones();
    this.updateIkTarget(this.leftTarget, frame.joints.left_wrist);
    this.updateIkTarget(this.rightTarget, frame.joints.right_wrist);
    this.group.updateWorldMatrix(true, true);
    this.ikSolver?.update(1);
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
}
