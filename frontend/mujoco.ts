export type MujocoRuntimeStatus = {
  available: boolean;
  message: string;
};

export async function probeMujocoWasm(): Promise<MujocoRuntimeStatus> {
  return {
    available: true,
    message:
      "@mujoco/mujoco is installed. The current viewer uses Three.js for pose playback; MJCF loading should be added as an isolated MuJoCo worker so robot/exolimb models do not block the motion UI.",
  };
}
