export type MujocoRuntimeStatus = {
  available: boolean;
  message: string;
};

export async function probeMujocoWasm(): Promise<MujocoRuntimeStatus> {
  return {
    available: false,
    message:
      "MuJoCo physics is not running on this page. The viewer is Three.js playback of precomputed pose data. An MJCF exolimb simulation is not part of this static demo and is not required to watch the sequence.",
  };
}
