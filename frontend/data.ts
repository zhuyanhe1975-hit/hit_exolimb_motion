import type { AssistEvent, OverheadSegment, PoseFrame } from "./types";

export async function loadPoseJsonl(url: string): Promise<PoseFrame[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load pose data: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as PoseFrame);
}

export async function loadSegments(url: string): Promise<OverheadSegment[]> {
  const response = await fetch(url);
  if (!response.ok) return [];
  const payload = (await response.json()) as { segments?: OverheadSegment[] };
  return payload.segments ?? [];
}

export async function loadAssistEvents(url: string): Promise<AssistEvent[]> {
  const response = await fetch(url);
  if (!response.ok) return [];
  const payload = (await response.json()) as { events?: AssistEvent[] };
  return payload.events ?? [];
}

