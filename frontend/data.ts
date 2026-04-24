import type { AssistEvent, OverheadSegment, PoseFrame } from "./types";

function resolveUrl(url: string) {
  return new URL(url, window.location.href).toString();
}

export async function loadPoseJsonl(url: string): Promise<PoseFrame[]> {
  let response: Response;
  try {
    response = await fetch(resolveUrl(url));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Pose fetch failed for ${url}: ${message}`);
  }
  if (!response.ok) {
    throw new Error(`Failed to load pose data: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  try {
    return text
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as PoseFrame);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Pose parse failed for ${url}: ${message}`);
  }
}

export async function loadSegments(url: string): Promise<OverheadSegment[]> {
  const response = await fetch(resolveUrl(url));
  if (!response.ok) return [];
  try {
    const payload = (await response.json()) as { segments?: OverheadSegment[] };
    return payload.segments ?? [];
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Segment parse failed for ${url}: ${message}`);
  }
}

export async function loadAssistEvents(url: string): Promise<AssistEvent[]> {
  const response = await fetch(resolveUrl(url));
  if (!response.ok) return [];
  try {
    const payload = (await response.json()) as { events?: AssistEvent[] };
    return payload.events ?? [];
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Assist parse failed for ${url}: ${message}`);
  }
}
