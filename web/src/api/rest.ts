import type { ActivationCurves, TrackGeometry, TrackListResponse } from "./generated";

/** Ask the Python server something over plain HTTP; throw a readable error if it says no. */
async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: server answered ${response.status}`);
  return (await response.json()) as T;
}

/** The tracks the server can load (`GET /api/tracks`). */
export async function fetchTracks(): Promise<TrackListResponse> {
  return getJson<TrackListResponse>("/api/tracks");
}

/** The outline of one track, for drawing (`GET /api/tracks/{name}`). */
export async function fetchTrack(name: string): Promise<TrackGeometry> {
  return getJson<TrackGeometry>(`/api/tracks/${encodeURIComponent(name)}`);
}

/** The four activation functions sampled on the same z values (`GET /api/activations`). */
export async function fetchActivations(): Promise<ActivationCurves> {
  return getJson<ActivationCurves>("/api/activations");
}
