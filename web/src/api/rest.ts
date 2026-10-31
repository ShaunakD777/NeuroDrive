import type {
  ActivationCurves,
  CheckpointList,
  CompareResponse,
  RunDetail,
  RunListResponse,
  RunMetrics,
  ToyCheckpointResponse,
  TrackGeometry,
  TrackListResponse,
} from "./generated";

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

// Run folders (api v1.11, ADR-031, ADR-033). Run ids are folder names; they are encoded for the URL anyway.
const runPath = (runId: string) => `/api/runs/${encodeURIComponent(runId)}`;

/** Every run folder, newest first (`GET /api/runs`). */
export async function fetchRuns(): Promise<RunListResponse> {
  return getJson<RunListResponse>("/api/runs");
}

/** A run's config and meta (`GET /api/runs/{id}`). */
export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  return getJson<RunDetail>(runPath(runId));
}

/** A run's metrics as columns (`GET /api/runs/{id}/metrics`). */
export async function fetchRunMetrics(runId: string): Promise<RunMetrics> {
  return getJson<RunMetrics>(`${runPath(runId)}/metrics`);
}

/** The units that have a checkpoint (`GET /api/runs/{id}/checkpoints`). */
export async function fetchCheckpoints(runId: string): Promise<CheckpointList> {
  return getJson<CheckpointList>(`${runPath(runId)}/checkpoints`);
}

/** The toy lab's view of one checkpoint (`GET /api/runs/{id}/checkpoints/{unit}/toy`). */
export async function fetchToyCheckpoint(runId: string, unit: number): Promise<ToyCheckpointResponse> {
  return getJson<ToyCheckpointResponse>(`${runPath(runId)}/checkpoints/${unit}/toy`);
}

/**
 * Loss curves of several runs grouped by sweep, with median and IQR per epoch, computed by the
 * server (`GET /api/runs/compare`, ADR-033). A null median or quartile means +infinity (diverged).
 */
export async function fetchCompare(runIds: readonly string[], metric: "loss" | "val_loss"): Promise<CompareResponse> {
  const ids = runIds.map(encodeURIComponent).join(",");
  return getJson<CompareResponse>(`/api/runs/compare?ids=${ids}&metric=${metric}`);
}
