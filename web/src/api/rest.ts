import type {
  ActivationCurves,
  CheckpointDiff,
  CheckpointList,
  CompareResponse,
  ConfigListResponse,
  ExamCardResponse,
  LaunchRequest,
  LaunchResponse,
  RunDetail,
  RunListResponse,
  RunMetrics,
  StopResponse,
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

/**
 * What changed from checkpoint `from` to checkpoint `unit`, weight by weight, computed by the
 * server (`GET /api/runs/{id}/checkpoints/{unit}/diff?from=`, task 5.9).
 */
export async function fetchCheckpointDiff(runId: string, unit: number, from: number): Promise<CheckpointDiff> {
  return getJson<CheckpointDiff>(`${runPath(runId)}/checkpoints/${unit}/diff?from=${from}`);
}

/** A checkpoint's Driving Exam card as stored (`GET /api/runs/{id}/exams/{unit}`, task 5.9). */
export async function fetchExamCard(runId: string, unit: number): Promise<ExamCardResponse> {
  return getJson<ExamCardResponse>(`${runPath(runId)}/exams/${unit}`);
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

// Launching and stopping training runs (api v1.13, task 5.8, ADR-041). The server starts the
// training as a separate process; progress then arrives over the /ws/runs socket.

/** POST `body` as JSON; on refusal, throw the server's own explanation (FastAPI's `detail`). */
async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `server answered ${response.status}`;
    try {
      const answer = (await response.json()) as { detail?: unknown };
      if (typeof answer.detail === "string") detail = answer.detail;
      else if (Array.isArray(answer.detail))
        detail = answer.detail.map((d) => (typeof d === "string" ? d : (d as { msg?: string }).msg ?? JSON.stringify(d))).join("; ");
    } catch {
      // not JSON: keep the status line
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/** The training configs in `configs/` that the app may launch (`GET /api/configs`). */
export async function fetchConfigs(): Promise<ConfigListResponse> {
  return getJson<ConfigListResponse>("/api/configs");
}

/** Start `nd train` for a config with the CLI's overrides; resolves to the new run's id (`POST /api/runs`). */
export async function launchRun(request: LaunchRequest): Promise<LaunchResponse> {
  return postJson<LaunchResponse>("/api/runs", request);
}

/** Ask a running run to stop after its current unit (`POST /api/runs/{id}/stop`). */
export async function stopRun(runId: string): Promise<StopResponse> {
  return postJson<StopResponse>(`${runPath(runId)}/stop`);
}
