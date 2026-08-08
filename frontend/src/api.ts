export type Dataset = {
  id: number;
  name: string;
  description?: string | null;
  call_count: number;
  source_path?: string | null;
  created_at?: string | null;
};

export type Run = {
  id: number;
  dataset_id: number;
  name: string;
  tier: string;
  status: string;
  progress: number;
  total_calls: number;
  completed_calls: number;
  failed_calls: number;
  jobs_total?: number | null;
  progress_detail?: TaskProgress | null;
  summary_json?: Record<string, unknown> | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type SttProgressStat = {
  stt_id: string;
  label: string;
  completed: number;
  total: number;
  pct: number;
};

export type TrialProgressStat = {
  engine: string;
  strength: number;
  label: string;
  completed: number;
  total: number;
  pct: number;
};

export type TaskProgress = {
  task_id: string;
  task_type: "run" | "sweep";
  status: string;
  progress_pct: number;
  jobs_total: number;
  jobs_completed: number;
  jobs_failed: number;
  jobs_cached: number;
  calls_total: number;
  calls_touched: number;
  calls_completed: number;
  stt_stats: SttProgressStat[];
  trial_stats: TrialProgressStat[];
  result?: SweepResponse | null;
  error?: string | null;
};

export type CallResult = {
  id: number;
  call_log_id: string;
  status: string;
  report_json?: Record<string, unknown> | null;
  error_message?: string | null;
};

const API_BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      const detail = parsed.detail;
      if (typeof detail === "string") {
        throw new Error(detail);
      }
      if (detail && typeof detail === "object") {
        const d = detail as {
          message?: string;
          hints?: string[];
          failed_engines?: string[];
        };
        const parts = [
          d.message,
          d.failed_engines?.length ? `Failed: ${d.failed_engines.join(", ")}` : "",
          ...(d.hints || []),
        ].filter(Boolean);
        throw new Error(parts.join("\n") || text);
      }
    } catch (err) {
      if (err instanceof SyntaxError) {
        /* not JSON */
      } else if (err instanceof Error) {
        throw err;
      }
    }
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export type Diagnostics = {
  ok: boolean;
  message: string;
  python?: { executable?: string; version?: string; platform?: string };
  livekit_worker_root?: Record<string, unknown>;
  hecttor_sdk_import?: { ok?: boolean; error?: string; module?: string | null };
  hecttor_api_key_set?: boolean;
  engines?: Record<string, { ok?: boolean; error?: string; processor?: string }>;
  failed_engines?: string[];
  hints?: string[];
};

export type SweepResult = {
  engine: string;
  strength: number;
  stt_provider?: string;
  stt_model?: string;
  stt_id?: string;
  calls: number;
  word_weighted_wer_pct: number | null;
  turn_avg_wer_pct: number | null;
  substitutions?: number;
  deletions?: number;
  insertions?: number;
  ref_words?: number;
  error?: string;
};

export type SttConfig = {
  stt_id: string;
  stt_provider: string;
  stt_model: string;
  stt_language: string;
};

export type SweepResponse = {
  sweep_id?: string;
  dataset_id: number;
  dataset_name: string;
  total_calls: number;
  turn_align: string;
  scoring?: string;
  execution_mode?: string;
  workers?: number;
  cpu_threads_per_worker?: number;
  jobs_total?: number;
  jobs_failed?: number;
  jobs_cached?: number;
  stt_configs?: SttConfig[];
  results: SweepResult[];
};

export type StrengthSweepStart = {
  sweep_id: string;
  status: string;
  dataset_id: number;
  dataset_name: string;
  total_calls: number;
  turn_align: string;
  scoring: string;
  execution_mode: string;
  jobs_total: number;
  stt_configs: SttConfig[];
  trials_count: number;
};

export const api = {
  health: () => request<{ status: string }>("/health"),
  diagnostics: () => request<Diagnostics>("/diagnostics"),
  listDatasets: () => request<Dataset[]>("/datasets"),
  importDataset: async (name: string, file: File, description?: string) => {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    if (description) form.append("description", description);
    return request<Dataset>("/datasets/import", { method: "POST", body: form });
  },
  importDatasetPath: async (name: string, path: string, description?: string) => {
    const form = new FormData();
    form.append("name", name);
    form.append("path", path);
    if (description) form.append("description", description);
    return request<Dataset>("/datasets/import-path", { method: "POST", body: form });
  },
  listRuns: () => request<Run[]>("/runs"),
  getRun: (id: number) => request<Run>(`/runs/${id}`),
  getRunResults: (id: number) => request<CallResult[]>(`/runs/${id}/results`),
  createRun: (payload: {
    dataset_id: number;
    name: string;
    tier?: string;
    call_ids?: string[];
    config?: Record<string, unknown>;
  }) =>
    request<Run>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  strengthSweep: (payload: {
    dataset_id: number;
    dtln_strengths: number[];
    hush_strengths: number[];
    hecttor_strengths: number[];
    hecttor_models?: string[];
    max_calls?: number;
    turn_align?: string;
    scoring?: string;
    execution_mode?: "server" | "local";
    workers?: number;
    stt_provider?: string;
    stt_model?: string;
    stt_language?: string;
    stt_preset_ids?: string[];
  }) =>
    request<StrengthSweepStart>("/strength-sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getSweepProgress: (sweepId: string) =>
    request<TaskProgress>(`/strength-sweep/${sweepId}/progress`),
};
