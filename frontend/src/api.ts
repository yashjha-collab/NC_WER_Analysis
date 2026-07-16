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
  summary_json?: Record<string, unknown> | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
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
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
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
};
