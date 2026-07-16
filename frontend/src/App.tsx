import { useCallback, useEffect, useMemo, useState } from "react";
import { api, CallResult, Dataset, Run } from "./api";

function statusBadge(status: string) {
  const cls =
    status.includes("fail")
      ? "badge badge-failed"
      : status.includes("running")
        ? "badge badge-running"
        : "badge badge-completed";
  return <span className={cls}>{status}</span>;
}

function RankingTable({ summary }: { summary: Record<string, unknown> | null | undefined }) {
  const combined = (summary?.combined as Record<string, Record<string, number>>) || {};
  const ranking = (summary?.ranking_by_word_weighted_wer as string[]) || Object.keys(combined);

  if (!ranking.length) {
    return <p className="text-sm text-slate-500">No ranking yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b text-left text-slate-500">
            <th className="py-2 pr-4">Rank</th>
            <th className="py-2 pr-4">Engine</th>
            <th className="py-2 pr-4">Word-weighted WER</th>
            <th className="py-2 pr-4">Turn-avg WER</th>
            <th className="py-2">Calls</th>
          </tr>
        </thead>
        <tbody>
          {ranking.map((engine, idx) => {
            const row = combined[engine];
            if (!row) return null;
            return (
              <tr key={engine} className="border-b border-slate-100">
                <td className="py-2 pr-4">{idx + 1}</td>
                <td className="py-2 pr-4 font-medium">{engine}</td>
                <td className="py-2 pr-4">{row.word_weighted_wer_pct}%</td>
                <td className="py-2 pr-4">{row.turn_avg_wer_pct}%</td>
                <td className="py-2">{row.calls}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [results, setResults] = useState<CallResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [importName, setImportName] = useState("muthoot");
  const [importPath, setImportPath] = useState(
    "/Users/yash.jha.ext/Desktop/Muthoot_final_with_public_urls.json"
  );
  const [runName, setRunName] = useState("golden-tier-a");
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [tier, setTier] = useState("tier_a");

  const selectedRun = useMemo(
    () => runs.find((r) => r.id === selectedRunId) || null,
    [runs, selectedRunId]
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [ds, rs] = await Promise.all([api.listDatasets(), api.listRuns()]);
      setDatasets(ds);
      setRuns(rs);
      if (!datasetId && ds[0]) setDatasetId(ds[0].id);
      if (!selectedRunId && rs[0]) setSelectedRunId(rs[0].id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [datasetId, selectedRunId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selectedRunId) return;
    const timer = setInterval(async () => {
      try {
        const [run, runResults] = await Promise.all([
          api.getRun(selectedRunId),
          api.getRunResults(selectedRunId),
        ]);
        setRuns((prev) => prev.map((r) => (r.id === run.id ? run : r)));
        setResults(runResults);
      } catch {
        /* ignore polling errors */
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [selectedRunId]);

  const handleImportPath = async () => {
    try {
      const ds = await api.importDatasetPath(importName, importPath, "Imported from local path");
      setDatasets((prev) => [ds, ...prev.filter((d) => d.id !== ds.id)]);
      setDatasetId(ds.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    }
  };

  const handleCreateRun = async () => {
    if (!datasetId) return;
    try {
      const run = await api.createRun({
        dataset_id: Number(datasetId),
        name: runName,
        tier,
        config: {
          turn_align: "vad",
          segment_mode: "turn",
          stt_provider: "cartesia",
          stt_model: "ink-whisper",
          stt_language: "hi",
        },
        call_ids: undefined,
      });
      setRuns((prev) => [run, ...prev]);
      setSelectedRunId(run.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run creation failed");
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight">NC WER Analysis</h1>
        <p className="mt-2 text-slate-600">
          Benchmark noise-cancellation engines against golden transcripts using public recording URLs.
        </p>
      </header>

      {error && (
        <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card">
          <h2 className="mb-4 text-lg font-medium">Import dataset</h2>
          <div className="space-y-3">
            <input
              className="input"
              value={importName}
              onChange={(e) => setImportName(e.target.value)}
              placeholder="Dataset name"
            />
            <input
              className="input"
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              placeholder="Absolute path to manifest JSON"
            />
            <button className="btn" onClick={handleImportPath}>
              Import from path
            </button>
          </div>
          <div className="mt-6">
            <h3 className="mb-2 text-sm font-medium text-slate-700">Datasets</h3>
            {loading ? (
              <p className="text-sm text-slate-500">Loading...</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {datasets.map((d) => (
                  <li key={d.id} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                    <span>
                      <strong>{d.name}</strong> — {d.call_count} calls
                    </span>
                    <button className="text-xs text-slate-600 underline" onClick={() => setDatasetId(d.id)}>
                      select
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section className="card">
          <h2 className="mb-4 text-lg font-medium">Start benchmark run</h2>
          <div className="space-y-3">
            <input
              className="input"
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              placeholder="Run name"
            />
            <select
              className="input"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value ? Number(e.target.value) : "")}
            >
              <option value="">Select dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.call_count})
                </option>
              ))}
            </select>
            <select className="input" value={tier} onChange={(e) => setTier(e.target.value)}>
              <option value="tier_a">Tier A — none, dtln, hush, hecttor</option>
              <option value="tier_b">Tier B — sanas (Linux Docker)</option>
            </select>
            <button className="btn" onClick={handleCreateRun} disabled={!datasetId}>
              Start run
            </button>
          </div>
        </section>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <section className="card lg:col-span-1">
          <h2 className="mb-4 text-lg font-medium">Runs</h2>
          <ul className="space-y-2 text-sm">
            {runs.map((run) => (
              <li key={run.id}>
                <button
                  className={`w-full rounded-lg border px-3 py-2 text-left ${
                    selectedRunId === run.id ? "border-slate-900 bg-slate-50" : "border-slate-200"
                  }`}
                  onClick={() => setSelectedRunId(run.id)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">{run.name}</span>
                    {statusBadge(run.status)}
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    {run.tier} · {run.completed_calls}/{run.total_calls} · {run.progress}%
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="card lg:col-span-2">
          <h2 className="mb-4 text-lg font-medium">Engine ranking</h2>
          <RankingTable summary={selectedRun?.summary_json || undefined} />

          {selectedRun && (
            <div className="mt-6">
              <h3 className="mb-2 text-sm font-medium text-slate-700">Per-call results</h3>
              <div className="max-h-80 overflow-y-auto text-sm">
                {results.map((r) => (
                  <div key={r.id} className="mb-2 rounded-lg border border-slate-100 px-3 py-2">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs">{r.call_log_id}</span>
                      {statusBadge(r.status)}
                    </div>
                    {r.error_message && (
                      <p className="mt-1 text-xs text-rose-600">{r.error_message}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
