import { useCallback, useEffect, useMemo, useState } from "react";
import { api, CallResult, Dataset, Diagnostics, Run } from "./api";

function statusBadge(status: string) {
  const cls =
    status.includes("fail") || status.includes("missing")
      ? "badge badge-failed"
      : status.includes("running")
        ? "badge badge-running"
        : "badge badge-completed";
  return <span className={cls}>{status}</span>;
}

type EngineRow = {
  word_weighted_wer_pct?: number | null;
  turn_avg_wer_pct?: number | null;
  calls?: number;
  status?: string;
  skip_count?: number;
  skip_reason?: string;
  primary_reason?: string | null;
  high_wer_reasons?: string[];
};

function RankingTable({ summary }: { summary: Record<string, unknown> | null | undefined }) {
  const combined = (summary?.combined as Record<string, EngineRow>) || {};
  const ranking = (summary?.ranking_by_word_weighted_wer as string[]) || [];
  const expected = (summary?.expected_engines as string[]) || Object.keys(combined);
  const skippedCounts = (summary?.skipped_engine_counts as Record<string, number>) || {};
  const skippedReasons = (summary?.skipped_engine_reasons as Record<string, string>) || {};

  const labels = useMemo(() => {
    const ordered: string[] = [];
    for (const label of ranking) {
      if (!ordered.includes(label)) ordered.push(label);
    }
    for (const label of expected) {
      if (!ordered.includes(label)) ordered.push(label);
    }
    for (const label of Object.keys(combined)) {
      if (!ordered.includes(label)) ordered.push(label);
    }
    return ordered;
  }, [ranking, expected, combined]);

  if (!labels.length) {
    return <p className="text-sm text-slate-500">No ranking yet.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2 pr-4">Rank</th>
              <th className="py-2 pr-4">Engine</th>
              <th className="py-2 pr-4">Word-weighted WER</th>
              <th className="py-2 pr-4">Turn-avg WER</th>
              <th className="py-2 pr-4">Calls</th>
              <th className="py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {labels.map((engine) => {
              const row = combined[engine] || {};
              const ok = row.status !== "missing" && row.word_weighted_wer_pct != null;
              const rankIdx = ranking.indexOf(engine);
              const reason = row.skip_reason || skippedReasons[engine];
              return (
                <tr key={engine} className="border-b border-slate-100 align-top">
                  <td className="py-2 pr-4">{ok && rankIdx >= 0 ? rankIdx + 1 : "—"}</td>
                  <td className="py-2 pr-4 font-medium">
                    {engine}
                    {!ok && reason && (
                      <div className="mt-1 max-w-md text-[11px] font-normal text-rose-600">
                        {reason}
                      </div>
                    )}
                    {ok && row.primary_reason && (row.word_weighted_wer_pct ?? 0) >= 35 && (
                      <div className="mt-1 max-w-lg rounded bg-amber-50 px-2 py-1 text-[11px] font-normal text-amber-950">
                        <span className="font-medium">Reason:</span> {row.primary_reason}
                      </div>
                    )}
                  </td>
                  <td className="py-2 pr-4">
                    {row.word_weighted_wer_pct != null ? `${row.word_weighted_wer_pct}%` : "—"}
                  </td>
                  <td className="py-2 pr-4">
                    {row.turn_avg_wer_pct != null ? `${row.turn_avg_wer_pct}%` : "—"}
                  </td>
                  <td className="py-2 pr-4">{row.calls ?? 0}</td>
                  <td className="py-2">
                    {ok
                      ? statusBadge("ok")
                      : statusBadge(row.status === "missing" ? "missing" : "failed")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {Object.keys(skippedCounts).length > 0 && (
        <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <div className="font-medium">Failed engines (with reason)</div>
          <ul className="mt-1 list-disc pl-4">
            {Object.entries(skippedCounts).map(([label, count]) => (
              <li key={label}>
                <code>{label}</code>: {count} calls
                {skippedReasons[label] ? ` — ${skippedReasons[label]}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function CallEngineTable({ report }: { report: Record<string, unknown> }) {
  const engines = (report.engines as Record<string, Record<string, unknown>>) || {};
  const skipped = (report.skipped_engines as Record<string, string>) || {};
  const labels = [
    ...Object.keys(engines),
    ...Object.keys(skipped).filter((k) => !(k in engines)),
  ];
  const [openEngine, setOpenEngine] = useState<string | null>(null);

  if (!labels.length) {
    return <p className="text-xs text-slate-500">No engine data.</p>;
  }

  return (
    <div className="mt-2 overflow-x-auto">
      <div className="mb-1 text-[11px] text-slate-500">
        audio_is_human_track: {String(report.audio_is_human_track ?? "unknown")}
      </div>
      <table className="min-w-full text-xs">
        <thead>
          <tr className="border-b text-left text-slate-500">
            <th className="py-1 pr-3">Engine</th>
            <th className="py-1 pr-3">WER</th>
            <th className="py-1 pr-3">Δ vs none</th>
            <th className="py-1">Calculation + reason</th>
          </tr>
        </thead>
        <tbody>
          {labels.map((label) => {
            const eng = engines[label];
            if (!eng) {
              return (
                <tr key={label} className="border-b border-slate-50">
                  <td className="py-1 pr-3 font-medium">{label}</td>
                  <td className="py-1 pr-3 text-rose-600" colSpan={3}>
                    SKIP: {skipped[label]}
                  </td>
                </tr>
              );
            }
            const turns = (eng.turns as Array<Record<string, unknown>>) || [];
            const delta =
              eng.delta_wer_vs_none != null
                ? `${(Number(eng.delta_wer_vs_none) * 100).toFixed(1)} pp`
                : "—";
            const edits = (eng.edit_totals as Record<string, unknown>) || {};
            const reasons = (eng.high_wer_reasons as string[]) || [];
            const primary = String(eng.primary_reason || reasons[0] || "");
            const isOpen = openEngine === label;
            const showWhy = Number(eng.avg_wer_pct ?? 0) >= 35 || reasons.length > 0;
            return (
              <tr key={label} className="border-b border-slate-50 align-top">
                <td className="py-1 pr-3 font-medium">
                  <button
                    className="text-left underline-offset-2 hover:underline"
                    onClick={() => setOpenEngine(isOpen ? null : label)}
                  >
                    {label}
                  </button>
                </td>
                <td className="py-1 pr-3">{String(eng.avg_wer_pct ?? "—")}%</td>
                <td className="py-1 pr-3">{delta}</td>
                <td className="py-1 text-slate-700">
                  {edits.formula != null && (
                    <div className="font-mono text-[11px] text-slate-600">
                      Calculation: {String(edits.formula)}
                      <span className="ml-1 text-slate-500">
                        (S={String(edits.substitutions)} D={String(edits.deletions)} I=
                        {String(edits.insertions)})
                      </span>
                    </div>
                  )}
                  {showWhy && primary && (
                    <div className="mt-1 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-950">
                      <span className="font-medium">Reason:</span> {primary}
                    </div>
                  )}
                  {showWhy && reasons.length > 1 && (
                    <ul className="mt-1 list-disc pl-4 text-[11px] text-amber-900">
                      {reasons.slice(1).map((r) => (
                        <li key={r}>{r}</li>
                      ))}
                    </ul>
                  )}
                  {!showWhy && turns[0] && (
                    <div className="text-slate-500">
                      <div>
                        <span className="text-slate-400">ref:</span>{" "}
                        {String(turns[0].reference || "")}
                      </div>
                      <div>
                        <span className="text-slate-400">hyp:</span>{" "}
                        {String(turns[0].hypothesis || "")}
                      </div>
                    </div>
                  )}
                  {isOpen && (
                    <div className="mt-2 space-y-2 rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="text-[11px] font-medium text-slate-700">
                        Per-turn breakdown
                      </div>
                      {turns.map((t) => {
                        const detail =
                          (t.wer_detail as Record<string, unknown> | undefined) || {};
                        const turnReasons = (detail.reasons as string[]) || [];
                        return (
                          <div
                            key={String(t.turn)}
                            className="border-t border-slate-200 pt-2 text-[11px]"
                          >
                            <div className="font-medium">
                              Turn {String(t.turn)} — {String(t.wer_pct ?? "—")}%
                            </div>
                            {detail.formula ? (
                              <div className="font-mono text-[10px] text-slate-500">
                                Calculation: {String(detail.formula)}
                              </div>
                            ) : null}
                            <div>
                              <span className="text-slate-400">ref:</span>{" "}
                              {String(t.reference || "")}
                            </div>
                            <div>
                              <span className="text-slate-400">hyp:</span>{" "}
                              {String(t.hypothesis || "")}
                            </div>
                            {turnReasons.length > 0 && (
                              <div className="mt-1 space-y-0.5">
                                {turnReasons.map((r) => (
                                  <div key={r} className="text-amber-900">
                                    Reason: {r}
                                  </div>
                                ))}
                              </div>
                            )}
                            {Array.isArray(detail.substitution_pairs) &&
                              (detail.substitution_pairs as string[]).length > 0 && (
                                <div className="text-slate-600">
                                  subs:{" "}
                                  {(detail.substitution_pairs as string[]).join(", ")}
                                </div>
                              )}
                            {Array.isArray(detail.deleted_words) &&
                              (detail.deleted_words as string[]).length > 0 && (
                                <div className="text-slate-600">
                                  deleted:{" "}
                                  {(detail.deleted_words as string[]).join(", ")}
                                </div>
                              )}
                            {Array.isArray(detail.inserted_words) &&
                              (detail.inserted_words as string[]).length > 0 && (
                                <div className="text-slate-600">
                                  inserted:{" "}
                                  {(detail.inserted_words as string[]).join(", ")}
                                </div>
                              )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <button
                    className="mt-1 text-[10px] text-slate-500 underline"
                    onClick={() => setOpenEngine(isOpen ? null : label)}
                  >
                    {isOpen ? "Hide turn details" : "Show turn details"}
                  </button>
                </td>
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
  const [expandedCall, setExpandedCall] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [importName, setImportName] = useState("muthoot");
  const [importPath, setImportPath] = useState(
    "/Users/yash.jha.ext/Desktop/Muthoot_final_with_public_urls.json"
  );
  const [runName, setRunName] = useState("golden-tier-a");
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [tier, setTier] = useState("tier_a");
  const [turnAlign, setTurnAlign] = useState("forced");
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);

  const selectedRun = useMemo(
    () => runs.find((r) => r.id === selectedRunId) || null,
    [runs, selectedRunId]
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [ds, rs, diag] = await Promise.all([
        api.listDatasets(),
        api.listRuns(),
        api.diagnostics().catch(() => null),
      ]);
      setDatasets(ds);
      setRuns(rs);
      setDiagnostics(diag);
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
          // In full mode turn_align is unused; send a valid placeholder since
          // the worker only accepts timestamp/vad/forced for --turn-align.
          turn_align: turnAlign === "full" ? "vad" : turnAlign,
          segment_mode: turnAlign === "full" ? "full" : "turn",
          stt_provider: "deepgram",
          stt_model: "nova-2",
          stt_language: "hi",
        },
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

      {diagnostics && (
        <div
          className={`mb-4 rounded-lg border px-4 py-3 text-sm ${
            diagnostics.ok
              ? "border-emerald-200 bg-emerald-50 text-emerald-900"
              : "border-amber-200 bg-amber-50 text-amber-950"
          }`}
        >
          <div className="font-medium">
            Engine diagnostics: {diagnostics.ok ? "ready" : "not ready"}
          </div>
          <p className="mt-1">{diagnostics.message}</p>
          {diagnostics.hints && diagnostics.hints.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs">
              {diagnostics.hints.map((hint) => (
                <li key={hint}>{hint}</li>
              ))}
            </ul>
          )}
          {diagnostics.engines && (
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              {Object.entries(diagnostics.engines).map(([name, result]) => (
                <span
                  key={name}
                  className={`rounded px-2 py-0.5 ${
                    result.ok ? "bg-emerald-100" : "bg-rose-100 text-rose-800"
                  }`}
                  title={result.error || result.processor || ""}
                >
                  {name}: {result.ok ? "ok" : "fail"}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

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
                  <li
                    key={d.id}
                    className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2"
                  >
                    <span>
                      <strong>{d.name}</strong> — {d.call_count} calls
                    </span>
                    <button
                      className="text-xs text-slate-600 underline"
                      onClick={() => setDatasetId(d.id)}
                    >
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
              <option value="tier_a">
                Tier A — none, dtln, hush, hecttor (all 5 models)
              </option>
              <option value="tier_b">Tier B — sanas (both models, Linux Docker)</option>
            </select>
            <label className="block text-sm font-medium text-slate-700">
              Turn alignment
              <select
                className="input mt-1"
                value={turnAlign}
                onChange={(e) => setTurnAlign(e.target.value)}
              >
                <option value="forced">
                  Forced alignment — align reference to audio (ignores createdAt)
                </option>
                <option value="vad">VAD — trim createdAt windows to speech</option>
                <option value="timestamp">Timestamp — raw createdAt windows</option>
                <option value="full">Full recording — transcribe whole call at once</option>
              </select>
            </label>
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
          <h2 className="mb-4 text-lg font-medium">Engine ranking (all models)</h2>
          <RankingTable summary={selectedRun?.summary_json || undefined} />

          {selectedRun && (
            <div className="mt-6">
              <h3 className="mb-2 text-sm font-medium text-slate-700">Per-call reports</h3>
              <div className="max-h-[32rem] space-y-2 overflow-y-auto text-sm">
                {results.map((r) => (
                  <div key={r.id} className="rounded-lg border border-slate-100 px-3 py-2">
                    <button
                      className="flex w-full items-center justify-between text-left"
                      onClick={() =>
                        setExpandedCall((prev) => (prev === r.id ? null : r.id))
                      }
                    >
                      <span className="font-mono text-xs">{r.call_log_id}</span>
                      {statusBadge(r.status)}
                    </button>
                    {r.error_message && (
                      <p className="mt-1 text-xs text-rose-600">{r.error_message}</p>
                    )}
                    {expandedCall === r.id && r.report_json && (
                      <CallEngineTable report={r.report_json as Record<string, unknown>} />
                    )}
                    {expandedCall === r.id && !r.report_json && (
                      <p className="mt-2 text-xs text-slate-500">No report JSON yet.</p>
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
