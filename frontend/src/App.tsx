import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  CallResult,
  Dataset,
  Diagnostics,
  Run,
  SweepResponse,
  SweepResult,
  TaskProgress,
} from "./api";

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
      if (label !== "none" && !ordered.includes(label)) ordered.push(label);
    }
    for (const label of expected) {
      if (label !== "none" && !ordered.includes(label)) ordered.push(label);
    }
    for (const label of Object.keys(combined)) {
      if (label !== "none" && !ordered.includes(label)) ordered.push(label);
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

const DTLN_HUSH_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const HECTTOR_STRENGTHS = [0.25, 0.5, 0.75, 0.8, 0.9, 1.0];
const STT_PRESETS = [
  { id: "deepgram-nova-2", label: "Deepgram Nova-2" },
  { id: "deepgram-nova-3", label: "Deepgram Nova-3" },
  { id: "google-chirp-3", label: "Google Chirp 3" },
  { id: "sarvam-saaras-v3", label: "Sarvam Saaras v3" },
] as const;

/** Tier A: none + dtln + hush + 5 hecttor sub-models */
const TIER_A_NC_MODEL_COUNT = 8;
const HECTTOR_SUBMODEL_COUNT = 5;

type NcConfigBreakdown = {
  none: number;
  dtln: number;
  hush: number;
  hecttor: number;
  /** Configs scored by this run */
  inRun: number;
  /** Full comparison grid incl. none baseline */
  full: number;
};

type RunJobEstimate = {
  calls: number;
  sttCount: number;
  mode: "sweep" | "tier";
  subprocessJobs: number;
  /** NC evaluations this run produces */
  ncEvaluations: number;
  /** calls × STTs × full config grid (51 when all strengths selected) */
  ncEvaluationsFullGrid: number;
  strengthTrials: number;
  configBreakdown: NcConfigBreakdown;
};

function estimateRunJobs(input: {
  callCount: number;
  sttCount: number;
  dtlnStrengths: number;
  hushStrengths: number;
  hecttorStrengths: number;
  isSweep: boolean;
}): RunJobEstimate | null {
  const { callCount, sttCount, dtlnStrengths, hushStrengths, hecttorStrengths, isSweep } =
    input;
  if (!callCount || !sttCount) return null;

  const noneConfigs = 1;
  const hecttorConfigs = hecttorStrengths * HECTTOR_SUBMODEL_COUNT;
  const fullConfigs = noneConfigs + dtlnStrengths + hushStrengths + hecttorConfigs;

  if (isSweep) {
    const strengthTrials = dtlnStrengths + hushStrengths + hecttorStrengths;
    if (strengthTrials === 0) return null;
    const inRunConfigs = dtlnStrengths + hushStrengths + hecttorConfigs;
    return {
      calls: callCount,
      sttCount,
      mode: "sweep",
      strengthTrials,
      subprocessJobs: callCount * sttCount * strengthTrials,
      ncEvaluations: callCount * sttCount * inRunConfigs,
      ncEvaluationsFullGrid: callCount * sttCount * fullConfigs,
      configBreakdown: {
        none: noneConfigs,
        dtln: dtlnStrengths,
        hush: hushStrengths,
        hecttor: hecttorConfigs,
        inRun: inRunConfigs,
        full: fullConfigs,
      },
    };
  }

  return {
    calls: callCount,
    sttCount,
    mode: "tier",
    strengthTrials: 0,
    subprocessJobs: callCount * sttCount,
    ncEvaluations: callCount * sttCount * TIER_A_NC_MODEL_COUNT,
    ncEvaluationsFullGrid: callCount * sttCount * TIER_A_NC_MODEL_COUNT,
    configBreakdown: {
      none: noneConfigs,
      dtln: 1,
      hush: 1,
      hecttor: HECTTOR_SUBMODEL_COUNT,
      inRun: TIER_A_NC_MODEL_COUNT,
      full: TIER_A_NC_MODEL_COUNT,
    },
  };
}

function JobEstimatePanel({ estimate }: { estimate: RunJobEstimate | null }) {
  if (!estimate) {
    return (
      <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
        Select a dataset and at least one STT to see job estimates.
      </div>
    );
  }

  const fmt = (n: number) => n.toLocaleString();
  const { configBreakdown: b } = estimate;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-xs text-slate-700">
      <div className="mb-2 font-medium text-slate-900">Estimated workload</div>

      <div className="mb-3 overflow-x-auto">
        <table className="min-w-full text-[11px]">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-1 pr-3 font-medium">NC component</th>
              <th className="py-1 pr-3 font-medium">Configs / call / STT</th>
              <th className="py-1 font-medium">Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-100">
              <td className="py-1.5 pr-3">none (baseline)</td>
              <td className="py-1.5 pr-3 font-medium">{b.none}</td>
              <td className="py-1.5 text-slate-500">
                {estimate.mode === "sweep"
                  ? "Tier A run only — not in strength sweep"
                  : "included in Tier A run"}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1.5 pr-3">dtln</td>
              <td className="py-1.5 pr-3 font-medium">{b.dtln}</td>
              <td className="py-1.5 text-slate-500">
                {estimate.mode === "sweep" ? `${b.dtln} strength settings` : "default strength"}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1.5 pr-3">hush</td>
              <td className="py-1.5 pr-3 font-medium">{b.hush}</td>
              <td className="py-1.5 text-slate-500">
                {estimate.mode === "sweep" ? `${b.hush} strength settings` : "default strength"}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1.5 pr-3">hecttor (5 sub-models)</td>
              <td className="py-1.5 pr-3 font-medium">{b.hecttor}</td>
              <td className="py-1.5 text-slate-500">
                {estimate.mode === "sweep"
                  ? `${estimate.configBreakdown.hecttor / HECTTOR_SUBMODEL_COUNT || 0} strengths × ${HECTTOR_SUBMODEL_COUNT} models`
                  : `${HECTTOR_SUBMODEL_COUNT} sub-models`}
              </td>
            </tr>
            <tr className="font-medium text-slate-900">
              <td className="py-1.5 pr-3">Total configs / call / STT</td>
              <td className="py-1.5 pr-3">
                {b.inRun}
                {estimate.mode === "sweep" && b.full !== b.inRun ? (
                  <span className="font-normal text-slate-500"> ({b.full} with none)</span>
                ) : null}
              </td>
              <td className="py-1.5 font-normal text-slate-500">
                {estimate.mode === "sweep"
                  ? "Compare best NC + strength + STT"
                  : "8 models at default strength"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <dl className="grid gap-1.5 sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">Calls × STTs</dt>
          <dd className="font-medium">
            {fmt(estimate.calls)} × {estimate.sttCount} ={" "}
            {fmt(estimate.calls * estimate.sttCount)}
          </dd>
        </div>
        <div>
          <dt className="text-slate-500">Subprocess jobs (this run)</dt>
          <dd className="font-medium">{fmt(estimate.subprocessJobs)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">NC evaluations (this run)</dt>
          <dd className="font-medium">{fmt(estimate.ncEvaluations)}</dd>
        </div>
        {estimate.mode === "sweep" && estimate.ncEvaluationsFullGrid !== estimate.ncEvaluations ? (
          <div>
            <dt className="text-slate-500">Full grid incl. none baseline</dt>
            <dd className="font-medium">{fmt(estimate.ncEvaluationsFullGrid)}</dd>
          </div>
        ) : null}
      </dl>

      <p className="mt-2 text-[11px] text-slate-500">
        {estimate.mode === "sweep" ? (
          <>
            Strength sweep: {estimate.strengthTrials} subprocess jobs per call/STT ({b.dtln}{" "}
            DTLN + {b.hush} Hush + {b.hecttor / HECTTOR_SUBMODEL_COUNT} Hecttor trials; each
            Hecttor trial scores all {HECTTOR_SUBMODEL_COUNT} sub-models). Add a Tier A run for
            the none baseline ({fmt(estimate.calls * estimate.sttCount * b.none)} extra
            evaluations) to reach {fmt(estimate.ncEvaluationsFullGrid)} total.
          </>
        ) : (
          <>
            Tier A: one job per call/STT scoring all {TIER_A_NC_MODEL_COUNT} NC models (none, dtln,
            hush, {HECTTOR_SUBMODEL_COUNT}× hecttor) at default strength.
          </>
        )}
      </p>
    </div>
  );
}

function RunProgressPanel({
  progress,
  title,
}: {
  progress: TaskProgress | null | undefined;
  title?: string;
}) {
  if (!progress) return null;

  const fmt = (n: number) => n.toLocaleString();
  const pct = Math.min(100, Math.max(0, progress.progress_pct));

  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-xs text-slate-800">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="font-medium text-slate-900">
          {title ?? (progress.task_type === "sweep" ? "Strength sweep progress" : "Run progress")}
        </span>
        {statusBadge(progress.status)}
      </div>

      <div className="mb-2 h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-slate-900 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      <dl className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <dt className="text-slate-500">Jobs</dt>
          <dd className="font-medium">
            {fmt(progress.jobs_completed)} / {fmt(progress.jobs_total)} ({pct}%)
          </dd>
        </div>
        <div>
          <dt className="text-slate-500">Calls</dt>
          <dd className="font-medium">
            {fmt(progress.calls_completed)} fully done · {fmt(progress.calls_touched)} touched /{" "}
            {fmt(progress.calls_total)}
          </dd>
        </div>
        <div>
          <dt className="text-slate-500">Failed jobs</dt>
          <dd className="font-medium">{fmt(progress.jobs_failed)}</dd>
        </div>
        {progress.jobs_cached > 0 ? (
          <div>
            <dt className="text-slate-500">Cached</dt>
            <dd className="font-medium">{fmt(progress.jobs_cached)}</dd>
          </div>
        ) : null}
      </dl>

      {progress.stt_stats.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 font-medium text-slate-700">STT progress</div>
          <div className="space-y-1.5">
            {progress.stt_stats.map((stt) => (
              <div key={stt.stt_id}>
                <div className="flex justify-between text-[11px] text-slate-600">
                  <span>{stt.label}</span>
                  <span>
                    {stt.completed}/{stt.total} ({stt.pct}%)
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-blue-700 transition-all duration-500"
                    style={{ width: `${stt.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {progress.trial_stats.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 font-medium text-slate-700">
            Strength trials ({progress.trial_stats.filter((t) => t.pct >= 100).length} /{" "}
            {progress.trial_stats.length} complete)
          </div>
          <div className="max-h-40 space-y-1 overflow-y-auto">
            {progress.trial_stats.map((trial) => (
              <div key={trial.label} className="flex items-center gap-2 text-[11px]">
                <span className="w-28 shrink-0 truncate text-slate-600">{trial.label}</span>
                <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-emerald-600 transition-all duration-500"
                    style={{ width: `${trial.pct}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right text-slate-500">
                  {trial.completed}/{trial.total}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {progress.error && (
        <p className="mt-2 text-[11px] text-rose-700">{progress.error}</p>
      )}
    </div>
  );
}

function SweepResultsTable({
  sweepResult,
  onDownload,
}: {
  sweepResult: SweepResponse;
  onDownload: () => void;
}) {
  const sttGroups = useMemo(() => {
    if (!sweepResult?.results?.length) return [];
    const configs = sweepResult.stt_configs?.length
      ? sweepResult.stt_configs
      : [{ stt_id: "default", stt_provider: "deepgram", stt_model: "nova-2", stt_language: "hi" }];
    return configs.map((cfg) => {
      const rows = sweepResult.results.filter(
        (r) =>
          (r.stt_id && r.stt_id === cfg.stt_id) ||
          (r.stt_provider === cfg.stt_provider && r.stt_model === cfg.stt_model),
      );
      const engines = [...new Set(rows.map((r) => r.engine))];
      const strengths = [...new Set(rows.map((r) => r.strength))].sort((a, b) => a - b);
      const lookup = new Map<string, SweepResult>();
      for (const r of rows) lookup.set(`${r.engine}::${r.strength}`, r);
      return {
        cfg,
        rows,
        pivot: rows.length ? { engines, strengths, lookup } : null,
      };
    });
  }, [sweepResult]);

  if (!sttGroups.length || !sttGroups.some((g) => g.pivot)) return null;

  return (
    <div className="mt-6 space-y-8">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-700">
          Strength sweep — {sweepResult.dataset_name} ({sweepResult.total_calls} calls,
          align: {sweepResult.turn_align}
          {sweepResult.scoring ? `, scoring: ${sweepResult.scoring}` : ""}
          {sweepResult.execution_mode ? `, mode: ${sweepResult.execution_mode}` : ""}
          {sweepResult.workers ? `, workers: ${sweepResult.workers}` : ""}
          {typeof sweepResult.jobs_cached === "number"
            ? `, cached: ${sweepResult.jobs_cached}`
            : ""}
          )
        </h3>
        <button
          className="rounded bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
          onClick={onDownload}
        >
          Download JSON
        </button>
      </div>

      {sttGroups.map(({ cfg, pivot }) =>
        !pivot ? null : (
          <div key={cfg.stt_id}>
            <h4 className="mb-2 text-sm font-medium text-slate-800">
              STT: {cfg.stt_provider}/{cfg.stt_model}
            </h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-slate-500">
                    <th className="py-2 pr-4">Strength</th>
                    {pivot.engines.map((eng) => (
                      <th key={eng} className="py-2 px-3 text-center" colSpan={2}>
                        {eng}
                      </th>
                    ))}
                  </tr>
                  <tr className="border-b text-left text-[11px] text-slate-400">
                    <th className="py-1 pr-4"></th>
                    {pivot.engines.map((eng) => (
                      <Fragment key={eng}>
                        <th className="py-1 px-3 text-center font-normal">WW-WER</th>
                        <th className="py-1 px-3 text-center font-normal">Avg-WER</th>
                      </Fragment>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pivot.strengths.map((s) => (
                    <tr key={s} className="border-b border-slate-100">
                      <td className="py-2 pr-4 font-medium">{s}</td>
                      {pivot.engines.map((eng) => {
                        const r = pivot.lookup.get(`${eng}::${s}`);
                        if (!r) {
                          return (
                            <Fragment key={eng}>
                              <td className="py-2 px-3 text-center text-slate-400">—</td>
                              <td className="py-2 px-3 text-center text-slate-400">—</td>
                            </Fragment>
                          );
                        }
                        return (
                          <Fragment key={eng}>
                            <td className="py-2 px-3 text-center">
                              {r.word_weighted_wer_pct != null
                                ? `${r.word_weighted_wer_pct}%`
                                : "—"}
                            </td>
                            <td className="py-2 px-3 text-center">
                              {r.turn_avg_wer_pct != null ? `${r.turn_avg_wer_pct}%` : "—"}
                            </td>
                          </Fragment>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ),
      )}

      <div className="mt-4 overflow-x-auto">
        <h4 className="mb-2 text-xs font-medium text-slate-500">Detailed breakdown</h4>
        <table className="min-w-full text-xs">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-1 pr-3">STT</th>
              <th className="py-1 pr-3">Engine</th>
              <th className="py-1 pr-3">Strength</th>
              <th className="py-1 pr-3">WW-WER%</th>
              <th className="py-1 pr-3">Avg-WER%</th>
              <th className="py-1 pr-3">Calls</th>
              <th className="py-1 pr-3">Subs</th>
              <th className="py-1 pr-3">Dels</th>
              <th className="py-1 pr-3">Ins</th>
              <th className="py-1">Ref words</th>
            </tr>
          </thead>
          <tbody>
            {sweepResult.results.map((r, i) => {
              const sttLabel = r.stt_provider
                ? `${r.stt_provider}/${r.stt_model ?? ""}`
                : "—";
              const best =
                sweepResult.results
                  .filter(
                    (x) =>
                      x.engine === r.engine &&
                      x.stt_provider === r.stt_provider &&
                      x.stt_model === r.stt_model &&
                      x.word_weighted_wer_pct != null,
                  )
                  .sort(
                    (a, b) => (a.word_weighted_wer_pct ?? 999) - (b.word_weighted_wer_pct ?? 999),
                  )[0]?.strength === r.strength;
              return (
                <tr
                  key={i}
                  className={`border-b border-slate-50 ${best ? "bg-emerald-50" : ""}`}
                >
                  <td className="py-1 pr-3">{sttLabel}</td>
                  <td className="py-1 pr-3 font-medium">{r.engine}</td>
                  <td className="py-1 pr-3">{r.strength}</td>
                  <td className="py-1 pr-3">
                    {r.word_weighted_wer_pct != null ? `${r.word_weighted_wer_pct}%` : "—"}
                  </td>
                  <td className="py-1 pr-3">
                    {r.turn_avg_wer_pct != null ? `${r.turn_avg_wer_pct}%` : "—"}
                  </td>
                  <td className="py-1 pr-3">{r.calls}</td>
                  <td className="py-1 pr-3">{r.substitutions ?? "—"}</td>
                  <td className="py-1 pr-3">{r.deletions ?? "—"}</td>
                  <td className="py-1 pr-3">{r.insertions ?? "—"}</td>
                  <td className="py-1">{r.ref_words ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
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
  const [executionMode, setExecutionMode] = useState<"server" | "local">("server");
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);

  const [enableSweep, setEnableSweep] = useState(false);
  const [dtlnSelected, setDtlnSelected] = useState<Set<number>>(new Set());
  const [dtlnAll, setDtlnAll] = useState(false);
  const [hushSelected, setHushSelected] = useState<Set<number>>(new Set());
  const [hushAll, setHushAll] = useState(false);
  const [hecttorSelected, setHecttorSelected] = useState<Set<number>>(new Set());
  const [hecttorAll, setHecttorAll] = useState(false);
  const [sweepRunning, setSweepRunning] = useState(false);
  const [sweepResult, setSweepResult] = useState<SweepResponse | null>(null);
  const [activeSweepId, setActiveSweepId] = useState<string | null>(null);
  const [sweepProgress, setSweepProgress] = useState<TaskProgress | null>(null);
  const [workers, setWorkers] = useState(8);
  const [sttSelected, setSttSelected] = useState<Set<string>>(
    () => new Set(STT_PRESETS.map((p) => p.id)),
  );
  const [sttAll, setSttAll] = useState(true);

  const toggleStrength = (
    set: Set<number>,
    setter: (s: Set<number>) => void,
    val: number,
  ) => {
    const next = new Set(set);
    if (next.has(val)) next.delete(val);
    else next.add(val);
    setter(next);
  };

  const effectiveDtln = dtlnAll ? DTLN_HUSH_STRENGTHS : [...dtlnSelected].sort((a, b) => a - b);
  const effectiveHush = hushAll ? DTLN_HUSH_STRENGTHS : [...hushSelected].sort((a, b) => a - b);
  const effectiveHecttor = hecttorAll ? HECTTOR_STRENGTHS : [...hecttorSelected].sort((a, b) => a - b);
  const effectiveSttPresets = sttAll
    ? STT_PRESETS.map((p) => p.id)
    : [...sttSelected];

  const selectedDataset = useMemo(
    () => datasets.find((d) => d.id === datasetId) ?? null,
    [datasets, datasetId],
  );

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
    }, 2000);
    return () => clearInterval(timer);
  }, [selectedRunId]);

  const anyRunRunning = runs.some((r) => r.status === "running");

  useEffect(() => {
    if (!anyRunRunning) return;
    const timer = setInterval(async () => {
      try {
        const rs = await api.listRuns();
        setRuns(rs);
      } catch {
        /* ignore */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [anyRunRunning]);

  useEffect(() => {
    if (!activeSweepId || !sweepRunning) return;
    const poll = async () => {
      try {
        const snap = await api.getSweepProgress(activeSweepId);
        setSweepProgress(snap);
        if (snap.status === "completed" && snap.result) {
          setSweepResult(snap.result);
          setSweepRunning(false);
          setActiveSweepId(null);
        } else if (snap.status === "failed") {
          setError(snap.error || "Strength sweep failed");
          setSweepRunning(false);
          setActiveSweepId(null);
        }
      } catch {
        /* ignore transient 404 while sweep starts */
      }
    };
    poll();
    const timer = setInterval(poll, 2000);
    return () => clearInterval(timer);
  }, [activeSweepId, sweepRunning]);

  const handleImportPath = async () => {
    try {
      const ds = await api.importDatasetPath(importName, importPath, "Imported from local path");
      setDatasets((prev) => [ds, ...prev.filter((d) => d.id !== ds.id)]);
      setDatasetId(ds.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    }
  };

  const hasSweepSelection =
    effectiveDtln.length > 0 || effectiveHush.length > 0 || effectiveHecttor.length > 0;
  const hasSttSelection = effectiveSttPresets.length > 0;

  const runJobEstimate = useMemo(
    () =>
      estimateRunJobs({
        callCount: selectedDataset?.call_count ?? 0,
        sttCount: effectiveSttPresets.length,
        dtlnStrengths: effectiveDtln.length,
        hushStrengths: effectiveHush.length,
        hecttorStrengths: effectiveHecttor.length,
        isSweep: enableSweep && hasSweepSelection,
      }),
    [
      selectedDataset?.call_count,
      effectiveSttPresets.length,
      effectiveDtln.length,
      effectiveHush.length,
      effectiveHecttor.length,
      enableSweep,
      hasSweepSelection,
    ],
  );

  const handleCreateRun = async () => {
    if (!datasetId) return;
    if (!hasSttSelection) {
      setError("Select at least one STT model");
      return;
    }

    if (enableSweep && hasSweepSelection) {
      setSweepRunning(true);
      setSweepResult(null);
      setSweepProgress(null);
      setError(null);
      try {
        const alignValue = turnAlign === "full" ? "vad" : turnAlign;
        const start = await api.strengthSweep({
          dataset_id: Number(datasetId),
          dtln_strengths: effectiveDtln,
          hush_strengths: effectiveHush,
          hecttor_strengths: effectiveHecttor,
          turn_align: alignValue,
          scoring: "itn+oiwer",
          execution_mode: executionMode,
          workers,
          stt_preset_ids: effectiveSttPresets,
        });
        setActiveSweepId(start.sweep_id);
        setSweepProgress({
          task_id: start.sweep_id,
          task_type: "sweep",
          status: "running",
          progress_pct: 0,
          jobs_total: start.jobs_total,
          jobs_completed: 0,
          jobs_failed: 0,
          jobs_cached: 0,
          calls_total: start.total_calls,
          calls_touched: 0,
          calls_completed: 0,
          stt_stats: start.stt_configs.map((s) => ({
            stt_id: s.stt_id,
            label: s.stt_id,
            completed: 0,
            total: start.total_calls * start.trials_count,
            pct: 0,
          })),
          trial_stats: [],
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Strength sweep failed");
        setSweepRunning(false);
        setActiveSweepId(null);
      }
      return;
    }

    try {
      const run = await api.createRun({
        dataset_id: Number(datasetId),
        name: runName,
        tier,
        config: {
          turn_align: turnAlign === "full" ? "vad" : turnAlign,
          segment_mode: turnAlign === "full" ? "full" : "turn",
          scoring: "itn+oiwer",
          execution_mode: executionMode,
          stt_preset_ids: effectiveSttPresets,
          workers,
        },
      });
      setRuns((prev) => [run, ...prev]);
      setSelectedRunId(run.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run creation failed");
    }
  };

  const handleDownloadSweepJSON = () => {
    if (!sweepResult) return;
    const blob = new Blob([JSON.stringify(sweepResult, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `strength_sweep_${sweepResult.dataset_name || sweepResult.dataset_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
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
                Tier A — 8 NC models (none, dtln, hush, 5× hecttor)
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

            <label className="block text-sm font-medium text-slate-700">
              Execution mode
              <select
                className="input mt-1"
                value={executionMode}
                onChange={(e) => setExecutionMode(e.target.value as "server" | "local")}
              >
                <option value="server">
                  Server — multiprocessing (ProcessPool) + asyncio fan-out
                </option>
                <option value="local">
                  Local — single process, sequential across calls (asyncio STT only)
                </option>
              </select>
              <span className="mt-1 block text-xs text-slate-500">
                Both modes apply forced alignment + ITN + OI-WER + cross-script matching.
              </span>
            </label>

            <div className="rounded-lg border border-slate-200 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-slate-700">STT models</span>
                <label className="flex items-center gap-1.5 text-xs">
                  <input
                    type="checkbox"
                    checked={sttAll}
                    onChange={(e) => setSttAll(e.target.checked)}
                  />
                  All
                </label>
              </div>
              <div className="flex flex-wrap gap-2">
                {STT_PRESETS.map((preset) => (
                  <label
                    key={preset.id}
                    className={`flex cursor-pointer items-center gap-2 rounded px-2.5 py-1.5 text-xs ${
                      sttAll || sttSelected.has(preset.id)
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="sr-only"
                      checked={sttAll || sttSelected.has(preset.id)}
                      disabled={sttAll}
                      onChange={() => {
                        const next = new Set(sttSelected);
                        if (next.has(preset.id)) next.delete(preset.id);
                        else next.add(preset.id);
                        setSttSelected(next);
                      }}
                    />
                    {preset.label}
                  </label>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-slate-500">
                Runs iterate each selected STT across all NC engines/strengths.
              </p>
            </div>

            <label className="block">
              <span className="mb-1 block text-sm text-slate-600">
                Parallel workers (multiprocessing)
              </span>
              <input
                className="input w-28"
                type="number"
                min={1}
                max={16}
                value={executionMode === "local" ? 1 : workers}
                disabled={executionMode === "local"}
                onChange={(e) => setWorkers(Math.max(1, Number(e.target.value) || 1))}
              />
              <span className="ml-2 text-xs text-slate-500">
                {executionMode === "local"
                  ? "forced to 1 in local mode"
                  : "default 8 on n2-standard-16 (~2 CPUs each)"}
              </span>
            </label>

            <label className="flex items-center gap-2 pt-1">
              <input
                type="checkbox"
                checked={enableSweep}
                onChange={(e) => setEnableSweep(e.target.checked)}
              />
              <span className="text-sm font-medium text-slate-700">
                Strength sweep
              </span>
              <span className="text-xs text-slate-500">
                — compare WER at different NC strengths
              </span>
            </label>

            {enableSweep && (
              <div className="space-y-3 rounded-lg border border-slate-200 p-4">
                <div className="grid gap-4 lg:grid-cols-2">
                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium">DTLN strengths</span>
                      <label className="flex items-center gap-1.5 text-xs">
                        <input
                          type="checkbox"
                          checked={dtlnAll}
                          onChange={(e) => setDtlnAll(e.target.checked)}
                        />
                        All
                      </label>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {DTLN_HUSH_STRENGTHS.map((s) => (
                        <button
                          key={s}
                          className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                            dtlnAll || dtlnSelected.has(s)
                              ? "bg-slate-900 text-white"
                              : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                          }`}
                          onClick={() => {
                            if (!dtlnAll) toggleStrength(dtlnSelected, setDtlnSelected, s);
                          }}
                          disabled={dtlnAll}
                        >
                          {s}
                        </button>
                      ))}
                    </div>

                    <div className="mt-3 mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium">Hush strengths</span>
                      <label className="flex items-center gap-1.5 text-xs">
                        <input
                          type="checkbox"
                          checked={hushAll}
                          onChange={(e) => setHushAll(e.target.checked)}
                        />
                        All
                      </label>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {DTLN_HUSH_STRENGTHS.map((s) => (
                        <button
                          key={s}
                          className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                            hushAll || hushSelected.has(s)
                              ? "bg-slate-900 text-white"
                              : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                          }`}
                          onClick={() => {
                            if (!hushAll) toggleStrength(hushSelected, setHushSelected, s);
                          }}
                          disabled={hushAll}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium">Hecttor strengths</span>
                      <label className="flex items-center gap-1.5 text-xs">
                        <input
                          type="checkbox"
                          checked={hecttorAll}
                          onChange={(e) => setHecttorAll(e.target.checked)}
                        />
                        All
                      </label>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {HECTTOR_STRENGTHS.map((s) => (
                        <button
                          key={s}
                          className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                            hecttorAll || hecttorSelected.has(s)
                              ? "bg-slate-900 text-white"
                              : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                          }`}
                          onClick={() => {
                            if (!hecttorAll) toggleStrength(hecttorSelected, setHecttorSelected, s);
                          }}
                          disabled={hecttorAll}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    <p className="mt-2 text-[11px] text-slate-500">
                      Runs all Hecttor models configured in the worker.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <JobEstimatePanel estimate={runJobEstimate} />

            {(sweepRunning && sweepProgress) || selectedRun?.status === "running" ? (
              <RunProgressPanel
                progress={
                  sweepRunning
                    ? sweepProgress
                    : (selectedRun?.progress_detail as TaskProgress | undefined)
                }
              />
            ) : null}

            <button
              className="btn"
              onClick={handleCreateRun}
              disabled={!datasetId || sweepRunning}
            >
              {sweepRunning
                ? "Running strength sweep..."
                : enableSweep && hasSweepSelection
                  ? "Run strength sweep"
                  : "Start run"}
            </button>
            {sweepRunning && (
              <p className="text-xs text-slate-500">
                Strength sweep running — progress updates every few seconds.
              </p>
            )}
          </div>
        </section>
      </div>

      {sweepResult && (
        <SweepResultsTable sweepResult={sweepResult} onDownload={handleDownloadSweepJSON} />
      )}

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
                    {run.tier}
                    {run.jobs_total ? (
                      <> · jobs {run.completed_calls}/{run.jobs_total}</>
                    ) : (
                      <> · {run.completed_calls}/{run.total_calls} calls</>
                    )}
                    {" · "}
                    {run.progress}%
                  </div>
                  {run.status === "running" && (
                    <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-blue-600 transition-all duration-500"
                        style={{ width: `${Math.min(100, run.progress)}%` }}
                      />
                    </div>
                  )}
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
