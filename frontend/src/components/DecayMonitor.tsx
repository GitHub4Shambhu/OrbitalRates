"use client";

import { DecayMetrics } from "@/lib/api";

export default function DecayMonitor({ data }: { data: DecayMetrics | null }) {
  if (!data) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No decay metrics — awaiting cycle
      </div>
    );
  }

  const warn = data.retraining_recommended;

  return (
    <div className={`card p-6 border ${warn ? "border-amber-500/30" : "border-neutral-800"}`}>
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Strategy Decay / Meta-Learning
        </h2>
        <div className="flex items-center gap-2">
          {warn && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 font-bold">
              RETRAIN SUGGESTED
            </span>
          )}
          <span className="text-xs px-2 py-0.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-400">
            Layer 7
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <MetricBar label="Halflife Drift" value={data.halflife_drift_pct} />
        <MetricBar label="Edge Decay" value={data.edge_decay_pct} />
        <MetricBar label="Regime Freq Δ" value={data.regime_frequency_change} />
      </div>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <MetricBar label="Correlation Clustering" value={data.correlation_clustering} />
        <MetricBar label="Crowding Increase" value={data.crowding_increase} />
        <div className="flex items-center justify-center">
          <span className={`text-xs font-mono ${warn ? "text-amber-400" : "text-emerald-400"}`}>
            {warn ? "⚠ Degraded" : "✓ Healthy"}
          </span>
        </div>
      </div>

      {Object.keys(data.parameter_adjustments).length > 0 && (
        <div className="pt-3 border-t border-neutral-800">
          <p className="text-[10px] text-neutral-500 mb-2">Auto-Adjustments Applied</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.parameter_adjustments).map(([key, val]) => (
              <span key={key} className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 font-mono">
                {key}: {typeof val === "number" ? val.toFixed(2) : val}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.abs(value) * 100);
  const color = pct > 20 ? "bg-red-500" : pct > 10 ? "bg-amber-500" : "bg-emerald-500";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-neutral-500">{label}</span>
        <span className="text-neutral-300 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1 bg-neutral-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
