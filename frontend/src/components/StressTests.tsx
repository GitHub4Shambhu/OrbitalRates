"use client";

import { StressResult } from "@/lib/api";

export default function StressTests({ data }: { data: StressResult[] }) {
  if (data.length === 0) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No stress scenarios executed
      </div>
    );
  }

  const allPass = data.every(s => s.survival);

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Stress Scenarios
        </h2>
        <span className={`text-xs px-2 py-0.5 rounded-full border ${
          allPass
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
            : "border-red-500/30 bg-red-500/10 text-red-400"
        }`}>
          {allPass ? "ALL PASS" : "BREACH"}
        </span>
      </div>

      <div className="space-y-3">
        {data.map((s) => (
          <div
            key={s.scenario_name}
            className={`p-3 rounded-lg border ${
              s.survival ? "border-neutral-800 bg-neutral-900/30" : "border-red-500/30 bg-red-500/5"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className={`text-sm ${s.survival ? "text-emerald-400" : "text-red-400"}`}>
                  {s.survival ? "✓" : "✗"}
                </span>
                <span className="text-sm font-medium text-neutral-200">{s.scenario_name}</span>
              </div>
              <span className={`text-sm font-mono font-bold ${
                s.portfolio_loss_pct > 3 ? "text-red-400" : "text-amber-400"
              }`}>
                -{s.portfolio_loss_pct.toFixed(2)}%
              </span>
            </div>
            <p className="text-[10px] text-neutral-500 ml-6">{s.description}</p>
            {s.margin_call_triggered && (
              <p className="text-[10px] text-red-400 ml-6 mt-1">⚠ Margin call triggered</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
