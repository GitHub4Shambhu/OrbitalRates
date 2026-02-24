"use client";

import { ExecutionResult } from "@/lib/api";

export default function ExecutionSim({ data }: { data: ExecutionResult[] }) {
  if (data.length === 0) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No executions this cycle
      </div>
    );
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Execution Simulation
        </h2>
        <span className="text-xs px-2 py-0.5 rounded-full border border-pink-500/30 bg-pink-500/10 text-pink-400">
          Layer 6
        </span>
      </div>

      <div className="space-y-3">
        {data.map((e) => {
          const fillPct = e.fill_rate * 100;
          const fillColor = fillPct >= 90 ? "text-emerald-400" : fillPct >= 70 ? "text-amber-400" : "text-red-400";

          return (
            <div key={e.spread_id} className="p-3 rounded-lg bg-neutral-900/30 border border-neutral-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono text-neutral-300">{e.spread_id}</span>
                <span className={`text-xs font-mono font-bold ${fillColor}`}>
                  {fillPct.toFixed(0)}% fill
                </span>
              </div>

              <div className="grid grid-cols-4 gap-3 text-[10px]">
                <div>
                  <p className="text-neutral-500">Intended</p>
                  <p className="text-neutral-200 font-mono">${(e.intended_notional / 1e6).toFixed(1)}M</p>
                </div>
                <div>
                  <p className="text-neutral-500">Executed</p>
                  <p className="text-neutral-200 font-mono">${(e.executed_notional / 1e6).toFixed(1)}M</p>
                </div>
                <div>
                  <p className="text-neutral-500">Slippage</p>
                  <p className={`font-mono ${e.slippage_bps > 5 ? "text-amber-400" : "text-neutral-200"}`}>
                    {e.slippage_bps.toFixed(1)} bps
                  </p>
                </div>
                <div>
                  <p className="text-neutral-500">Impact</p>
                  <p className="text-neutral-200 font-mono">{e.market_impact_bps.toFixed(1)} bps</p>
                </div>
              </div>

              {/* Fill rate bar */}
              <div className="mt-2 h-1 bg-neutral-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${fillPct >= 90 ? "bg-emerald-500" : fillPct >= 70 ? "bg-amber-500" : "bg-red-500"}`}
                  style={{ width: `${fillPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
