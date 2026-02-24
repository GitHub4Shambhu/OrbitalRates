"use client";

import { SpreadCandidate } from "@/lib/api";

const SIGNAL_STYLE: Record<string, string> = {
  enter_long:  "text-emerald-400 bg-emerald-500/10",
  enter_short: "text-red-400 bg-red-500/10",
  exit:        "text-amber-400 bg-amber-500/10",
  hold:        "text-neutral-400 bg-neutral-500/10",
  reject:      "text-neutral-600 bg-neutral-800/50",
};

export default function SpreadOpportunities({ data, total }: { data: SpreadCandidate[]; total: number }) {
  const active = data.filter(s => s.signal !== "reject");
  const rejected = data.filter(s => s.signal === "reject");

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Spread Discovery
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500 font-mono">{active.length} active / {total} scanned</span>
          <span className="text-xs px-2 py-0.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-400">
            Layer 2
          </span>
        </div>
      </div>

      {active.length === 0 ? (
        <p className="text-sm text-neutral-500 text-center py-4">No active spread opportunities</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-neutral-500 border-b border-neutral-800">
                <th className="text-left py-2 px-2">Spread</th>
                <th className="text-left py-2 px-1">Type</th>
                <th className="text-right py-2 px-1">Z-Score</th>
                <th className="text-right py-2 px-1">Halflife</th>
                <th className="text-right py-2 px-1">Hurst</th>
                <th className="text-center py-2 px-1">Coint</th>
                <th className="text-right py-2 px-1">E[R] bps</th>
                <th className="text-right py-2 px-1">Liq.</th>
                <th className="text-center py-2 px-1">Signal</th>
              </tr>
            </thead>
            <tbody>
              {active.map((s) => (
                <tr key={s.spread_id} className="border-b border-neutral-800/50 hover:bg-neutral-800/30 transition-colors">
                  <td className="py-2 px-2 font-mono text-neutral-200">
                    {s.leg1}<span className="text-neutral-600"> / </span>{s.leg2}
                  </td>
                  <td className="py-2 px-1 text-neutral-400">{s.spread_type}</td>
                  <td className={`py-2 px-1 text-right font-mono ${Math.abs(s.zscore) > 2.5 ? "text-amber-400" : "text-neutral-300"}`}>
                    {s.zscore.toFixed(2)}
                  </td>
                  <td className="py-2 px-1 text-right font-mono text-neutral-300">{s.halflife_days.toFixed(0)}d</td>
                  <td className={`py-2 px-1 text-right font-mono ${s.hurst_exponent < 0.45 ? "text-emerald-400" : s.hurst_exponent > 0.55 ? "text-red-400" : "text-neutral-300"}`}>
                    {s.hurst_exponent.toFixed(2)}
                  </td>
                  <td className="py-2 px-1 text-center">
                    <span className={`text-[10px] ${s.is_cointegrated ? "text-emerald-400" : "text-neutral-600"}`}>
                      {s.is_cointegrated ? "✓" : "✗"}
                    </span>
                  </td>
                  <td className={`py-2 px-1 text-right font-mono ${s.expected_return_bps > 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {s.expected_return_bps.toFixed(1)}
                  </td>
                  <td className="py-2 px-1 text-right font-mono text-neutral-300">{(s.liquidity_score * 100).toFixed(0)}%</td>
                  <td className="py-2 px-1 text-center">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${SIGNAL_STYLE[s.signal] || ""}`}>
                      {s.signal.toUpperCase().replace("_", " ")}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {rejected.length > 0 && (
        <details className="mt-4">
          <summary className="text-xs text-neutral-500 cursor-pointer hover:text-neutral-400 transition-colors">
            {rejected.length} rejected spreads
          </summary>
          <div className="mt-2 space-y-1">
            {rejected.map((s) => (
              <div key={s.spread_id} className="flex justify-between text-[10px] text-neutral-600 font-mono">
                <span>{s.leg1}/{s.leg2}</span>
                <span>{s.rejection_reason || "filtered"}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
