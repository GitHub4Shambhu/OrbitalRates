"use client";

import { Position } from "@/lib/api";

export default function Positions({ data }: { data: Position[] }) {
  if (data.length === 0) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No active positions
      </div>
    );
  }

  const totalPnL = data.reduce((sum, p) => sum + p.unrealized_pnl_bps, 0);

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Active Positions
        </h2>
        <div className="flex items-center gap-3">
          <span className={`text-xs font-mono font-bold ${totalPnL >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            P&L: {totalPnL >= 0 ? "+" : ""}{totalPnL.toFixed(1)} bps
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400">
            Layer 5
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-neutral-500 border-b border-neutral-800">
              <th className="text-left py-2 px-2">Spread</th>
              <th className="text-left py-2 px-1">Type</th>
              <th className="text-center py-2 px-1">Dir.</th>
              <th className="text-right py-2 px-1">Notional</th>
              <th className="text-right py-2 px-1">DV01</th>
              <th className="text-right py-2 px-1">Weight</th>
              <th className="text-right py-2 px-1">Entry Z</th>
              <th className="text-right py-2 px-1">Curr Z</th>
              <th className="text-right py-2 px-1">P&L bps</th>
            </tr>
          </thead>
          <tbody>
            {data.map((p) => (
              <tr key={p.position_id} className="border-b border-neutral-800/50 hover:bg-neutral-800/30 transition-colors">
                <td className="py-2 px-2 font-mono text-neutral-200">
                  {p.leg1}<span className="text-neutral-600"> / </span>{p.leg2}
                </td>
                <td className="py-2 px-1 text-neutral-400">{p.spread_type}</td>
                <td className={`py-2 px-1 text-center font-bold ${
                  p.direction === "LONG" ? "text-emerald-400" : "text-red-400"
                }`}>
                  {p.direction === "LONG" ? "▲" : "▼"}
                </td>
                <td className="py-2 px-1 text-right font-mono text-neutral-300">
                  ${(p.notional / 1e6).toFixed(1)}M
                </td>
                <td className="py-2 px-1 text-right font-mono text-neutral-300">
                  ${p.dv01_net.toFixed(0)}
                </td>
                <td className="py-2 px-1 text-right font-mono text-neutral-300">
                  {(p.weight * 100).toFixed(1)}%
                </td>
                <td className="py-2 px-1 text-right font-mono text-neutral-400">
                  {p.entry_zscore.toFixed(2)}
                </td>
                <td className={`py-2 px-1 text-right font-mono ${
                  Math.abs(p.current_zscore) > 2 ? "text-amber-400" : "text-neutral-300"
                }`}>
                  {p.current_zscore.toFixed(2)}
                </td>
                <td className={`py-2 px-1 text-right font-mono font-bold ${
                  p.unrealized_pnl_bps >= 0 ? "text-emerald-400" : "text-red-400"
                }`}>
                  {p.unrealized_pnl_bps >= 0 ? "+" : ""}{p.unrealized_pnl_bps.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
