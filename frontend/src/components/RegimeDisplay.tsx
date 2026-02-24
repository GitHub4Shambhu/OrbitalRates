"use client";

import { RegimeData } from "@/lib/api";

const REGIME_CONFIG: Record<string, { color: string; bg: string; icon: string; label: string }> = {
  stable_mean_reverting:   { color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/30", icon: "◈", label: "STABLE" },
  volatile_mean_reverting: { color: "text-amber-400",   bg: "bg-amber-500/10 border-amber-500/30",     icon: "◇", label: "VOLATILE" },
  liquidity_tightening:    { color: "text-orange-400",  bg: "bg-orange-500/10 border-orange-500/30",   icon: "⬡", label: "LIQUIDITY" },
  structural_shift:        { color: "text-red-400",     bg: "bg-red-500/10 border-red-500/30",         icon: "⬢", label: "STRUCTURAL" },
  crisis:                  { color: "text-red-500",     bg: "bg-red-500/20 border-red-500/50 crisis-pulse", icon: "⚠", label: "CRISIS" },
};

function MeterBar({ label, value, max = 1, color }: { label: string; value: number; max?: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-neutral-400">{label}</span>
        <span className="text-neutral-300 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RegimeDisplay({ data }: { data: RegimeData | null }) {
  if (!data) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No regime data — awaiting first cycle
      </div>
    );
  }

  const cfg = REGIME_CONFIG[data.regime] || REGIME_CONFIG.stable_mean_reverting;

  return (
    <div className={`card p-6 border ${cfg.bg}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Market Regime
        </h2>
        <span className={`text-xs px-2 py-0.5 rounded-full border ${cfg.bg} ${cfg.color}`}>
          Layer 3
        </span>
      </div>

      <div className="flex items-center gap-3 mb-6">
        <span className={`text-4xl ${cfg.color}`}>{cfg.icon}</span>
        <div>
          <p className={`text-2xl font-bold tracking-wide ${cfg.color}`}>{cfg.label}</p>
          <p className="text-xs text-neutral-500">
            Confidence: {(data.confidence * 100).toFixed(0)}% · Duration: {data.regime_duration_days}d
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        <MeterBar label="Volatility %ile" value={data.vol_percentile} color="bg-amber-500" />
        <MeterBar label="Correlation" value={data.correlation_level} color="bg-blue-500" />
        <MeterBar label="Liquidity" value={data.liquidity_index} color="bg-emerald-500" />
        <MeterBar label="Funding Stress" value={data.funding_stress} color="bg-red-500" />
      </div>

      <div className="mt-4 pt-4 border-t border-neutral-800 grid grid-cols-2 gap-4 text-xs">
        <div>
          <span className="text-neutral-500">Leverage Cap</span>
          <p className="text-neutral-200 font-mono">{data.leverage_cap.toFixed(1)}×</p>
        </div>
        <div>
          <span className="text-neutral-500">Halflife Tolerance</span>
          <p className="text-neutral-200 font-mono">{data.halflife_tolerance.toFixed(1)}×</p>
        </div>
      </div>
    </div>
  );
}
