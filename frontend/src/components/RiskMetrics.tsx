"use client";

import { RiskMetrics as RiskMetricsType } from "@/lib/api";

function Gauge({ label, value, unit, color, warn }: {
  label: string; value: number; unit: string; color: string; warn?: boolean;
}) {
  return (
    <div className={`p-3 rounded-lg bg-neutral-900/50 border ${warn ? "border-red-500/40" : "border-neutral-800"}`}>
      <p className="text-[10px] text-neutral-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-xl font-bold font-mono ${color}`}>
        {typeof value === "number" ? value.toFixed(2) : "—"}{" "}
        <span className="text-xs text-neutral-500">{unit}</span>
      </p>
    </div>
  );
}

export default function RiskMetrics({ data }: { data: RiskMetricsType | null }) {
  if (!data) {
    return (
      <div className="card p-6 text-center text-neutral-500">
        No risk metrics — awaiting cycle
      </div>
    );
  }

  const survivalColor =
    data.survival_probability >= 0.99 ? "text-emerald-400" :
    data.survival_probability >= 0.95 ? "text-amber-400" : "text-red-400";

  const ddColor = data.current_drawdown_pct > 5 ? "text-red-400" : "text-neutral-200";
  const leverageColor = data.gross_leverage > 8 ? "text-red-400" : data.gross_leverage > 5 ? "text-amber-400" : "text-emerald-400";

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Risk & Survival Engine
        </h2>
        <span className="text-xs px-2 py-0.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-400">
          Layer 4
        </span>
      </div>

      {/* Survival Probability — hero metric */}
      <div className="text-center mb-6 p-4 rounded-xl bg-neutral-900/60 border border-neutral-800">
        <p className="text-[10px] uppercase text-neutral-500 tracking-widest mb-1">Survival Probability</p>
        <p className={`text-5xl font-bold font-mono ${survivalColor}`}>
          {(data.survival_probability * 100).toFixed(2)}%
        </p>
        <p className="text-xs text-neutral-500 mt-1">Target &ge; 99.00%</p>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <Gauge label="Sharpe (est)" value={data.sharpe_estimate} unit="" color="text-blue-400" />
        <Gauge label="Gross Leverage" value={data.gross_leverage} unit="×" color={leverageColor} warn={data.gross_leverage > 8} />
        <Gauge label="Net Leverage" value={data.net_leverage} unit="×" color="text-neutral-200" />
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <Gauge label="VaR 99%" value={data.var_99_pct} unit="%" color="text-amber-400" />
        <Gauge label="ES 99%" value={data.expected_shortfall_99_pct} unit="%" color="text-orange-400" />
        <Gauge label="Max Stress Loss" value={data.max_stress_loss_pct} unit="%" color="text-red-400" warn={data.max_stress_loss_pct > 3} />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Gauge label="Drawdown" value={data.current_drawdown_pct} unit="%" color={ddColor} warn={data.current_drawdown_pct > 5} />
        <Gauge label="Crowding Risk" value={data.crowding_risk} unit="" color="text-neutral-200" />
        <Gauge label="Correlation Risk" value={data.correlation_risk} unit="" color="text-neutral-200" />
      </div>

      {/* EVT / Advanced Tail Risk */}
      <div className="mt-4 pt-4 border-t border-neutral-800">
        <p className="text-[10px] text-neutral-500 uppercase tracking-wider mb-3">Extreme Value Theory (GPD)</p>
        <div className="grid grid-cols-3 gap-3">
          <Gauge label="EVT VaR 99%" value={data.evt_var_99_pct} unit="%" color="text-orange-400" />
          <Gauge label="EVT ES 99%" value={data.evt_es_99_pct} unit="%" color="text-red-400" warn={data.evt_es_99_pct > 5} />
          <Gauge label="GPD Shape (ξ)" value={data.evt_shape_parameter} unit="" color={data.evt_shape_parameter > 0.3 ? "text-red-400" : "text-blue-400"} />
        </div>
        <div className="grid grid-cols-2 gap-3 mt-3">
          <Gauge label="Tail Dependence (λ)" value={data.tail_dependence_coeff} unit="" color={data.tail_dependence_coeff > 0.5 ? "text-red-400" : "text-neutral-300"} />
          <Gauge label="Regime Vol ×" value={data.regime_vol_multiplier} unit="×" color={data.regime_vol_multiplier > 2 ? "text-amber-400" : "text-neutral-300"} />
        </div>
      </div>
    </div>
  );
}
