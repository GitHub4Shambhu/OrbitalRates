"use client";

import { useState, useEffect, useCallback } from "react";
import { DashboardData, fetchDashboard } from "@/lib/api";
import { MOCK_DASHBOARD } from "@/lib/mockData";
import RegimeDisplay from "@/components/RegimeDisplay";
import RiskMetrics from "@/components/RiskMetrics";
import SpreadOpportunities from "@/components/SpreadOpportunities";
import StressTests from "@/components/StressTests";
import Positions from "@/components/Positions";
import ExecutionSim from "@/components/ExecutionSim";
import GovernancePanel from "@/components/GovernancePanel";
import DecayMonitor from "@/components/DecayMonitor";
import DataSourceBadge from "@/components/DataSourceBadge";

type Mode = "live" | "demo";

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [nav, setNav] = useState(1_000_000_000);
  const [mode, setMode] = useState<Mode>("live");

  const loadMockData = useCallback(() => {
    setData({ ...MOCK_DASHBOARD, timestamp: new Date().toISOString() });
    setLastRefresh(new Date());
    setError(null);
    setMode("demo");
  }, []);

  const runCycle = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchDashboard(nav);
      setData(result);
      setLastRefresh(new Date());
      setMode("live");
    } catch (e) {
      // If backend is unavailable, auto-fallback to demo mode
      console.warn("Backend unavailable, loading demo data:", e);
      loadMockData();
    } finally {
      setLoading(false);
    }
  }, [nav, loadMockData]);

  useEffect(() => {
    runCycle();
  }, [runCycle]);

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-neutral-100">
      {/* ── Demo Banner ────────────────────────────────── */}
      {mode === "demo" && (
        <div className="bg-amber-500/10 border-b border-amber-500/30 px-6 py-2 text-center">
          <span className="text-xs text-amber-400">
            🛰️ <strong>DEMO MODE</strong> — Showing simulated institutional data ·
            Cycle #847 · $1B NAV · 4 active positions ·
            <button onClick={runCycle} className="underline ml-1 hover:text-amber-300 transition-colors">
              Connect to live backend →
            </button>
          </span>
        </div>
      )}

      {/* ── Header ─────────────────────────────────────── */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[var(--bg-primary)]/80 border-b border-neutral-800">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="text-blue-400">Orbital</span>
                <span className="text-neutral-200">Rates</span>
              </h1>
              <p className="text-[10px] text-neutral-500 tracking-widest uppercase">
                Fixed Income Relative Value Engine
              </p>
            </div>
            {data && <DataSourceBadge source={data.data_source} />}
          </div>

          <div className="flex items-center gap-4">
            {/* Mode Toggle */}
            <div className="flex items-center gap-1 bg-neutral-800 rounded-lg p-0.5">
              <button
                onClick={loadMockData}
                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-colors ${
                  mode === "demo" ? "bg-amber-500/20 text-amber-400" : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                DEMO
              </button>
              <button
                onClick={runCycle}
                className={`px-3 py-1 rounded-md text-[10px] font-bold transition-colors ${
                  mode === "live" ? "bg-blue-500/20 text-blue-400" : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                LIVE
              </button>
            </div>

            {/* NAV Input */}
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-neutral-500 uppercase">NAV</label>
              <input
                type="number"
                value={nav}
                onChange={(e) => setNav(Number(e.target.value))}
                className="w-32 px-2 py-1 rounded bg-neutral-800 border border-neutral-700 text-xs font-mono text-neutral-200 focus:border-blue-500 focus:outline-none"
              />
            </div>

            {/* Cycle Info */}
            {data && (
              <div className="text-right text-[10px] text-neutral-500">
                <p>Cycle #{data.cycle_count}</p>
                <p>{data.cycle_duration_ms.toFixed(0)}ms</p>
              </div>
            )}

            {/* Refresh */}
            <button
              onClick={runCycle}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-bold text-white transition-colors disabled:opacity-50"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Running...
                </span>
              ) : (
                "Run Cycle"
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Content ────────────────────────────────── */}
      <main className="max-w-[1600px] mx-auto px-6 py-6">
        {/* Error State — only shown when both live + demo fail */}
        {error && !data && (
          <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading State */}
        {loading && !data && (
          <div className="flex flex-col items-center justify-center py-32 gap-4">
            <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            <p className="text-sm text-neutral-500">Running orbital analysis cycle...</p>
            <p className="text-xs text-neutral-600">Fetching market data → Discovery → Regime → Risk → Capital → Execution → Meta</p>
          </div>
        )}

        {/* Dashboard Grid */}
        {data && (
          <div className="space-y-6">
            {/* Row 1: Regime + Risk (key metrics) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <RegimeDisplay data={data.regime} />
              <RiskMetrics data={data.risk_metrics} />
            </div>

            {/* Row 2: Spread Opportunities (full width) */}
            <SpreadOpportunities data={data.opportunities} total={data.total_candidates} />

            {/* Row 3: Positions + Stress Tests */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Positions data={data.positions} />
              <StressTests data={data.stress_results} />
            </div>

            {/* Row 4: Execution + Decay + Governance */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <ExecutionSim data={data.execution_results} />
              <DecayMonitor data={data.decay_metrics} />
              <GovernancePanel
                isHalted={data.is_halted}
                auditLog={data.audit_log}
                onRefresh={runCycle}
              />
            </div>

            {/* Footer */}
            <div className="text-center text-[10px] text-neutral-600 py-4 border-t border-neutral-800/50">
              OrbitalRates v1.0 · 7-Layer Multi-Agent Architecture ·{" "}
              {lastRefresh ? `Last refresh: ${lastRefresh.toLocaleTimeString()}` : ""}
              {data.rejections.length > 0 && (
                <span className="ml-2 text-amber-500/50">{data.rejections.length} rejections this cycle</span>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
