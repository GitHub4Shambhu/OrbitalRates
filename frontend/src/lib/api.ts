/**
 * OrbitalRates — API Client
 * 
 * TypeScript types and fetch functions for the backend API.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ── Types ───────────────────────────────────────────────────────

export interface RegimeData {
  regime: string;
  confidence: number;
  vol_percentile: number;
  correlation_level: number;
  liquidity_index: number;
  funding_stress: number;
  leverage_cap: number;
  halflife_tolerance: number;
  regime_duration_days: number;
  transition_probability: Record<string, number>;
}

export interface RiskMetrics {
  total_dv01: number;
  total_convexity: number;
  gross_leverage: number;
  net_leverage: number;
  expected_return_annual_pct: number;
  expected_shortfall_99_pct: number;
  var_99_pct: number;
  max_stress_loss_pct: number;
  survival_probability: number;
  sharpe_estimate: number;
  current_drawdown_pct: number;
  max_drawdown_pct: number;
  liquidity_weighted_exposure: number;
  correlation_risk: number;
  crowding_risk: number;
  funding_cost_bps: number;
}

export interface StressResult {
  scenario_name: string;
  portfolio_loss_pct: number;
  margin_call_triggered: boolean;
  survival: boolean;
  description: string;
}

export interface SpreadCandidate {
  spread_id: string;
  spread_type: string;
  leg1: string;
  leg2: string;
  current_spread_bps: number;
  zscore: number;
  halflife_days: number;
  ar1_coefficient: number;
  is_stationary: boolean;
  structural_break_prob: number;
  liquidity_score: number;
  crowding_proxy: number;
  expected_return_bps: number;
  expected_shortfall_bps: number;
  vol_adjusted_return: number;
  tail_5pct_bps: number;
  tail_1pct_bps: number;
  signal: string;
  rejection_reason?: string;
}

export interface Position {
  position_id: string;
  spread_id: string;
  spread_type: string;
  leg1: string;
  leg2: string;
  direction: string;
  notional: number;
  dv01_net: number;
  weight: number;
  entry_spread_bps: number;
  current_spread_bps: number;
  entry_zscore: number;
  current_zscore: number;
  unrealized_pnl_bps: number;
}

export interface ExecutionResult {
  spread_id: string;
  intended_notional: number;
  executed_notional: number;
  slippage_bps: number;
  market_impact_bps: number;
  fill_rate: number;
  execution_cost_bps: number;
  liquidity_available: boolean;
}

export interface DecayMetrics {
  halflife_drift_pct: number;
  edge_decay_pct: number;
  regime_frequency_change: number;
  correlation_clustering: number;
  crowding_increase: number;
  retraining_recommended: boolean;
  parameter_adjustments: Record<string, number>;
}

export interface AuditEntry {
  timestamp: string;
  agent: string;
  action: string;
  decision: string;
  rationale: string;
  approved: boolean;
}

export interface DashboardData {
  cycle_count: number;
  data_source: string;
  is_halted: boolean;
  cycle_duration_ms: number;
  timestamp: string;
  regime: RegimeData | null;
  opportunities: SpreadCandidate[];
  total_candidates: number;
  positions: Position[];
  total_positions: number;
  risk_metrics: RiskMetrics | null;
  stress_results: StressResult[];
  execution_results: ExecutionResult[];
  decay_metrics: DecayMetrics | null;
  rejections: string[];
  audit_log: AuditEntry[];
}

// ── API Functions ───────────────────────────────────────────────

export async function fetchDashboard(nav?: number): Promise<DashboardData> {
  const params = nav ? `?nav=${nav}` : "";
  const res = await fetch(`${API_BASE}/orbital/dashboard${params}`);
  if (!res.ok) throw new Error(`Dashboard fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchCurrentState(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE}/orbital/state`);
  if (!res.ok) throw new Error(`State fetch failed: ${res.status}`);
  return res.json();
}

export async function activateKillSwitch(reason: string): Promise<void> {
  await fetch(`${API_BASE}/orbital/kill-switch/activate?reason=${encodeURIComponent(reason)}`, {
    method: "POST",
  });
}

export async function deactivateKillSwitch(reason: string): Promise<void> {
  await fetch(`${API_BASE}/orbital/kill-switch/deactivate?reason=${encodeURIComponent(reason)}`, {
    method: "POST",
  });
}

export async function fetchAuditLog(): Promise<AuditEntry[]> {
  const res = await fetch(`${API_BASE}/orbital/audit`);
  if (!res.ok) throw new Error(`Audit fetch failed: ${res.status}`);
  const data = await res.json();
  return data.entries;
}
