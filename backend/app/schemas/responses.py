"""
Pydantic response models for the REST API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# ── Spread Candidate ─────────────────────────────────────────────

class SpreadCandidateResponse(BaseModel):
    spread_id: str
    spread_type: str
    leg1: str
    leg2: str
    current_spread_bps: float
    zscore: float
    halflife_days: float
    ar1_coefficient: float
    is_stationary: bool
    structural_break_prob: float
    liquidity_score: float
    crowding_proxy: float
    expected_return_bps: float
    expected_shortfall_bps: float
    vol_adjusted_return: float
    tail_5pct_bps: float
    tail_1pct_bps: float
    signal: str
    rejection_reason: Optional[str] = None


# ── Regime ───────────────────────────────────────────────────────

class RegimeResponse(BaseModel):
    regime: str
    confidence: float
    vol_percentile: float
    correlation_level: float
    liquidity_index: float
    funding_stress: float
    leverage_cap: float
    halflife_tolerance: int
    regime_duration_days: int
    transition_probability: Dict[str, float]


# ── Risk Metrics ─────────────────────────────────────────────────

class RiskMetricsResponse(BaseModel):
    total_dv01: float
    total_convexity: float
    gross_leverage: float
    net_leverage: float
    expected_return_annual_pct: float
    expected_shortfall_99_pct: float
    var_99_pct: float
    max_stress_loss_pct: float
    survival_probability: float
    sharpe_estimate: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    liquidity_weighted_exposure: float
    correlation_risk: float
    crowding_risk: float
    funding_cost_bps: float


class StressResultResponse(BaseModel):
    scenario_name: str
    portfolio_loss_pct: float
    margin_call_triggered: bool
    survival: bool
    description: str


# ── Portfolio Position ───────────────────────────────────────────

class PositionResponse(BaseModel):
    position_id: str
    spread_id: str
    spread_type: str
    leg1: str
    leg2: str
    direction: str
    notional: float
    dv01_net: float
    weight: float
    entry_spread_bps: float
    current_spread_bps: float
    entry_zscore: float
    current_zscore: float
    unrealized_pnl_bps: float


# ── Execution ────────────────────────────────────────────────────

class ExecutionResponse(BaseModel):
    spread_id: str
    intended_notional: float
    executed_notional: float
    slippage_bps: float
    market_impact_bps: float
    fill_rate: float
    execution_cost_bps: float
    liquidity_available: bool


# ── Decay / Meta Learning ───────────────────────────────────────

class DecayResponse(BaseModel):
    halflife_drift_pct: float
    edge_decay_pct: float
    regime_frequency_change: float
    correlation_clustering: float
    crowding_increase: float
    retraining_recommended: bool
    parameter_adjustments: Dict[str, float]


# ── Audit ────────────────────────────────────────────────────────

class AuditEntryResponse(BaseModel):
    timestamp: str
    agent: str
    action: str
    decision: str
    rationale: str
    approved: bool


# ── Full Dashboard ───────────────────────────────────────────────

class DashboardResponse(BaseModel):
    """Complete system state for the dashboard."""
    # System
    cycle_count: int = 0
    data_source: str = "live"
    is_halted: bool = False
    cycle_duration_ms: float = 0.0
    timestamp: str = ""

    # Regime
    regime: Optional[RegimeResponse] = None

    # Opportunities
    opportunities: List[SpreadCandidateResponse] = []
    total_candidates: int = 0

    # Portfolio
    positions: List[PositionResponse] = []
    total_positions: int = 0

    # Risk
    risk_metrics: Optional[RiskMetricsResponse] = None
    stress_results: List[StressResultResponse] = []

    # Execution
    execution_results: List[ExecutionResponse] = []

    # Decay
    decay_metrics: Optional[DecayResponse] = None

    # Governance
    rejections: List[str] = []
    audit_log: List[AuditEntryResponse] = []
