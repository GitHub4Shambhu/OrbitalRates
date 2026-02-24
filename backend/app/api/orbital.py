"""
REST API — OrbitalRates Dashboard Endpoints

Provides full system state for the institutional dashboard.
"""

from fastapi import APIRouter, Query
from typing import Optional
from loguru import logger

from app.orchestrator.orchestrator import get_orchestrator, SystemState
from app.governance.governance import get_governance
from app.schemas.responses import (
    DashboardResponse, RegimeResponse, RiskMetricsResponse,
    StressResultResponse, SpreadCandidateResponse, PositionResponse,
    ExecutionResponse, DecayResponse, AuditEntryResponse,
)

router = APIRouter(prefix="/orbital", tags=["OrbitalRates"])


# ── Conversion Helpers ───────────────────────────────────────────

def _state_to_dashboard(state: SystemState, cycle_count: int) -> DashboardResponse:
    """Convert SystemState to DashboardResponse."""
    # Regime
    regime_resp = None
    if state.regime:
        regime_resp = RegimeResponse(
            regime=state.regime.regime.value,
            confidence=state.regime.confidence,
            vol_percentile=state.regime.vol_percentile,
            correlation_level=state.regime.correlation_level,
            liquidity_index=state.regime.liquidity_index,
            funding_stress=state.regime.funding_stress,
            leverage_cap=state.regime.leverage_cap,
            halflife_tolerance=state.regime.halflife_tolerance,
            regime_duration_days=state.regime.regime_duration_days,
            transition_probability=state.regime.transition_probability,
            hmm_state_probabilities=state.regime.hmm_state_probabilities,
            hmm_confidence=state.regime.hmm_confidence,
            heuristic_regime=state.regime.heuristic_regime,
            ensemble_agreement=state.regime.ensemble_agreement,
        )

    # Risk
    risk_resp = None
    if state.risk_metrics:
        r = state.risk_metrics
        risk_resp = RiskMetricsResponse(
            total_dv01=r.total_dv01,
            total_convexity=r.total_convexity,
            gross_leverage=r.gross_leverage,
            net_leverage=r.net_leverage,
            expected_return_annual_pct=r.expected_return_annual_pct,
            expected_shortfall_99_pct=r.expected_shortfall_99_pct,
            var_99_pct=r.var_99_pct,
            max_stress_loss_pct=r.max_stress_loss_pct,
            survival_probability=r.survival_probability,
            sharpe_estimate=r.sharpe_estimate,
            current_drawdown_pct=r.current_drawdown_pct,
            max_drawdown_pct=r.max_drawdown_pct,
            liquidity_weighted_exposure=r.liquidity_weighted_exposure,
            correlation_risk=r.correlation_risk,
            crowding_risk=r.crowding_risk,
            funding_cost_bps=r.funding_cost_bps,
            evt_var_99_pct=r.evt_var_99_pct,
            evt_es_99_pct=r.evt_es_99_pct,
            evt_shape_parameter=r.evt_shape_parameter,
            tail_dependence_coeff=r.tail_dependence_coeff,
            regime_vol_multiplier=r.regime_vol_multiplier,
        )

    # Stress
    stress = [
        StressResultResponse(
            scenario_name=s.scenario_name,
            portfolio_loss_pct=s.portfolio_loss_pct,
            margin_call_triggered=s.margin_call_triggered,
            survival=s.survival,
            description=s.description,
        )
        for s in state.stress_results
    ]

    # Opportunities
    opps = [
        SpreadCandidateResponse(
            spread_id=c.spread_id,
            spread_type=c.spread_type.value,
            leg1=c.leg1_instrument,
            leg2=c.leg2_instrument,
            current_spread_bps=c.current_spread_bps,
            zscore=c.zscore,
            halflife_days=c.halflife_days,
            ar1_coefficient=c.ar1_coefficient,
            is_stationary=c.is_stationary,
            structural_break_prob=c.structural_break_prob,
            liquidity_score=c.liquidity_score,
            crowding_proxy=c.crowding_proxy,
            expected_return_bps=c.expected_return_bps,
            expected_shortfall_bps=c.expected_shortfall_bps,
            vol_adjusted_return=c.vol_adjusted_return,
            tail_5pct_bps=c.tail_5pct_bps,
            tail_1pct_bps=c.tail_1pct_bps,
            signal=c.signal.value,
            rejection_reason=c.rejection_reason,
            hurst_exponent=c.hurst_exponent,
            johansen_trace_stat=c.johansen_trace_stat,
            is_cointegrated=c.is_cointegrated,
            kalman_hedge_ratio=c.kalman_hedge_ratio,
            kalman_hedge_ratio_std=c.kalman_hedge_ratio_std,
        )
        for c in state.candidates[:30]  # Top 30
    ]

    # Positions
    positions = [
        PositionResponse(
            position_id=p.position_id,
            spread_id=p.spread_id,
            spread_type=p.spread_type.value,
            leg1=p.leg1_instrument,
            leg2=p.leg2_instrument,
            direction=p.direction,
            notional=p.notional,
            dv01_net=p.dv01_net,
            weight=p.weight,
            entry_spread_bps=p.entry_spread_bps,
            current_spread_bps=p.current_spread_bps,
            entry_zscore=p.entry_zscore,
            current_zscore=p.current_zscore,
            unrealized_pnl_bps=p.unrealized_pnl_bps,
        )
        for p in state.approved_positions
    ]

    # Execution
    executions = [
        ExecutionResponse(
            spread_id=e.spread_id,
            intended_notional=e.intended_notional,
            executed_notional=e.executed_notional,
            slippage_bps=e.slippage_bps,
            market_impact_bps=e.market_impact_bps,
            fill_rate=e.fill_rate,
            execution_cost_bps=e.execution_cost_bps,
            liquidity_available=e.liquidity_available,
        )
        for e in state.execution_results
    ]

    # Decay
    decay_resp = None
    if state.decay_metrics:
        d = state.decay_metrics
        decay_resp = DecayResponse(
            halflife_drift_pct=d.halflife_drift_pct,
            edge_decay_pct=d.edge_decay_pct,
            regime_frequency_change=d.regime_frequency_change,
            correlation_clustering=d.correlation_clustering,
            crowding_increase=d.crowding_increase,
            retraining_recommended=d.retraining_recommended,
            parameter_adjustments=d.parameter_adjustments,
            cusum_statistic=d.cusum_statistic,
            cusum_alert=d.cusum_alert,
            ewma_edge_estimate=d.ewma_edge_estimate,
            forgetting_factor=d.forgetting_factor,
            bayesian_zscore_threshold=d.bayesian_zscore_threshold,
        )

    # Audit
    audit = [
        AuditEntryResponse(**entry) for entry in state.audit_log
    ]

    return DashboardResponse(
        cycle_count=cycle_count,
        data_source=state.data_source,
        is_halted=state.is_halted,
        cycle_duration_ms=state.cycle_duration_ms,
        timestamp=state.timestamp.isoformat(),
        regime=regime_resp,
        opportunities=opps,
        total_candidates=len(state.candidates),
        positions=positions,
        total_positions=len(state.approved_positions),
        risk_metrics=risk_resp,
        stress_results=stress,
        execution_results=executions,
        decay_metrics=decay_resp,
        rejections=state.rejections,
        audit_log=audit,
    )


# ── Endpoints ────────────────────────────────────────────────────

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    nav: float = Query(100_000_000, description="NAV in dollars"),
):
    """
    Run a full agent cycle and return complete system state.
    
    This is the main endpoint — triggers all 7 agent layers.
    """
    orchestrator = get_orchestrator()
    state = await orchestrator.run_cycle(nav=nav)
    return _state_to_dashboard(state, orchestrator.cycle_count)


@router.get("/state", response_model=DashboardResponse)
async def get_current_state():
    """
    Get the last computed system state without running a new cycle.
    """
    orchestrator = get_orchestrator()
    state = orchestrator.current_state
    if state is None:
        return DashboardResponse()
    return _state_to_dashboard(state, orchestrator.cycle_count)


@router.post("/kill-switch/activate")
async def activate_kill_switch(reason: str = Query(..., description="Reason for halt")):
    """Activate emergency kill switch — halts all trading."""
    governance = get_governance()
    governance.activate_kill_switch(reason)
    return {"status": "HALTED", "reason": reason}


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch(reason: str = Query(..., description="Reason for resumption")):
    """Deactivate kill switch — resume trading."""
    governance = get_governance()
    governance.deactivate_kill_switch(reason)
    return {"status": "ACTIVE", "reason": reason}


@router.get("/audit")
async def get_audit_log():
    """Get recent governance audit log."""
    governance = get_governance()
    return {"entries": governance.recent_audit}


@router.get("/health")
async def health():
    """System health check."""
    orchestrator = get_orchestrator()
    governance = get_governance()
    return {
        "status": "halted" if governance.is_halted else "operational",
        "cycles_completed": orchestrator.cycle_count,
        "version": "0.1.0",
    }
