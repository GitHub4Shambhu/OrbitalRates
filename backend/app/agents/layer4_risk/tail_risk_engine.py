"""
Layer 4 — Tail Risk & Survival Simulation Engine

THE CORE AGENT. Non-negotiable constraints:
  - Max drawdown limit: configurable
  - Max expected shortfall: defined
  - Leverage cap: dynamic

Simulates:
  - 2008-style spread blowout
  - 2020 liquidity seizure
  - Correlation → 1 shock
  - Volatility doubling
  - Funding spread spike
  - Forced deleveraging cascade

If survival probability < 99% annual → REDUCE LEVERAGE. ALWAYS.
"""

import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import (
    RiskMetrics, StressResult, PortfolioPosition, RegimeState,
    SpreadCandidate, MarketRegime,
)


class TailRiskEngine:
    """
    Layer 4: Survival-First Risk Core.

    Priority: Survival over return. Always.
    """

    def __init__(self):
        self._risk_history: List[RiskMetrics] = []
        logger.info("Layer 4 — Tail Risk & Survival Engine initialized")

    # ── Public API ───────────────────────────────────────────

    async def compute_risk_metrics(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
        candidates: Optional[List[SpreadCandidate]] = None,
    ) -> RiskMetrics:
        """
        Compute comprehensive portfolio risk metrics.
        """
        # Portfolio aggregates
        total_dv01 = sum(p.dv01_net for p in positions)
        total_convexity = sum(p.convexity_net for p in positions)
        gross_notional = sum(abs(p.notional) for p in positions)
        net_notional = sum(p.notional for p in positions)
        
        # Leverage
        nav = max(gross_notional / settings.max_leverage, 1e6)  # Implied NAV
        gross_leverage = gross_notional / nav if nav > 0 else 0
        net_leverage = abs(net_notional) / nav if nav > 0 else 0

        # Unrealized P&L
        total_pnl = sum(p.unrealized_pnl_bps for p in positions)

        # Expected return (sum of position expected returns)
        expected_return = 0.0
        if candidates:
            for c in candidates:
                for p in positions:
                    if p.spread_id == c.spread_id:
                        expected_return += c.expected_return_bps * p.weight

        # Monte Carlo simulation
        mc_losses = self._monte_carlo_simulation(positions, regime, n_sims=10000)
        
        # VaR & ES
        var_99 = np.percentile(mc_losses, 1) if len(mc_losses) > 0 else 0
        es_99 = mc_losses[mc_losses <= np.percentile(mc_losses, 1)].mean() if len(mc_losses) > 0 else 0

        # Stress tests
        stress_results = self._run_stress_tests(positions, regime)
        max_stress = max((s.portfolio_loss_pct for s in stress_results), default=0)

        # Survival probability
        survival = self._compute_survival_probability(mc_losses, nav)

        # Drawdown
        current_dd, max_dd = self._compute_drawdown(positions)

        # Liquidity-weighted exposure
        liq_exposure = sum(
            abs(p.notional) * (1 - p.leverage_contribution) 
            for p in positions
        ) / (nav + 1e-10)

        # Correlation risk (average across positions)
        corr_risk = 0.5  # Placeholder — computed from regime

        # Crowding risk
        crowding = 0.0
        if candidates:
            relevant = [c for c in candidates if any(p.spread_id == c.spread_id for p in positions)]
            if relevant:
                crowding = np.mean([c.crowding_proxy for c in relevant])

        # Sharpe estimate
        vol_annual = np.std(mc_losses) * np.sqrt(252) if len(mc_losses) > 0 else 1.0
        sharpe = (expected_return * 252 / 10000) / (vol_annual + 1e-10)

        metrics = RiskMetrics(
            total_dv01=round(total_dv01, 4),
            total_convexity=round(total_convexity, 4),
            gross_leverage=round(gross_leverage, 2),
            net_leverage=round(net_leverage, 2),
            expected_return_annual_pct=round(expected_return * 252 / 10000, 4),
            expected_shortfall_99_pct=round(abs(es_99) * 100, 4),
            var_99_pct=round(abs(var_99) * 100, 4),
            max_stress_loss_pct=round(abs(max_stress), 4),
            survival_probability=round(survival, 6),
            sharpe_estimate=round(sharpe, 4),
            current_drawdown_pct=round(current_dd, 4),
            max_drawdown_pct=round(max_dd, 4),
            liquidity_weighted_exposure=round(liq_exposure, 4),
            correlation_risk=round(corr_risk, 4),
            crowding_risk=round(crowding, 4),
            funding_cost_bps=round(regime.funding_stress * 50, 2),
        )

        self._risk_history.append(metrics)
        return metrics

    async def validate_trade(
        self,
        candidate: SpreadCandidate,
        current_positions: List[PortfolioPosition],
        regime: RegimeState,
    ) -> tuple[bool, str]:
        """
        Validate whether a new trade passes survival constraints.
        
        Returns (approved, reason).
        """
        reasons = []

        # Check leverage
        current_leverage = sum(abs(p.notional) for p in current_positions)
        if current_leverage > regime.leverage_cap * 1e6:
            reasons.append(f"leverage {current_leverage/1e6:.1f}x exceeds regime cap {regime.leverage_cap}x")

        # Check half-life vs regime tolerance
        if candidate.halflife_days > regime.halflife_tolerance:
            reasons.append(
                f"halflife {candidate.halflife_days:.0f}d > regime tolerance "
                f"{regime.halflife_tolerance}d ({regime.regime.value})"
            )

        # Check expected shortfall
        if candidate.expected_shortfall_bps > settings.max_single_event_loss_pct * 100:
            reasons.append(
                f"ES {candidate.expected_shortfall_bps:.0f}bps > "
                f"{settings.max_single_event_loss_pct}% NAV cap"
            )

        # Check liquidity in current regime
        min_liq = settings.min_liquidity_score
        if regime.regime in [MarketRegime.LIQUIDITY_TIGHTENING, MarketRegime.CRISIS]:
            min_liq = min_liq * 1.5  # Higher bar in stress
        if candidate.liquidity_score < min_liq:
            reasons.append(f"liquidity {candidate.liquidity_score:.2f} < min {min_liq:.2f}")

        # Crisis mode: reject all new trades
        if regime.regime == MarketRegime.CRISIS:
            reasons.append("CRISIS regime — no new positions")

        if reasons:
            return False, "; ".join(reasons)
        return True, "Approved"

    def run_stress_tests(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
    ) -> List[StressResult]:
        """Run all stress scenarios (public wrapper)."""
        return self._run_stress_tests(positions, regime)

    # ── Monte Carlo ──────────────────────────────────────────

    def _monte_carlo_simulation(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
        n_sims: int = 10000,
    ) -> np.ndarray:
        """
        Fat-tail Monte Carlo simulation of portfolio returns.
        Uses Student-t distribution (df=4) for realistic tails.
        """
        if not positions:
            return np.zeros(n_sims)

        n_positions = len(positions)
        
        # Regime-adjusted volatility
        vol_multiplier = 1.0
        if regime.regime == MarketRegime.VOLATILE_MEAN_REVERTING:
            vol_multiplier = 1.5
        elif regime.regime == MarketRegime.LIQUIDITY_TIGHTENING:
            vol_multiplier = 2.0
        elif regime.regime == MarketRegime.STRUCTURAL_SHIFT:
            vol_multiplier = 2.5
        elif regime.regime == MarketRegime.CRISIS:
            vol_multiplier = 3.0

        # Simulate daily returns (fat-tailed)
        losses = np.zeros(n_sims)
        for pos in positions:
            # Base daily vol from spread std
            daily_vol = 0.01 * vol_multiplier  # ~1% daily vol * regime multiplier
            
            # Student-t with df=4 for fat tails
            t_draws = np.random.standard_t(df=4, size=n_sims)
            t_draws = t_draws / np.std(t_draws)  # Normalize
            
            pos_returns = t_draws * daily_vol * pos.weight
            losses += pos_returns

        return losses

    # ── Stress Tests ─────────────────────────────────────────

    def _run_stress_tests(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
    ) -> List[StressResult]:
        """Run institutional stress scenarios."""
        scenarios = [
            self._stress_2008_spread_blowout(positions),
            self._stress_2020_liquidity_seizure(positions),
            self._stress_correlation_one(positions),
            self._stress_vol_doubling(positions, regime),
            self._stress_funding_spike(positions),
            self._stress_forced_deleveraging(positions),
        ]
        return scenarios

    def _stress_2008_spread_blowout(self, positions: List[PortfolioPosition]) -> StressResult:
        """2008-style: spreads widen 200+ bps across the board."""
        loss = sum(
            abs(p.weight) * settings.stress_spread_widening_bps / 10000
            for p in positions
        )
        return StressResult(
            scenario_name="2008 Spread Blowout",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=loss > settings.max_single_event_loss_pct / 100,
            survival=loss < settings.max_drawdown_pct / 100,
            description=f"All spreads widen {settings.stress_spread_widening_bps}bps",
            spread_widening_bps=settings.stress_spread_widening_bps,
        )

    def _stress_2020_liquidity_seizure(self, positions: List[PortfolioPosition]) -> StressResult:
        """2020-style: liquidity evaporates, forced selling at haircut."""
        haircut = settings.stress_liquidity_haircut
        loss = sum(abs(p.weight) * haircut for p in positions)
        return StressResult(
            scenario_name="2020 Liquidity Seizure",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=loss > settings.max_single_event_loss_pct / 100,
            survival=loss < settings.max_drawdown_pct / 100,
            description=f"Forced liquidation at {haircut*100}% haircut",
        )

    def _stress_correlation_one(self, positions: List[PortfolioPosition]) -> StressResult:
        """All correlations go to ~1: hedges fail."""
        loss = sum(abs(p.weight) * 0.05 for p in positions)  # 5% loss per position
        return StressResult(
            scenario_name="Correlation → 1",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=loss > settings.max_single_event_loss_pct / 100,
            survival=loss < settings.max_drawdown_pct / 100,
            description="All hedges fail, correlations spike to 0.95",
            correlation_shock=settings.stress_correlation_shock,
        )

    def _stress_vol_doubling(self, positions: List[PortfolioPosition], regime: RegimeState) -> StressResult:
        """Volatility doubles from current level."""
        loss = sum(abs(p.weight) * 0.03 * settings.stress_vol_multiplier for p in positions)
        return StressResult(
            scenario_name="Volatility Doubling",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=loss > settings.max_single_event_loss_pct / 100,
            survival=loss < settings.max_drawdown_pct / 100,
            description=f"Vol multiplied by {settings.stress_vol_multiplier}x",
            vol_shock_multiplier=settings.stress_vol_multiplier,
        )

    def _stress_funding_spike(self, positions: List[PortfolioPosition]) -> StressResult:
        """Funding costs spike (repo rate shock)."""
        funding_cost = settings.stress_funding_spread_bps / 10000
        loss = sum(abs(p.weight) * funding_cost for p in positions)
        return StressResult(
            scenario_name="Funding Spike",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=loss > settings.max_single_event_loss_pct / 100,
            survival=loss < settings.max_drawdown_pct / 100,
            description=f"Funding spread spikes {settings.stress_funding_spread_bps}bps",
        )

    def _stress_forced_deleveraging(self, positions: List[PortfolioPosition]) -> StressResult:
        """Forced deleveraging cascade — must sell 50% at market."""
        loss = sum(abs(p.weight) * 0.02 for p in positions) * 0.5  # Sell half at 2% discount
        return StressResult(
            scenario_name="Forced Deleveraging",
            portfolio_loss_pct=round(loss * 100, 4),
            margin_call_triggered=False,
            survival=True,
            description="Forced to sell 50% of positions at market",
        )

    # ── Survival Computation ─────────────────────────────────

    def _compute_survival_probability(self, mc_losses: np.ndarray, nav: float) -> float:
        """
        Estimate annual survival probability.
        
        Survival = probability that no single day loss exceeds
        fatal threshold over ~252 trading days.
        """
        if len(mc_losses) == 0:
            return 1.0

        # Fatal threshold: max drawdown limit
        fatal_threshold = settings.max_drawdown_pct / 100

        # Daily probability of fatal loss
        daily_p_fatal = (mc_losses < -fatal_threshold).mean()

        # Annual survival = (1 - daily_p_fatal)^252
        annual_survival = (1 - daily_p_fatal) ** 252

        return max(0.0, min(1.0, annual_survival))

    def _compute_drawdown(self, positions: List[PortfolioPosition]) -> tuple[float, float]:
        """Compute current and max drawdown from position history."""
        if not positions:
            return 0.0, 0.0

        total_pnl = sum(p.unrealized_pnl_bps for p in positions) / 10000
        current_dd = max(0, -total_pnl)

        # Max drawdown from risk history
        max_dd = current_dd
        for metrics in self._risk_history:
            max_dd = max(max_dd, metrics.max_drawdown_pct)

        return current_dd, max_dd


# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[TailRiskEngine] = None


def get_tail_risk_engine() -> TailRiskEngine:
    global _engine
    if _engine is None:
        _engine = TailRiskEngine()
    return _engine
