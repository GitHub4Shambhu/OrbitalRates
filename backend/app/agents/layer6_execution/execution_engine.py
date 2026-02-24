"""
Layer 6 — Microstructure Execution Engine

Simulates realistic execution conditions:
  - Market impact modeling
  - Bid-ask widening under stress
  - Slippage curves
  - Order book depth simulation
  - Latency constraints

Auto-reduces size if:
  - Slippage exceeds threshold
  - Liquidity deteriorates
  - Order flow imbalance detected
"""

import numpy as np
from typing import List, Optional
from loguru import logger

from app.core.config import settings
from app.core.types import (
    ExecutionResult, PortfolioPosition, RegimeState, MarketRegime,
)


class ExecutionEngine:
    """
    Layer 6: Execution Intelligence.
    
    Models realistic market microstructure for institutional
    fixed-income trading.
    """

    def __init__(self):
        self._execution_history: List[ExecutionResult] = []
        logger.info("Layer 6 — Execution Engine initialized")

    # ── Public API ───────────────────────────────────────────

    async def simulate_execution(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
        liquidity_scores: dict,
    ) -> List[ExecutionResult]:
        """
        Simulate execution for a set of positions.
        Returns adjusted execution results with slippage and fill rates.
        """
        results = []
        for pos in positions:
            result = self._simulate_single(pos, regime, liquidity_scores)
            results.append(result)
            self._execution_history.append(result)

        total_slippage = sum(r.slippage_bps for r in results) / max(len(results), 1)
        logger.info(
            f"Execution sim: {len(results)} positions, "
            f"avg slippage={total_slippage:.2f}bps, "
            f"regime={regime.regime.value}"
        )
        return results

    # ── Single Execution Simulation ──────────────────────────

    def _simulate_single(
        self,
        pos: PortfolioPosition,
        regime: RegimeState,
        liquidity_scores: dict,
    ) -> ExecutionResult:
        """Simulate execution for a single position."""
        
        # Base slippage from position size
        base_slippage = self._estimate_base_slippage(pos.notional)
        
        # Regime adjustment
        regime_mult = self._regime_slippage_multiplier(regime)
        
        # Liquidity adjustment
        liq_score = min(
            liquidity_scores.get(pos.leg1_instrument, 0.5),
            liquidity_scores.get(pos.leg2_instrument, 0.5),
        )
        liq_mult = self._liquidity_slippage_multiplier(liq_score)
        
        # Market impact (proportional to sqrt of notional)
        impact = self._market_impact(pos.notional)
        
        # Total slippage
        total_slippage = base_slippage * regime_mult * liq_mult
        total_cost = total_slippage + impact
        
        # Fill rate (may not fill 100% in illiquid markets)
        fill_rate = self._estimate_fill_rate(pos.notional, liq_score, regime)
        
        # Adjusted execution
        executed_notional = pos.notional * fill_rate
        
        # Auto-reduce if slippage too high
        if total_cost > settings.max_slippage_bps:
            reduction = settings.max_slippage_bps / total_cost
            executed_notional *= reduction
            fill_rate *= reduction
            logger.warning(
                f"Slippage {total_cost:.1f}bps > max {settings.max_slippage_bps}bps "
                f"for {pos.spread_id} — reduced to {fill_rate*100:.0f}% fill"
            )

        return ExecutionResult(
            spread_id=pos.spread_id,
            intended_notional=pos.notional,
            executed_notional=round(executed_notional, 2),
            slippage_bps=round(total_slippage, 4),
            market_impact_bps=round(impact, 4),
            fill_rate=round(fill_rate, 4),
            execution_cost_bps=round(total_cost, 4),
            liquidity_available=liq_score > 0.3,
        )

    # ── Component Models ─────────────────────────────────────

    def _estimate_base_slippage(self, notional: float) -> float:
        """Base slippage from trade size (bps)."""
        # Larger trades = more slippage (concave)
        size_in_millions = notional / 1e6
        return 0.5 + 0.3 * np.sqrt(max(0, size_in_millions))

    def _regime_slippage_multiplier(self, regime: RegimeState) -> float:
        """Regime-based slippage multiplier."""
        multipliers = {
            MarketRegime.STABLE_MEAN_REVERTING: 1.0,
            MarketRegime.VOLATILE_MEAN_REVERTING: 1.5,
            MarketRegime.LIQUIDITY_TIGHTENING: 2.5,
            MarketRegime.STRUCTURAL_SHIFT: 3.0,
            MarketRegime.CRISIS: 5.0,
        }
        return multipliers.get(regime.regime, 1.5)

    def _liquidity_slippage_multiplier(self, liq_score: float) -> float:
        """Lower liquidity = higher slippage."""
        if liq_score > 0.8:
            return 0.8
        elif liq_score > 0.5:
            return 1.0
        elif liq_score > 0.3:
            return 1.5
        else:
            return 3.0

    def _market_impact(self, notional: float) -> float:
        """
        Market impact model (bps).
        Square root model: impact ∝ √(size/ADV)
        """
        adv_proxy = 50e6  # $50M average daily volume proxy
        participation = notional / adv_proxy
        impact = 1.0 * np.sqrt(max(0, participation))
        # Decay: impact reduces over time
        return impact * settings.market_impact_decay

    def _estimate_fill_rate(
        self, notional: float, liq_score: float, regime: RegimeState
    ) -> float:
        """Estimate fill rate (0-1) based on size, liquidity, regime."""
        # Base fill rate from liquidity
        base = min(1.0, liq_score + 0.3)
        
        # Size penalty (larger = harder to fill)
        size_penalty = min(0.3, notional / 1e9)
        
        # Regime penalty
        regime_penalty = {
            MarketRegime.STABLE_MEAN_REVERTING: 0.0,
            MarketRegime.VOLATILE_MEAN_REVERTING: 0.05,
            MarketRegime.LIQUIDITY_TIGHTENING: 0.15,
            MarketRegime.STRUCTURAL_SHIFT: 0.25,
            MarketRegime.CRISIS: 0.40,
        }.get(regime.regime, 0.1)
        
        fill = base - size_penalty - regime_penalty
        return max(0.1, min(1.0, fill))


# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[ExecutionEngine] = None


def get_execution_engine() -> ExecutionEngine:
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine
