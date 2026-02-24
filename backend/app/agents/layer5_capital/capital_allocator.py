"""
Layer 5 — Adaptive Capital Allocation & Leverage Optimizer

Maximizes geometric growth subject to survival constraints.

Advanced features (state-of-the-art 2024/2025):
  - Drawdown-controlled dynamic Kelly fraction
  - Hurst-exponent quality weighting
  - Cointegration-confidence scaling
  - Regime-adaptive multiplier with HMM confidence
  - Liquidity & crowding penalties

Method:
  - Base Kelly: f* = μ/σ² (full Kelly)
  - Fractional Kelly: f = κ · f* where κ ∈ [0.15, 0.45]
  - Drawdown control: κ_dd = κ · max(0, 1 - dd/dd_max)
  - Final: f_final = f * regime_adj * liq_adj * crowd_adj * quality_adj
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

from app.core.config import settings
from app.core.types import (
    SpreadCandidate, PortfolioPosition, RegimeState,
    MarketRegime, TradeSignal,
)


class CapitalAllocator:
    """
    Layer 5: Adaptive Leverage Optimizer.
    
    Objective: Maximize geometric growth subject to survival.
    """

    def __init__(self):
        self._allocation_history: List[Dict] = []
        self._current_drawdown_pct: float = 0.0  # tracked across cycles
        self._peak_nav: float = 0.0
        logger.info("Layer 5 — Capital Allocator initialized (drawdown-controlled Kelly)")

    def update_drawdown(self, current_nav: float) -> float:
        """Track drawdown from peak NAV. Call before allocate()."""
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav
        if self._peak_nav > 0:
            self._current_drawdown_pct = (self._peak_nav - current_nav) / self._peak_nav * 100
        return self._current_drawdown_pct

    # ── Public API ───────────────────────────────────────────

    async def allocate(
        self,
        candidates: List[SpreadCandidate],
        regime: RegimeState,
        nav: float = 100_000_000.0,  # $100M default
        existing_positions: Optional[List[PortfolioPosition]] = None,
    ) -> List[PortfolioPosition]:
        """
        Construct optimal portfolio from ranked candidates.

        Steps:
        1. Filter to actionable signals
        2. Compute Kelly fractions
        3. Apply regime adjustment
        4. Apply liquidity penalty
        5. Apply crowding penalty
        6. Normalize to leverage cap
        7. Build positions
        """
        existing_positions = existing_positions or []

        # Step 1: Only consider actionable signals
        actionable = [
            c for c in candidates
            if c.signal in [TradeSignal.ENTER_LONG, TradeSignal.ENTER_SHORT]
        ]

        if not actionable:
            logger.info("No actionable opportunities")
            return existing_positions

        logger.info(f"Allocating across {len(actionable)} opportunities")

        # Step 2: Compute raw Kelly fraction per candidate
        raw_weights = self._compute_kelly_weights(actionable)

        # Step 3: Regime adjustment
        regime_weights = self._apply_regime_adjustment(raw_weights, actionable, regime)

        # Step 4: Liquidity penalty
        liq_weights = self._apply_liquidity_penalty(regime_weights, actionable)

        # Step 5: Crowding penalty
        crowd_weights = self._apply_crowding_penalty(liq_weights, actionable)

        # Step 6: Normalize to leverage cap
        final_weights = self._normalize_to_leverage(crowd_weights, regime)

        # Step 7: Build positions
        positions = self._build_positions(actionable, final_weights, nav)

        # Log allocation
        total_leverage = sum(abs(w) for w in final_weights.values())
        self._allocation_history.append({
            "n_positions": len(positions),
            "total_leverage": total_leverage,
            "regime": regime.regime.value,
            "nav": nav,
        })
        logger.info(
            f"Allocated {len(positions)} positions, "
            f"leverage={total_leverage:.2f}x, "
            f"regime={regime.regime.value}"
        )

        return positions

    # ── Drawdown-Controlled Kelly ──────────────────────────────

    def _compute_kelly_weights(
        self, candidates: List[SpreadCandidate]
    ) -> Dict[str, float]:
        """
        Compute drawdown-controlled fractional Kelly weight per candidate.
        
        Kelly f* = (μ/σ²)
        Fractional: f = κ · f*
        Drawdown control: κ_dd = κ · max(0, 1 - dd / dd_max)^2
        Quality boost: × (1 + 0.3 · (0.5 - H)) for mean-reverting Hurst
        Cointegration boost: × 1.2 if Johansen confirms cointegration
        """
        weights = {}

        # Dynamic Kelly fraction — shrink aggressively as drawdown grows
        dd_ratio = self._current_drawdown_pct / settings.max_drawdown_pct
        dd_multiplier = max(0.0, 1.0 - dd_ratio) ** 2  # quadratic decay

        base_fraction = settings.kelly_fraction_default
        effective_fraction = base_fraction * dd_multiplier

        if dd_ratio > 0.5:
            logger.warning(
                f"Drawdown control active: dd={self._current_drawdown_pct:.1f}%, "
                f"Kelly fraction scaled to {effective_fraction:.3f}"
            )

        for c in candidates:
            if c.expected_shortfall_bps <= 0:
                weights[c.spread_id] = 0
                continue

            # Raw Kelly: f* = μ/σ²
            # Use z-score-based edge estimate: the expected mean-reversion
            # return is |z| * σ annualised over the half-life horizon.
            vol = c.std_spread_bps + 1e-10
            edge_sigma = abs(c.zscore)  # Sharpe-like edge estimate
            raw_kelly = edge_sigma / (abs(c.zscore) + 1.0)  # bounded [0, 1)

            # Apply drawdown-controlled fraction
            kelly_weight = raw_kelly * effective_fraction

            # Quality adjustment: Hurst < 0.5 is mean-reverting (good)
            hurst_boost = 1.0 + 0.3 * max(0, 0.5 - c.hurst_exponent)
            kelly_weight *= hurst_boost

            # Cointegration confidence boost
            if c.is_cointegrated:
                kelly_weight *= 1.15  # 15% boost for confirmed cointegration

            # Clamp
            kelly_weight = max(0.001, min(0.2, kelly_weight))
            weights[c.spread_id] = kelly_weight

        return weights

    # ── Adjustments ──────────────────────────────────────────

    def _apply_regime_adjustment(
        self,
        weights: Dict[str, float],
        candidates: List[SpreadCandidate],
        regime: RegimeState,
    ) -> Dict[str, float]:
        """Scale weights by regime multiplier, blended with HMM confidence."""
        multipliers = {
            MarketRegime.STABLE_MEAN_REVERTING: 1.0,
            MarketRegime.VOLATILE_MEAN_REVERTING: 0.7,
            MarketRegime.LIQUIDITY_TIGHTENING: 0.4,
            MarketRegime.STRUCTURAL_SHIFT: 0.2,
            MarketRegime.CRISIS: 0.0,  # No new allocation in crisis
        }
        base_mult = multipliers.get(regime.regime, 0.5)

        # If HMM confidence is low (ensemble disagrees), be more conservative
        confidence_adj = 0.7 + 0.3 * regime.ensemble_agreement  # range [0.7, 1.0]
        mult = base_mult * confidence_adj

        return {k: v * mult for k, v in weights.items()}

    def _apply_liquidity_penalty(
        self,
        weights: Dict[str, float],
        candidates: List[SpreadCandidate],
    ) -> Dict[str, float]:
        """Reduce weight for illiquid spreads."""
        adjusted = {}
        for c in candidates:
            w = weights.get(c.spread_id, 0)
            # Liquidity penalty: weight * liquidity_score
            liq_adj = w * max(0.1, c.liquidity_score)
            adjusted[c.spread_id] = liq_adj
        return adjusted

    def _apply_crowding_penalty(
        self,
        weights: Dict[str, float],
        candidates: List[SpreadCandidate],
    ) -> Dict[str, float]:
        """Reduce weight for crowded trades."""
        adjusted = {}
        for c in candidates:
            w = weights.get(c.spread_id, 0)
            # Higher crowding → lower weight
            crowd_penalty = 1.0 - c.crowding_proxy * 0.5
            adjusted[c.spread_id] = w * max(0.1, crowd_penalty)
        return adjusted

    def _normalize_to_leverage(
        self,
        weights: Dict[str, float],
        regime: RegimeState,
    ) -> Dict[str, float]:
        """Normalize total weight to not exceed leverage cap."""
        total = sum(abs(w) for w in weights.values())
        if total <= 0:
            return weights

        max_lev = min(regime.leverage_cap, settings.max_leverage)
        if total > max_lev:
            scale = max_lev / total
            return {k: v * scale for k, v in weights.items()}

        return weights

    # ── Position Construction ────────────────────────────────

    def _build_positions(
        self,
        candidates: List[SpreadCandidate],
        weights: Dict[str, float],
        nav: float,
    ) -> List[PortfolioPosition]:
        """Convert weights into portfolio positions."""
        positions = []
        for c in candidates:
            w = weights.get(c.spread_id, 0)
            if abs(w) < 0.001:
                continue

            direction = "long_spread" if c.signal == TradeSignal.ENTER_LONG else "short_spread"
            notional = nav * abs(w)

            positions.append(PortfolioPosition(
                position_id=f"pos_{c.spread_id}_{int(datetime.utcnow().timestamp())}",
                spread_id=c.spread_id,
                spread_type=c.spread_type,
                leg1_instrument=c.leg1_instrument,
                leg2_instrument=c.leg2_instrument,
                direction=direction,
                notional=round(notional, 2),
                dv01_net=0.0,  # To be computed by risk engine
                convexity_net=0.0,
                entry_spread_bps=c.current_spread_bps,
                current_spread_bps=c.current_spread_bps,
                entry_zscore=c.zscore,
                current_zscore=c.zscore,
                unrealized_pnl_bps=0.0,
                weight=round(w, 6),
                leverage_contribution=round(abs(w), 6),
            ))

        return positions


from datetime import datetime

# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[CapitalAllocator] = None


def get_capital_allocator() -> CapitalAllocator:
    global _engine
    if _engine is None:
        _engine = CapitalAllocator()
    return _engine
