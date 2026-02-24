"""
Layer 3 — Regime Classification Engine (THE DIFFERENTIATOR)

The core survival mechanism. Continuously monitors:
  - Volatility clustering
  - Correlation expansion detection
  - Liquidity contraction
  - Funding stress
  - Cross-asset stress signals

Classifies into 5 regimes, each with different:
  - Leverage cap
  - Half-life tolerance
  - Trade sizing model
  - Stop discipline

This is how you survive LTCM-style failure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import MarketRegime, RegimeState
from app.agents.layer1_data.market_data_agent import CurveMatrix


# ── Regime Parameters ────────────────────────────────────────────

REGIME_PARAMS = {
    MarketRegime.STABLE_MEAN_REVERTING: {
        "leverage_cap": 8.0,
        "halflife_tolerance": 90,
        "description": "Normal markets — full alpha generation",
    },
    MarketRegime.VOLATILE_MEAN_REVERTING: {
        "leverage_cap": 5.0,
        "halflife_tolerance": 60,
        "description": "Elevated vol but mean-reversion intact",
    },
    MarketRegime.LIQUIDITY_TIGHTENING: {
        "leverage_cap": 3.0,
        "halflife_tolerance": 30,
        "description": "Liquidity withdrawing — reduce exposure",
    },
    MarketRegime.STRUCTURAL_SHIFT: {
        "leverage_cap": 2.0,
        "halflife_tolerance": 15,
        "description": "Regime change — old relationships may be broken",
    },
    MarketRegime.CRISIS: {
        "leverage_cap": 1.0,
        "halflife_tolerance": 5,
        "description": "Full crisis — survival mode only",
    },
}


class RegimeClassificationEngine:
    """
    Layer 3: Market Regime Classifier.

    Uses multiple signals to classify the current market regime
    and provide dynamic risk parameters.
    """

    def __init__(self):
        self._current_regime: Optional[RegimeState] = None
        self._regime_history: list = []
        self._regime_start: Optional[datetime] = None
        logger.info("Layer 3 — Regime Classification Engine initialized")

    # ── Public API ───────────────────────────────────────────

    async def classify_regime(self, curve_matrix: CurveMatrix) -> RegimeState:
        """
        Run full regime classification pipeline.
        
        Returns current regime state with dynamic parameters.
        """
        # Compute individual signals
        vol_signal = self._assess_volatility(curve_matrix)
        corr_signal = self._assess_correlation(curve_matrix)
        liq_signal = self._assess_liquidity(curve_matrix)
        funding_signal = curve_matrix.funding_stress_index
        stress_signal = self._assess_cross_asset_stress(curve_matrix)

        # Classify
        regime = self._classify(vol_signal, corr_signal, liq_signal, funding_signal, stress_signal)
        params = REGIME_PARAMS[regime]

        # Compute transition probabilities
        transitions = self._estimate_transitions(regime)

        # Regime duration
        now = datetime.utcnow()
        if self._current_regime is None or self._current_regime.regime != regime:
            self._regime_start = now
            duration = 0
        else:
            duration = (now - self._regime_start).days if self._regime_start else 0

        state = RegimeState(
            regime=regime,
            confidence=self._compute_confidence(vol_signal, corr_signal, liq_signal, funding_signal),
            vol_percentile=vol_signal,
            correlation_level=corr_signal,
            liquidity_index=liq_signal,
            funding_stress=funding_signal,
            leverage_cap=params["leverage_cap"],
            halflife_tolerance=params["halflife_tolerance"],
            regime_duration_days=duration,
            transition_probability=transitions,
        )

        # Track history
        if self._current_regime is None or self._current_regime.regime != regime:
            logger.warning(
                f"REGIME CHANGE: {self._current_regime.regime.value if self._current_regime else 'INIT'} "
                f"→ {regime.value} (confidence={state.confidence:.2f})"
            )
        
        self._current_regime = state
        self._regime_history.append(state)

        return state

    @property
    def current_regime(self) -> Optional[RegimeState]:
        return self._current_regime

    # ── Signal Assessment ────────────────────────────────────

    def _assess_volatility(self, matrix: CurveMatrix) -> float:
        """
        Volatility percentile (0-1).
        High values = elevated vol.
        """
        vols = matrix.volatilities
        if vols.empty:
            return 0.5

        # Current vol vs historical distribution
        recent_vols = []
        for col in vols.columns:
            series = vols[col].dropna()
            if len(series) >= 60:
                current = series.iloc[-1]
                pctile = (series < current).mean()
                recent_vols.append(pctile)

        if not recent_vols:
            return 0.5

        return float(np.mean(recent_vols))

    def _assess_correlation(self, matrix: CurveMatrix) -> float:
        """
        Average cross-correlation level (0-1).
        High values = correlation expansion (risk-off).
        """
        corr = matrix.correlations
        if corr.empty:
            return 0.4

        # Get most recent correlation snapshot
        try:
            # Rolling corr returns multi-index — get last window
            if isinstance(corr.index, pd.MultiIndex):
                last_date = corr.index.get_level_values(0).unique()[-1]
                recent_corr = corr.loc[last_date]
            else:
                recent_corr = corr

            # Average absolute correlation (excluding diagonal)
            if recent_corr.ndim == 2:
                mask = np.ones(recent_corr.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_corr = np.abs(recent_corr.values[mask]).mean()
            else:
                avg_corr = 0.4

            return float(min(1.0, max(0.0, avg_corr)))

        except Exception:
            return 0.4

    def _assess_liquidity(self, matrix: CurveMatrix) -> float:
        """
        Aggregate liquidity index (0-1, 1=abundant).
        """
        scores = matrix.liquidity_scores
        if not scores:
            return 0.5
        return float(np.mean(list(scores.values())))

    def _assess_cross_asset_stress(self, matrix: CurveMatrix) -> float:
        """
        Cross-asset stress indicator (0-1, 1=severe).
        Looks for simultaneous selloffs across rate instruments.
        """
        curves = matrix.curves
        if not curves:
            return 0.3

        # Check if multiple instruments are declining simultaneously
        negative_count = 0
        total = 0
        for sym, df in curves.items():
            if "Close" in df.columns and len(df) >= 5:
                recent_ret = df["Close"].pct_change().tail(5).sum()
                total += 1
                if recent_ret < -0.005:  # Down >0.5% in 5 days
                    negative_count += 1

        if total == 0:
            return 0.3

        # Higher = more stress
        return min(1.0, negative_count / total)

    # ── Classification Logic ─────────────────────────────────

    def _classify(
        self,
        vol: float,
        corr: float,
        liq: float,
        funding: float,
        stress: float,
    ) -> MarketRegime:
        """
        Multi-factor regime classification.
        
        Priority: Crisis > Structural > Liquidity > Volatile > Stable
        """
        # Crisis: multiple severe signals
        crisis_score = (
            0.25 * (1 if vol > 0.9 else 0) +
            0.25 * (1 if corr > settings.correlation_expansion_threshold else 0) +
            0.25 * (1 if liq < 0.2 else 0) +
            0.15 * (1 if funding > 0.7 else 0) +
            0.10 * (1 if stress > 0.7 else 0)
        )
        if crisis_score >= 0.5:
            return MarketRegime.CRISIS

        # Structural shift: high vol + correlation breakdown
        if vol > 0.8 and (corr > 0.7 or corr < 0.15):
            return MarketRegime.STRUCTURAL_SHIFT

        # Liquidity tightening
        if liq < settings.liquidity_contraction_threshold or funding > 0.5:
            return MarketRegime.LIQUIDITY_TIGHTENING

        # Volatile but functional
        if vol > 0.6:
            return MarketRegime.VOLATILE_MEAN_REVERTING

        # Normal
        return MarketRegime.STABLE_MEAN_REVERTING

    def _compute_confidence(self, vol: float, corr: float, liq: float, funding: float) -> float:
        """Confidence in regime classification (0-1)."""
        # Higher if signals are clearly in one direction
        signals = [vol, corr, 1 - liq, funding]
        std = np.std(signals)
        # Low std = signals agree = high confidence
        confidence = max(0.3, 1.0 - std * 2)
        return round(confidence, 4)

    def _estimate_transitions(self, current: MarketRegime) -> Dict[str, float]:
        """Estimate probability of transitioning to other regimes."""
        # Based on historical regime transition matrix (stylized)
        transitions = {
            MarketRegime.STABLE_MEAN_REVERTING: {
                "stable_mean_reverting": 0.85,
                "volatile_mean_reverting": 0.10,
                "liquidity_tightening": 0.03,
                "structural_shift": 0.01,
                "crisis": 0.01,
            },
            MarketRegime.VOLATILE_MEAN_REVERTING: {
                "stable_mean_reverting": 0.20,
                "volatile_mean_reverting": 0.55,
                "liquidity_tightening": 0.15,
                "structural_shift": 0.05,
                "crisis": 0.05,
            },
            MarketRegime.LIQUIDITY_TIGHTENING: {
                "stable_mean_reverting": 0.10,
                "volatile_mean_reverting": 0.15,
                "liquidity_tightening": 0.50,
                "structural_shift": 0.15,
                "crisis": 0.10,
            },
            MarketRegime.STRUCTURAL_SHIFT: {
                "stable_mean_reverting": 0.05,
                "volatile_mean_reverting": 0.15,
                "liquidity_tightening": 0.20,
                "structural_shift": 0.40,
                "crisis": 0.20,
            },
            MarketRegime.CRISIS: {
                "stable_mean_reverting": 0.05,
                "volatile_mean_reverting": 0.15,
                "liquidity_tightening": 0.20,
                "structural_shift": 0.10,
                "crisis": 0.50,
            },
        }
        return transitions.get(current, {})


# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[RegimeClassificationEngine] = None


def get_regime_engine() -> RegimeClassificationEngine:
    global _engine
    if _engine is None:
        _engine = RegimeClassificationEngine()
    return _engine
