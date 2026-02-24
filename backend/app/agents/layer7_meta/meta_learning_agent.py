"""
Layer 7 — Meta Learning & Strategy Decay Monitor

Continuously tracks:
  - Spread half-life drift
  - Regime frequency changes
  - Correlation clustering evolution
  - Crowding proxy trends
  - Model edge decay

Auto-adjusts:
  - Parameter windows
  - Risk weights
  - Leverage scaling

If decay detected → reduce allocation, re-train, adjust ranking.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import DecayMetrics, SpreadCandidate, RegimeState


class MetaLearningAgent:
    """
    Layer 7: Strategy Decay Monitor.

    Detects when alpha is decaying and auto-adjusts system parameters.
    """

    def __init__(self):
        self._decay_history: List[DecayMetrics] = []
        self._halflife_history: Dict[str, List[float]] = {}
        self._edge_history: Dict[str, List[float]] = {}
        self._regime_transitions: List[str] = []
        logger.info("Layer 7 — Meta Learning Agent initialized")

    # ── Public API ───────────────────────────────────────────

    async def assess_decay(
        self,
        candidates: List[SpreadCandidate],
        regime: RegimeState,
    ) -> DecayMetrics:
        """
        Run full decay assessment.
        
        Checks if the strategy's edge is deteriorating.
        """
        # Track half-life drift
        halflife_drift = self._measure_halflife_drift(candidates)
        
        # Track edge decay
        edge_decay = self._measure_edge_decay(candidates)
        
        # Track regime frequency
        self._regime_transitions.append(regime.regime.value)
        regime_freq_change = self._measure_regime_frequency_change()
        
        # Correlation clustering
        corr_clustering = self._measure_correlation_clustering(candidates)
        
        # Crowding increase
        crowding_increase = self._measure_crowding_trend(candidates)
        
        # Determine if retraining needed
        retraining = (
            halflife_drift > settings.retraining_trigger_decay_pct or
            edge_decay > settings.retraining_trigger_decay_pct or
            crowding_increase > 0.3
        )
        
        # Compute parameter adjustments
        adjustments = self._compute_adjustments(
            halflife_drift, edge_decay, regime_freq_change,
            corr_clustering, crowding_increase
        )
        
        metrics = DecayMetrics(
            halflife_drift_pct=round(halflife_drift, 4),
            edge_decay_pct=round(edge_decay, 4),
            regime_frequency_change=round(regime_freq_change, 4),
            correlation_clustering=round(corr_clustering, 4),
            crowding_increase=round(crowding_increase, 4),
            retraining_recommended=retraining,
            parameter_adjustments=adjustments,
        )
        
        self._decay_history.append(metrics)
        
        if retraining:
            logger.warning(
                f"DECAY DETECTED — halflife_drift={halflife_drift:.1f}%, "
                f"edge_decay={edge_decay:.1f}%, crowding={crowding_increase:.2f}"
            )
        
        return metrics

    # ── Measurement Functions ────────────────────────────────

    def _measure_halflife_drift(self, candidates: List[SpreadCandidate]) -> float:
        """Measure drift in half-life estimates vs history."""
        current_halflives = {}
        for c in candidates:
            current_halflives[c.spread_id] = c.halflife_days
            if c.spread_id not in self._halflife_history:
                self._halflife_history[c.spread_id] = []
            self._halflife_history[c.spread_id].append(c.halflife_days)

        # Compare recent vs historical
        drifts = []
        for sid, history in self._halflife_history.items():
            if len(history) >= 5:
                old_mean = np.mean(history[:-3])
                new_mean = np.mean(history[-3:])
                if old_mean > 0:
                    drift_pct = abs(new_mean - old_mean) / old_mean * 100
                    drifts.append(drift_pct)

        return np.mean(drifts) if drifts else 0.0

    def _measure_edge_decay(self, candidates: List[SpreadCandidate]) -> float:
        """Measure decay in expected returns."""
        for c in candidates:
            if c.spread_id not in self._edge_history:
                self._edge_history[c.spread_id] = []
            self._edge_history[c.spread_id].append(c.vol_adjusted_return)

        decays = []
        for sid, history in self._edge_history.items():
            if len(history) >= 5:
                old_mean = np.mean(history[:-3])
                new_mean = np.mean(history[-3:])
                if old_mean > 0:
                    decay = max(0, (old_mean - new_mean) / old_mean * 100)
                    decays.append(decay)

        return np.mean(decays) if decays else 0.0

    def _measure_regime_frequency_change(self) -> float:
        """Measure if regime transitions are accelerating."""
        transitions = self._regime_transitions
        if len(transitions) < 10:
            return 0.0

        # Count transitions in first half vs second half
        mid = len(transitions) // 2
        first_changes = sum(
            1 for i in range(1, mid) 
            if transitions[i] != transitions[i-1]
        )
        second_changes = sum(
            1 for i in range(mid + 1, len(transitions))
            if transitions[i] != transitions[i-1]
        )

        if first_changes == 0:
            return 0.0

        return (second_changes - first_changes) / max(first_changes, 1)

    def _measure_correlation_clustering(self, candidates: List[SpreadCandidate]) -> float:
        """Measure if spread returns are becoming more correlated."""
        if len(candidates) < 3:
            return 0.0

        returns = [c.vol_adjusted_return for c in candidates]
        # Higher std of returns = less clustering
        # Lower std = more clustering (they're all doing the same thing)
        std = np.std(returns)
        mean_abs = np.mean(np.abs(returns)) + 1e-10
        clustering = 1.0 - min(1.0, std / mean_abs)
        return max(0, clustering)

    def _measure_crowding_trend(self, candidates: List[SpreadCandidate]) -> float:
        """Measure if crowding is increasing."""
        if not candidates:
            return 0.0
        current_crowding = np.mean([c.crowding_proxy for c in candidates])
        return current_crowding

    # ── Auto-Adjustment ──────────────────────────────────────

    def _compute_adjustments(
        self,
        halflife_drift: float,
        edge_decay: float,
        regime_freq: float,
        corr_clustering: float,
        crowding: float,
    ) -> Dict[str, float]:
        """Compute parameter adjustments based on decay signals."""
        adjustments = {}

        # If half-lives are drifting, tighten the window
        if halflife_drift > 10:
            new_max_hl = max(30, settings.max_halflife_days - 10)
            adjustments["max_halflife_days"] = new_max_hl

        # If edge is decaying, increase entry threshold
        if edge_decay > 15:
            new_zscore = min(3.0, settings.zscore_entry_threshold + 0.25)
            adjustments["zscore_entry_threshold"] = new_zscore

        # If regimes are changing faster, reduce leverage
        if regime_freq > 0.5:
            adjustments["leverage_reduction_pct"] = 20.0

        # If crowding is high, penalize more
        if crowding > settings.crowding_correlation_threshold:
            adjustments["crowding_penalty_increase"] = 0.2

        # If correlations are clustering, increase diversification requirement
        if corr_clustering > 0.7:
            adjustments["min_spread_diversity"] = 5

        return adjustments


# ── Singleton ────────────────────────────────────────────────────
_agent: Optional[MetaLearningAgent] = None


def get_meta_learning_agent() -> MetaLearningAgent:
    global _agent
    if _agent is None:
        _agent = MetaLearningAgent()
    return _agent
