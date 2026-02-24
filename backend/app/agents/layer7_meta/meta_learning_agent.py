"""
Layer 7 — Meta Learning & Strategy Decay Monitor

Advanced features (state-of-the-art 2024/2025):
  - CUSUM (Cumulative Sum) change-point detection
  - Exponential forgetting for edge estimation
  - Bayesian z-score threshold shrinkage
  - Spread half-life drift tracking
  - Regime frequency change monitoring
  - Correlation clustering evolution
  - Crowding proxy trends
  - Auto-parameter adjustment

If decay detected → reduce allocation, adjust z-score entry, flag retraining.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import DecayMetrics, SpreadCandidate, RegimeState


# ── CUSUM Change-Point Detector ──────────────────────────────────

class CUSUMDetector:
    """
    Tabular CUSUM (Page's test) for detecting mean shifts
    in a time series of edge estimates.

    Maintains running statistics S_t = max(0, S_{t-1} + (x_t - μ₀ - k))
    where k = allowance (sensitivity), μ₀ = target mean.
    Alert when S_t > h (decision interval).
    """

    def __init__(self, k: float = 0.5, h: float = 5.0):
        """
        Args:
            k: allowance (sensitivity). Smaller = more sensitive.
            h: decision interval (threshold). Smaller = faster detection.
        """
        self.k = k
        self.h = h
        self._S_pos = 0.0  # Positive CUSUM (detect upward shift)
        self._S_neg = 0.0  # Negative CUSUM (detect downward shift)
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self._n_observations = 0

    def update(self, x: float) -> tuple:
        """
        Ingest one observation. Returns (cusum_stat, alert).
        """
        self._n_observations += 1

        # Initialize target from first observation
        if self._target_mean is None:
            self._target_mean = x
            self._target_std = abs(x) * 0.1 + 1e-6
            return 0.0, False

        # Normalize
        z = (x - self._target_mean) / (self._target_std + 1e-10)

        # Update CUSUM
        self._S_pos = max(0, self._S_pos + z - self.k)
        self._S_neg = max(0, self._S_neg - z - self.k)

        cusum_stat = max(self._S_pos, self._S_neg)
        alert = cusum_stat > self.h

        if alert:
            # Reset after alert (optional restart)
            self._target_mean = x
            self._S_pos = 0.0
            self._S_neg = 0.0

        return cusum_stat, alert

    def reset(self, new_mean: float, new_std: float):
        self._target_mean = new_mean
        self._target_std = new_std
        self._S_pos = 0.0
        self._S_neg = 0.0


# ── Exponential Forgetting Edge Estimator ────────────────────────

class ExponentialForgetting:
    """
    Exponentially weighted moving average (EWMA) for online edge tracking.
    
    μ_t = λ · μ_{t-1} + (1-λ) · x_t
    σ²_t = λ · σ²_{t-1} + (1-λ) · (x_t - μ_t)²

    λ close to 1 = long memory, λ close to 0 = short memory.
    """

    def __init__(self, lam: float = 0.95):
        self.lam = lam
        self._mean: Optional[float] = None
        self._var: float = 0.0

    def update(self, x: float) -> tuple:
        """Returns (ewma_mean, ewma_std)."""
        if self._mean is None:
            self._mean = x
            return x, 0.0

        self._mean = self.lam * self._mean + (1 - self.lam) * x
        self._var = self.lam * self._var + (1 - self.lam) * (x - self._mean) ** 2
        return self._mean, np.sqrt(max(0, self._var))

    @property
    def mean(self) -> float:
        return self._mean if self._mean is not None else 0.0


# ── Bayesian Z-Score Threshold Shrinkage ─────────────────────────

def bayesian_zscore_shrinkage(
    observed_zscores: List[float],
    prior_mean: float = 2.0,
    prior_var: float = 0.5,
) -> float:
    """
    Bayesian shrinkage of the z-score entry threshold.
    
    Uses a Normal-Normal conjugate model:
      Prior: μ ~ N(prior_mean, prior_var)
      Likelihood: x_i ~ N(μ, σ²)
      Posterior: μ | data ~ N(posterior_mean, posterior_var)

    When data is scarce, threshold shrinks toward conservative prior.
    As data accumulates, threshold adapts to observed optimal entry points.
    """
    if not observed_zscores or len(observed_zscores) < 3:
        return prior_mean

    x_bar = np.mean(observed_zscores)
    sigma_sq = np.var(observed_zscores)
    n = len(observed_zscores)

    # Posterior precision = prior precision + data precision
    prior_precision = 1.0 / prior_var
    data_precision = n / (sigma_sq + 1e-10)

    posterior_var = 1.0 / (prior_precision + data_precision)
    posterior_mean = posterior_var * (prior_precision * prior_mean + data_precision * x_bar)

    return float(np.clip(posterior_mean, 1.0, 4.0))


class MetaLearningAgent:
    """
    Layer 7: Strategy Decay Monitor with online learning.

    Detects when alpha is decaying via CUSUM change-point detection,
    tracks edge with exponential forgetting, and adapts z-score entry
    threshold via Bayesian shrinkage.
    """

    def __init__(self):
        self._decay_history: List[DecayMetrics] = []
        self._halflife_history: Dict[str, List[float]] = {}
        self._edge_history: Dict[str, List[float]] = {}
        self._regime_transitions: List[str] = []
        # Online learning components
        self._cusum = CUSUMDetector(k=0.5, h=5.0)
        self._ewma_edge = ExponentialForgetting(lam=0.95)
        self._successful_entry_zscores: List[float] = []
        logger.info("Layer 7 — Meta Learning Agent initialized (CUSUM + EWMA + Bayesian)")

    # ── Public API ───────────────────────────────────────────

    async def assess_decay(
        self,
        candidates: List[SpreadCandidate],
        regime: RegimeState,
    ) -> DecayMetrics:
        """
        Run full decay assessment with online learning diagnostics.
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

        # ── CUSUM change-point detection ─────────────────────
        # Feed aggregate edge into CUSUM
        avg_edge = np.mean([c.vol_adjusted_return for c in candidates]) if candidates else 0
        cusum_stat, cusum_alert = self._cusum.update(avg_edge)

        # ── Exponential forgetting edge tracker ──────────────
        ewma_mean, ewma_std = self._ewma_edge.update(avg_edge)

        # ── Bayesian z-score threshold ───────────────────────
        # Track z-scores of successful (profitable) entries
        for c in candidates:
            if c.signal in ("enter_long", "enter_short") and c.vol_adjusted_return > 0:
                self._successful_entry_zscores.append(abs(c.zscore))
        bayesian_threshold = bayesian_zscore_shrinkage(
            self._successful_entry_zscores,
            prior_mean=settings.zscore_entry_threshold,
        )

        # Determine if retraining needed
        retraining = (
            halflife_drift > settings.retraining_trigger_decay_pct or
            edge_decay > settings.retraining_trigger_decay_pct or
            crowding_increase > 0.3 or
            cusum_alert  # CUSUM detected structural change
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
            # Online learning fields
            cusum_statistic=round(cusum_stat, 4),
            cusum_alert=cusum_alert,
            ewma_edge_estimate=round(ewma_mean, 4),
            forgetting_factor=self._ewma_edge.lam,
            bayesian_zscore_threshold=round(bayesian_threshold, 4),
        )
        
        self._decay_history.append(metrics)
        
        if retraining:
            reason = "CUSUM alert" if cusum_alert else "threshold breach"
            logger.warning(
                f"DECAY DETECTED ({reason}) — halflife_drift={halflife_drift:.1f}%, "
                f"edge_decay={edge_decay:.1f}%, cusum={cusum_stat:.2f}, "
                f"bayesian_z_threshold={bayesian_threshold:.2f}"
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
