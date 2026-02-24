"""
Layer 3 — Regime Classification Engine (THE DIFFERENTIATOR)

The core survival mechanism. Uses a Gaussian Hidden Markov Model (HMM)
for probabilistic regime detection with learned transition matrices.

Advanced features (state-of-the-art 2024/2025):
  - Gaussian HMM with Baum-Welch EM fitting for 5 latent states
  - Online transition matrix learning from observed regime sequences
  - Ensemble classification: HMM posterior + heuristic signals
  - Regime persistence filtering (avoids whipsaw on noisy transitions)
  - Adaptive confidence via signal agreement scoring

Continuously monitors:
  - Volatility clustering (realized vol percentile)
  - Correlation expansion detection (risk-off signal)
  - Liquidity contraction (volume-weighted)
  - Funding stress (repo/OIS spreads)
  - Cross-asset stress signals (simultaneous selloffs)

Classifies into 5 regimes, each with different:
  - Leverage cap
  - Half-life tolerance
  - Trade sizing model
  - Stop discipline

This is how you survive LTCM-style failure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
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

# Ordered regime list for HMM state mapping (index → regime)
REGIME_ORDER = [
    MarketRegime.STABLE_MEAN_REVERTING,
    MarketRegime.VOLATILE_MEAN_REVERTING,
    MarketRegime.LIQUIDITY_TIGHTENING,
    MarketRegime.STRUCTURAL_SHIFT,
    MarketRegime.CRISIS,
]


class GaussianHMM:
    """
    Lightweight Gaussian Hidden Markov Model with EM (Baum-Welch).
    
    5 hidden states, 4 observable features:
      [vol_percentile, correlation_level, 1-liquidity, funding_stress]
    
    Uses online Baum-Welch: fits on a sliding window of recent observations
    to adapt the model to structural changes in market dynamics.
    """

    def __init__(self, n_states: int = 5, n_features: int = 4):
        self.n_states = n_states
        self.n_features = n_features
        self._fitted = False

        # Initialize transition matrix with prior: regimes are sticky
        self.transition_matrix = self._init_transition_matrix()
        # Initial state distribution: bias towards stable
        self.initial_probs = np.array([0.50, 0.25, 0.12, 0.08, 0.05])

        # Emission parameters: means and covariances per state
        # [vol_pctile, corr, illiquidity, funding_stress]
        self.means = np.array([
            [0.30, 0.35, 0.25, 0.15],  # Stable
            [0.65, 0.50, 0.35, 0.25],  # Volatile
            [0.50, 0.45, 0.65, 0.45],  # Liquidity
            [0.75, 0.70, 0.55, 0.65],  # Structural
            [0.92, 0.85, 0.80, 0.80],  # Crisis
        ])
        # Diagonal covariances (simplified)
        self.covs = np.array([
            np.diag([0.04, 0.04, 0.04, 0.04]),
            np.diag([0.06, 0.05, 0.05, 0.05]),
            np.diag([0.05, 0.05, 0.06, 0.06]),
            np.diag([0.06, 0.06, 0.06, 0.07]),
            np.diag([0.03, 0.03, 0.04, 0.04]),  # Crisis: tight cluster
        ])

        # Observation history for online fitting
        self._obs_history: List[np.ndarray] = []
        self._max_history = 200  # Rolling window for EM

    def _init_transition_matrix(self) -> np.ndarray:
        """Initialize with stylized prior: regimes are sticky."""
        T = np.array([
            [0.85, 0.10, 0.03, 0.01, 0.01],  # Stable
            [0.20, 0.55, 0.15, 0.05, 0.05],  # Volatile
            [0.10, 0.15, 0.50, 0.15, 0.10],  # Liquidity
            [0.05, 0.15, 0.20, 0.40, 0.20],  # Structural
            [0.05, 0.15, 0.20, 0.10, 0.50],  # Crisis
        ])
        return T

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Multivariate Gaussian probability density."""
        d = len(mean)
        diff = x - mean
        det = np.linalg.det(cov)
        if det < 1e-30:
            det = 1e-30
        inv_cov = np.linalg.inv(cov + np.eye(d) * 1e-6)
        exponent = -0.5 * diff @ inv_cov @ diff
        norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
        return norm * np.exp(exponent)

    def predict_proba(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities P(state | observation).
        Uses forward algorithm on recent history if available.
        """
        # Store observation
        self._obs_history.append(observation)
        if len(self._obs_history) > self._max_history:
            self._obs_history = self._obs_history[-self._max_history:]

        # Emission probabilities for current observation
        emissions = np.array([
            self._gaussian_pdf(observation, self.means[s], self.covs[s])
            for s in range(self.n_states)
        ])
        emissions = np.maximum(emissions, 1e-30)

        # Forward step using last posterior as prior
        if len(self._obs_history) == 1:
            prior = self.initial_probs
        else:
            # Use transition matrix to propagate last posterior
            prior = self._last_posterior @ self.transition_matrix

        posterior = prior * emissions
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = self.initial_probs.copy()

        self._last_posterior = posterior
        return posterior

    def fit_online(self, min_obs: int = 30) -> None:
        """
        Online Baum-Welch EM step on recent observation window.
        Updates emission means/covs and transition matrix.
        """
        if len(self._obs_history) < min_obs:
            return

        observations = np.array(self._obs_history[-min_obs:])
        T_obs = len(observations)

        # E-step: Forward-backward to get state responsibilities
        alpha = np.zeros((T_obs, self.n_states))
        beta = np.zeros((T_obs, self.n_states))

        # Forward pass
        emissions = np.zeros((T_obs, self.n_states))
        for t in range(T_obs):
            for s in range(self.n_states):
                emissions[t, s] = self._gaussian_pdf(
                    observations[t], self.means[s], self.covs[s]
                )
        emissions = np.maximum(emissions, 1e-30)

        alpha[0] = self.initial_probs * emissions[0]
        alpha[0] /= alpha[0].sum() + 1e-30

        for t in range(1, T_obs):
            alpha[t] = (alpha[t-1] @ self.transition_matrix) * emissions[t]
            alpha[t] /= alpha[t].sum() + 1e-30

        # Backward pass
        beta[-1] = 1.0
        for t in range(T_obs - 2, -1, -1):
            beta[t] = self.transition_matrix @ (emissions[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-30

        # Responsibilities (gamma)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-30

        # M-step: Update parameters with smoothing
        learning_rate = 0.3  # Blend new estimates with prior
        for s in range(self.n_states):
            resp = gamma[:, s]
            total_resp = resp.sum() + 1e-10

            new_mean = (resp[:, None] * observations).sum(axis=0) / total_resp
            diff = observations - new_mean
            new_cov = (resp[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / total_resp
            # Regularize covariance
            new_cov += np.eye(self.n_features) * 0.01

            self.means[s] = (1 - learning_rate) * self.means[s] + learning_rate * new_mean
            self.covs[s] = (1 - learning_rate) * self.covs[s] + learning_rate * new_cov

        # Update transition matrix from xi (pairwise state probabilities)
        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(T_obs - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi_sum[i, j] += (
                        alpha[t, i] * self.transition_matrix[i, j]
                        * emissions[t+1, j] * beta[t+1, j]
                    )

        row_sums = xi_sum.sum(axis=1, keepdims=True) + 1e-30
        new_trans = xi_sum / row_sums
        self.transition_matrix = (
            (1 - learning_rate) * self.transition_matrix
            + learning_rate * new_trans
        )

        self._fitted = True

    _last_posterior: np.ndarray = np.array([0.50, 0.25, 0.12, 0.08, 0.05])


class RegimeClassificationEngine:
    """
    Layer 3: Market Regime Classifier.

    Ensemble method combining:
      1. Gaussian HMM with learned transition matrix (probabilistic)
      2. Heuristic signal thresholds (interpretable fallback)
      3. Regime persistence filter (anti-whipsaw)

    HMM is updated online via Baum-Welch every N observations.
    """

    def __init__(self):
        self._current_regime: Optional[RegimeState] = None
        self._regime_history: list = []
        self._regime_start: Optional[datetime] = None
        self._hmm = GaussianHMM(n_states=5, n_features=4)
        self._hmm_update_interval = 10  # Refit HMM every N cycles
        self._cycle_count = 0
        self._persistence_count = 0  # Consecutive cycles in same regime
        self._persistence_threshold = 2  # Require N confirmations before switching
        self._pending_regime: Optional[MarketRegime] = None
        logger.info("Layer 3 — Regime Classification Engine initialized (HMM + Ensemble)")

    # ── Public API ───────────────────────────────────────────

    async def classify_regime(self, curve_matrix: CurveMatrix) -> RegimeState:
        """
        Run full regime classification pipeline.
        
        Ensemble: HMM posterior (60% weight) + heuristic (40% weight)
        with persistence filtering to avoid whipsaw.
        """
        self._cycle_count += 1

        # Compute individual signals
        vol_signal = self._assess_volatility(curve_matrix)
        corr_signal = self._assess_correlation(curve_matrix)
        liq_signal = self._assess_liquidity(curve_matrix)
        funding_signal = curve_matrix.funding_stress_index
        stress_signal = self._assess_cross_asset_stress(curve_matrix)

        # ── HMM Classification ───────────────────────────────
        obs = np.array([vol_signal, corr_signal, 1.0 - liq_signal, funding_signal])
        hmm_posteriors = self._hmm.predict_proba(obs)
        hmm_regime_idx = int(np.argmax(hmm_posteriors))
        hmm_regime = REGIME_ORDER[hmm_regime_idx]
        hmm_confidence = float(hmm_posteriors[hmm_regime_idx])

        # Online HMM update (Baum-Welch)
        if self._cycle_count % self._hmm_update_interval == 0:
            self._hmm.fit_online(min_obs=30)
            logger.info("HMM online Baum-Welch update completed")

        # ── Heuristic Classification ─────────────────────────
        heuristic_regime = self._classify_heuristic(
            vol_signal, corr_signal, liq_signal, funding_signal, stress_signal
        )
        heuristic_confidence = self._compute_confidence(
            vol_signal, corr_signal, liq_signal, funding_signal
        )

        # ── Ensemble ─────────────────────────────────────────
        # Build weighted probability over regimes
        hmm_weight = 0.6
        heuristic_weight = 0.4
        ensemble_probs = np.zeros(5)
        ensemble_probs += hmm_weight * hmm_posteriors
        # Heuristic: put all weight on its chosen regime
        h_idx = REGIME_ORDER.index(heuristic_regime)
        ensemble_probs[h_idx] += heuristic_weight

        ensemble_idx = int(np.argmax(ensemble_probs))
        raw_regime = REGIME_ORDER[ensemble_idx]
        ensemble_confidence = float(ensemble_probs[ensemble_idx])

        # ── Persistence Filter (anti-whipsaw) ────────────────
        regime = self._apply_persistence_filter(raw_regime)

        params = REGIME_PARAMS[regime]

        # Get learned transition probabilities from HMM
        regime_idx = REGIME_ORDER.index(regime)
        hmm_transitions = self._hmm.transition_matrix[regime_idx]
        transitions = {
            REGIME_ORDER[i].value: round(float(hmm_transitions[i]), 4)
            for i in range(5)
        }

        # Regime duration
        now = datetime.utcnow()
        if self._current_regime is None or self._current_regime.regime != regime:
            self._regime_start = now
            duration = 0
        else:
            duration = (now - self._regime_start).days if self._regime_start else 0

        state = RegimeState(
            regime=regime,
            confidence=round(ensemble_confidence, 4),
            vol_percentile=vol_signal,
            correlation_level=corr_signal,
            liquidity_index=liq_signal,
            funding_stress=funding_signal,
            leverage_cap=params["leverage_cap"],
            halflife_tolerance=params["halflife_tolerance"],
            regime_duration_days=duration,
            transition_probability=transitions,
            hmm_state_probabilities={
                REGIME_ORDER[i].value: round(float(hmm_posteriors[i]), 4)
                for i in range(5)
            },
            hmm_confidence=round(hmm_confidence, 4),
            heuristic_regime=heuristic_regime.value,
            ensemble_agreement=1.0 if hmm_regime == heuristic_regime else 0.0,
        )

        # Track history
        if self._current_regime is None or self._current_regime.regime != regime:
            logger.warning(
                f"REGIME CHANGE: {self._current_regime.regime.value if self._current_regime else 'INIT'} "
                f"→ {regime.value} (confidence={state.confidence:.2f}, "
                f"hmm={hmm_regime.value}, heuristic={heuristic_regime.value})"
            )
        
        self._current_regime = state
        self._regime_history.append(state)

        return state

    @property
    def current_regime(self) -> Optional[RegimeState]:
        return self._current_regime

    # ── Persistence Filter ───────────────────────────────────

    def _apply_persistence_filter(self, raw_regime: MarketRegime) -> MarketRegime:
        """
        Anti-whipsaw: require N consecutive observations of a new regime
        before switching. Exception: CRISIS overrides immediately.
        """
        if raw_regime == MarketRegime.CRISIS:
            # Crisis always takes effect immediately
            self._pending_regime = None
            self._persistence_count = 0
            return MarketRegime.CRISIS

        current = self._current_regime.regime if self._current_regime else None

        if raw_regime == current:
            self._pending_regime = None
            self._persistence_count = 0
            return raw_regime

        # New regime detected
        if raw_regime == self._pending_regime:
            self._persistence_count += 1
        else:
            self._pending_regime = raw_regime
            self._persistence_count = 1

        if self._persistence_count >= self._persistence_threshold:
            self._pending_regime = None
            self._persistence_count = 0
            return raw_regime

        # Not enough confirmation — stay in current regime
        return current if current else raw_regime

    # ── Signal Assessment ────────────────────────────────────

    def _assess_volatility(self, matrix: CurveMatrix) -> float:
        """
        Volatility percentile (0-1).
        High values = elevated vol.
        """
        vols = matrix.volatilities
        if vols.empty:
            return 0.5

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

        try:
            if isinstance(corr.index, pd.MultiIndex):
                last_date = corr.index.get_level_values(0).unique()[-1]
                recent_corr = corr.loc[last_date]
            else:
                recent_corr = corr

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
        """Aggregate liquidity index (0-1, 1=abundant)."""
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

        negative_count = 0
        total = 0
        for sym, df in curves.items():
            if "Close" in df.columns and len(df) >= 5:
                recent_ret = df["Close"].pct_change().tail(5).sum()
                total += 1
                if recent_ret < -0.005:
                    negative_count += 1

        if total == 0:
            return 0.3

        return min(1.0, negative_count / total)

    # ── Heuristic Classification ─────────────────────────────

    def _classify_heuristic(
        self,
        vol: float,
        corr: float,
        liq: float,
        funding: float,
        stress: float,
    ) -> MarketRegime:
        """
        Multi-factor heuristic regime classification.
        Priority: Crisis > Structural > Liquidity > Volatile > Stable
        """
        crisis_score = (
            0.25 * (1 if vol > 0.9 else 0) +
            0.25 * (1 if corr > settings.correlation_expansion_threshold else 0) +
            0.25 * (1 if liq < 0.2 else 0) +
            0.15 * (1 if funding > 0.7 else 0) +
            0.10 * (1 if stress > 0.7 else 0)
        )
        if crisis_score >= 0.5:
            return MarketRegime.CRISIS

        if vol > 0.8 and (corr > 0.7 or corr < 0.15):
            return MarketRegime.STRUCTURAL_SHIFT

        if liq < settings.liquidity_contraction_threshold or funding > 0.5:
            return MarketRegime.LIQUIDITY_TIGHTENING

        if vol > 0.6:
            return MarketRegime.VOLATILE_MEAN_REVERTING

        return MarketRegime.STABLE_MEAN_REVERTING

    def _compute_confidence(self, vol: float, corr: float, liq: float, funding: float) -> float:
        """Confidence in regime classification (0-1)."""
        signals = [vol, corr, 1 - liq, funding]
        std = np.std(signals)
        confidence = max(0.3, 1.0 - std * 2)
        return round(confidence, 4)


# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[RegimeClassificationEngine] = None


def get_regime_engine() -> RegimeClassificationEngine:
    global _engine
    if _engine is None:
        _engine = RegimeClassificationEngine()
    return _engine
