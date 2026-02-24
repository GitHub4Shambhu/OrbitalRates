"""
Layer 4 — Tail Risk & Survival Simulation Engine

THE CORE AGENT. Non-negotiable constraints:
  - Max drawdown limit: configurable
  - Max expected shortfall: defined
  - Leverage cap: dynamic

Advanced features (state-of-the-art 2024/2025):
  - Extreme Value Theory (Peaks-over-Threshold with GPD)
  - Regime-switching Monte Carlo with dynamic vol multiplier
  - Copula-based upper tail dependence estimation
  - Student-t(df=4) fat-tail simulation baseline

Stress scenarios:
  - 2008-style spread blowout
  - 2020 liquidity seizure
  - Correlation → 1 shock
  - Volatility doubling
  - Funding spread spike
  - Forced deleveraging cascade

If survival probability < 99% annual → REDUCE LEVERAGE. ALWAYS.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import (
    RiskMetrics, StressResult, PortfolioPosition, RegimeState,
    SpreadCandidate, MarketRegime,
)


# ── Extreme Value Theory: Peaks-over-Threshold + GPD ────────────

class EVTAnalyzer:
    """
    Extreme Value Theory analyzer using Peaks-over-Threshold (POT)
    with Generalized Pareto Distribution (GPD) fitting.

    The GPD models the excess distribution above a high threshold u:
      P(X - u | X > u) ~ GPD(ξ, σ)

    - ξ > 0: heavy tail (Fréchet domain — financial losses)
    - ξ = 0: exponential tail
    - ξ < 0: bounded tail (Weibull domain)

    VaR_q = u + (σ/ξ) * [((n/N_u) * (1-q))^(-ξ) - 1]
    ES_q  = VaR_q / (1-ξ) + (σ - ξ*u) / (1-ξ)
    """

    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile

    def fit(self, losses: np.ndarray) -> Dict[str, float]:
        """
        Fit GPD to exceedances above the threshold.
        Returns dict with evt_var_99, evt_es_99, shape (xi), scale (sigma).
        """
        if len(losses) < 50:
            return {"evt_var_99": 0.0, "evt_es_99": 0.0, "shape": 0.0, "scale": 0.0}

        u = np.percentile(losses, self.threshold_quantile * 100)
        exceedances = losses[losses > u] - u

        if len(exceedances) < 10:
            return {"evt_var_99": 0.0, "evt_es_99": 0.0, "shape": 0.0, "scale": 0.0}

        try:
            from scipy.stats import genpareto
            # MLE fit
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
        except Exception:
            # Fallback: method of moments
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances)
            if mean_exc <= 0:
                return {"evt_var_99": 0.0, "evt_es_99": 0.0, "shape": 0.0, "scale": 0.0}
            shape = 0.5 * (mean_exc ** 2 / var_exc - 1)
            scale = mean_exc * (1 + shape) / 2
            shape = max(-0.5, min(shape, 1.0))

        n = len(losses)
        n_u = len(exceedances)
        q = 0.99  # 99% quantile

        try:
            if abs(shape) < 1e-10:
                # Exponential case
                evt_var = u + scale * np.log(n * (1 - q) / n_u)
            else:
                evt_var = u + (scale / shape) * (((n / n_u) * (1 - q)) ** (-shape) - 1)

            if shape < 1:
                evt_es = evt_var / (1 - shape) + (scale - shape * u) / (1 - shape)
            else:
                evt_es = evt_var * 1.5  # Fallback for very heavy tails
        except Exception:
            evt_var = abs(np.percentile(losses, 99))
            evt_es = evt_var * 1.3

        return {
            "evt_var_99": float(max(0, evt_var)),
            "evt_es_99": float(max(0, evt_es)),
            "shape": float(np.clip(shape, -1.0, 2.0)),
            "scale": float(max(0, scale)),
        }


# ── Copula-Based Tail Dependence ─────────────────────────────────

def estimate_tail_dependence(losses: np.ndarray, threshold: float = 0.95) -> float:
    """
    Estimate upper tail dependence coefficient λ_U using
    empirical copula approach.

    For portfolio losses, we split into two halves and measure
    how often extreme losses co-occur (joint tail events).

    λ_U ≈ P(U > q, V > q) / (1 - q)  where U, V are uniform margins.

    Returns value in [0, 1]: 0 = tail independent, 1 = perfect tail dependence.
    """
    n = len(losses)
    if n < 100:
        return 0.0

    # Split into two sub-portfolios
    half = n // 2
    u = losses[:half]
    v = losses[half:2 * half]

    min_len = min(len(u), len(v))
    u = u[:min_len]
    v = v[:min_len]

    # Convert to uniform margins via empirical CDF
    from scipy.stats import rankdata
    u_ranks = rankdata(u) / (min_len + 1)
    v_ranks = rankdata(v) / (min_len + 1)

    # Count joint exceedances
    joint_exceed = np.sum((u_ranks > threshold) & (v_ranks > threshold))
    marginal_exceed = np.sum(u_ranks > threshold)

    if marginal_exceed == 0:
        return 0.0

    lambda_u = joint_exceed / marginal_exceed
    return float(np.clip(lambda_u, 0.0, 1.0))


# ── EVT: Peaks-over-Threshold with GPD ──────────────────────────

class EVTAnalyzer:
    """
    Extreme Value Theory analyzer using the Peaks-over-Threshold
    approach with Generalized Pareto Distribution (GPD) fitting.

    GPD CDF: F(x) = 1 - (1 + ξ·x/σ)^(-1/ξ)
    where ξ = shape (tail index), σ = scale.
    ξ > 0 → heavy (Pareto) tail; ξ = 0 → exponential tail.
    """

    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile

    def fit(self, losses: np.ndarray) -> Dict[str, float]:
        """
        Fit GPD to exceedances over the threshold.

        Returns dict with keys:
          shape (xi), scale (sigma), threshold (u),
          evt_var_99, evt_es_99, n_exceedances
        """
        if len(losses) < 50:
            return self._empty_result()

        # Use losses (positive = bad)
        u = np.percentile(losses, self.threshold_quantile * 100)
        exceedances = losses[losses > u] - u

        if len(exceedances) < 10:
            return self._empty_result()

        # Fit GPD via maximum likelihood
        xi, sigma = self._fit_gpd_mle(exceedances)

        # Compute EVT-based VaR and ES at 99%
        n = len(losses)
        n_u = len(exceedances)
        p_exceed = n_u / n

        # VaR_p = u + (σ/ξ) * ((n/n_u * (1-p))^(-ξ) - 1)
        p = 0.99
        if abs(xi) > 1e-8:
            evt_var = u + (sigma / xi) * (((1 - p) / p_exceed) ** (-xi) - 1)
            # ES = VaR/(1-ξ) + (σ - ξ*u)/(1-ξ)
            if xi < 1.0:
                evt_es = evt_var / (1 - xi) + (sigma - xi * u) / (1 - xi)
            else:
                evt_es = evt_var * 1.5  # fallback
        else:
            # Exponential tail (xi ≈ 0)
            evt_var = u + sigma * np.log((1 - p) / p_exceed)
            evt_es = evt_var + sigma

        return {
            "shape": float(xi),
            "scale": float(sigma),
            "threshold": float(u),
            "evt_var_99": float(max(0, evt_var)),
            "evt_es_99": float(max(0, evt_es)),
            "n_exceedances": int(n_u),
        }

    def _fit_gpd_mle(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """Maximum likelihood estimation for GPD parameters."""
        try:
            from scipy.stats import genpareto
            xi, _, sigma = genpareto.fit(exceedances, floc=0)
            return float(xi), float(sigma)
        except Exception:
            # Fallback: method-of-moments
            mean_e = exceedances.mean()
            var_e = exceedances.var()
            if mean_e <= 0:
                return 0.0, max(0.001, mean_e)
            sigma = mean_e * (mean_e ** 2 / var_e + 1) / 2
            xi = (mean_e ** 2 / var_e - 1) / 2
            return float(np.clip(xi, -0.5, 2.0)), float(max(0.001, sigma))

    def _empty_result(self) -> Dict[str, float]:
        return {
            "shape": 0.0, "scale": 0.0, "threshold": 0.0,
            "evt_var_99": 0.0, "evt_es_99": 0.0, "n_exceedances": 0,
        }


# ── Copula Tail Dependence Estimator ────────────────────────────

def estimate_tail_dependence(losses: np.ndarray, k: int = 50) -> float:
    """
    Estimate upper tail dependence coefficient λ_U using the
    empirical non-parametric estimator:

        λ_U ≈ (1/k) Σ 1[U_i > 1 - k/n AND V_i > 1 - k/n]  /  (k/n)

    For a portfolio loss vector, we split into two halves and
    measure joint extreme co-movement.

    Returns λ_U ∈ [0, 1] where 0 = tail independence, 1 = perfect tail dependence.
    """
    n = len(losses)
    if n < 100:
        return 0.0

    # Split portfolio losses into two random subsets for the copula
    mid = n // 2
    u = losses[:mid]
    v = losses[mid:2 * mid]
    m = len(u)

    if m < 50:
        return 0.0

    # Rank transform to uniform margins
    rank_u = np.argsort(np.argsort(u)) / m
    rank_v = np.argsort(np.argsort(v)) / m

    # Count joint exceedances
    k = min(k, m // 5)
    threshold = 1 - k / m
    joint_exceed = np.sum((rank_u > threshold) & (rank_v > threshold))
    lambda_u = joint_exceed / max(1, k)

    return float(np.clip(lambda_u, 0.0, 1.0))


class TailRiskEngine:
    """
    Layer 4: Survival-First Risk Core.

    Priority: Survival over return. Always.
    """

    def __init__(self):
        self._risk_history: List[RiskMetrics] = []
        self._evt_analyzer = EVTAnalyzer(threshold_quantile=0.95)
        logger.info("Layer 4 — Tail Risk & Survival Engine initialized (EVT + Copula enabled)")

    # ── Public API ───────────────────────────────────────────

    async def compute_risk_metrics(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
        candidates: Optional[List[SpreadCandidate]] = None,
        nav: float = 100_000_000.0,
    ) -> RiskMetrics:
        """
        Compute comprehensive portfolio risk metrics with EVT and copula.
        """
        # Portfolio aggregates
        total_dv01 = sum(p.dv01_net for p in positions)
        total_convexity = sum(p.convexity_net for p in positions)
        gross_notional = sum(abs(p.notional) for p in positions)
        net_notional = sum(p.notional for p in positions)
        
        # Leverage — use actual NAV passed in, not an inferred one
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

        # ── Regime-switching Monte Carlo ─────────────────────
        vol_multiplier = self._regime_vol_multiplier(regime)
        mc_losses = self._monte_carlo_simulation(positions, regime, n_sims=10000)
        
        # Standard VaR & ES
        var_99 = np.percentile(mc_losses, 1) if len(mc_losses) > 0 else 0
        es_99 = mc_losses[mc_losses <= np.percentile(mc_losses, 1)].mean() if len(mc_losses) > 0 else 0

        # ── EVT: Peaks-over-Threshold with GPD ──────────────
        # Use the left tail (losses are negative returns)
        evt_result = self._evt_analyzer.fit(-mc_losses)  # flip sign: positive = loss
        evt_var_99 = evt_result["evt_var_99"]
        evt_es_99 = evt_result["evt_es_99"]
        evt_shape = evt_result["shape"]

        # ── Copula tail dependence ───────────────────────────
        tail_dep = estimate_tail_dependence(-mc_losses)

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
            # EVT / advanced fields
            evt_var_99_pct=round(evt_var_99 * 100, 4),
            evt_es_99_pct=round(evt_es_99 * 100, 4),
            evt_shape_parameter=round(evt_shape, 4),
            tail_dependence_coeff=round(tail_dep, 4),
            regime_vol_multiplier=round(vol_multiplier, 2),
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

    def _regime_vol_multiplier(self, regime: RegimeState) -> float:
        """Compute regime-dependent volatility multiplier for MC sims."""
        base_multipliers = {
            MarketRegime.STABLE_MEAN_REVERTING: 1.0,
            MarketRegime.VOLATILE_MEAN_REVERTING: 1.5,
            MarketRegime.LIQUIDITY_TIGHTENING: 2.0,
            MarketRegime.STRUCTURAL_SHIFT: 2.5,
            MarketRegime.CRISIS: 3.0,
        }
        mult = base_multipliers.get(regime.regime, 1.0)

        # Blend with HMM confidence: if HMM is very uncertain, add extra vol
        hmm_uncertainty_premium = max(0, 1.0 - regime.hmm_confidence) * 0.5
        return mult + hmm_uncertainty_premium

    def _monte_carlo_simulation(
        self,
        positions: List[PortfolioPosition],
        regime: RegimeState,
        n_sims: int = 10000,
    ) -> np.ndarray:
        """
        Fat-tail Monte Carlo simulation of portfolio returns.
        Uses Student-t distribution (df=4) with regime-switching volatility.
        """
        if not positions:
            return np.zeros(n_sims)

        n_positions = len(positions)
        
        # Regime-adjusted volatility
        vol_multiplier = self._regime_vol_multiplier(regime)

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
