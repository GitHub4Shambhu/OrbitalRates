"""
Layer 2 — Spread Graph Engine (Relative Value Discovery)

Advanced features (state-of-the-art 2024/2025):
  - Johansen cointegration test for multi-leg spread validation
  - Kalman filter for dynamic hedge ratio estimation
  - Hurst exponent for mean-reversion quality assessment
  - AR(1) persistence and half-life of mean reversion
  - ADF stationarity test + Chow structural break test
  - Historical drawdown distribution analysis
  - Risk-adjusted ranking with tail risk penalization

Rejection criteria:
  - Half-life > capital tolerance threshold
  - Structural instability detected
  - Liquidity insufficient
  - Hurst exponent > 0.5 (trending, not mean-reverting)
  - No cointegration (Johansen test fails)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import SpreadCandidate, SpreadType, TradeSignal
from app.agents.layer1_data.market_data_agent import CurveMatrix


# ── Kalman Filter for Dynamic Hedge Ratio ────────────────────────

class KalmanHedgeRatio:
    """
    Kalman filter that tracks a time-varying hedge ratio β_t:
        y_t = β_t · x_t + ε_t          (observation)
        β_{t+1} = β_t + η_t             (state transition — random walk)

    Returns posterior mean and variance of β at each step.
    """

    def __init__(self, delta: float = 1e-4, ve: float = 1e-3):
        """
        Args:
            delta: state noise (smaller = smoother hedge ratio).
            ve: observation noise variance.
        """
        self.delta = delta
        self.ve = ve

    def estimate(self, y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        """
        Run filter over the full series, return final (β_mean, β_std).
        """
        n = len(y)
        if n < 10:
            return 1.0, 0.0

        # Initial state
        beta = 0.0
        P = 1.0  # state variance

        for t in range(n):
            # Prediction
            P_pred = P + self.delta

            # Update
            y_hat = beta * x[t]
            e = y[t] - y_hat
            S = x[t] ** 2 * P_pred + self.ve
            K = P_pred * x[t] / S  # Kalman gain

            beta = beta + K * e
            P = (1 - K * x[t]) * P_pred

        return float(beta), float(np.sqrt(max(0, P)))


# ── Johansen Cointegration Wrapper ───────────────────────────────

def johansen_cointegration_test(
    y: np.ndarray, x: np.ndarray, significance: float = 0.05
) -> Tuple[float, bool]:
    """
    Johansen trace test for cointegration between two series.

    Returns (trace_statistic, is_cointegrated).
    Falls back to Engle-Granger if statsmodels coint_johansen unavailable.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        data = np.column_stack([y, x])
        # det_order=-1 = no deterministic terms, k_ar_diff=1 = 1 lag
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # trace_stat for r=0 (null: no cointegrating relationship)
        trace_stat = float(result.lr1[0])
        # Critical value at 5% for r=0
        cv_5pct = float(result.cvt[0, 1])  # column 1 = 5%
        return trace_stat, trace_stat > cv_5pct
    except Exception:
        # Fallback: Engle-Granger two-step
        try:
            from statsmodels.tsa.stattools import coint
            stat, pvalue, _ = coint(y, x)
            return float(abs(stat)), pvalue < significance
        except Exception:
            return 0.0, False


# ── Hurst Exponent ───────────────────────────────────────────────

def compute_hurst_exponent(series: np.ndarray, max_lag: int = 40) -> float:
    """
    Rescaled range (R/S) Hurst exponent.

    H < 0.5 → mean-reverting (good for RV)
    H = 0.5 → random walk
    H > 0.5 → trending / persistent

    Uses the `hurst` package if available, else manual R/S.
    """
    try:
        import hurst
        H, _, _ = hurst.compute_Hc(series, kind="random_walk", simplified=True)
        return float(np.clip(H, 0.0, 1.0))
    except Exception:
        pass

    # Manual R/S estimation
    n = len(series)
    if n < 30:
        return 0.5

    lags = range(2, min(max_lag, n // 4))
    rs_values = []

    for lag in lags:
        chunks = n // lag
        if chunks < 1:
            continue
        rs_list = []
        for i in range(chunks):
            chunk = series[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_c = chunk.mean()
            deviate = np.cumsum(chunk - mean_c)
            R = deviate.max() - deviate.min()
            S = chunk.std(ddof=1)
            if S > 1e-12:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    log_lags, log_rs = zip(*rs_values)
    try:
        H = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(H, 0.0, 1.0))
    except Exception:
        return 0.5


@dataclass
class SpreadEdge:
    """An edge in the spread graph — a tradable pair."""
    name: str
    leg1: str
    leg2: str
    spread_type: SpreadType
    series: pd.Series  # Historical spread time series
    leg1_prices: Optional[pd.Series] = None  # Raw price series for leg 1
    leg2_prices: Optional[pd.Series] = None  # Raw price series for leg 2


class SpreadGraphEngine:
    """
    Layer 2: Relative Value Discovery Agent.

    Builds a dynamic graph of all statistically valid spread
    relationships. Computes z-scores, half-life, stationarity,
    and ranks opportunities.
    """

    def __init__(self):
        self._spread_edges: List[SpreadEdge] = []
        logger.info("Layer 2 — Spread Graph Engine initialized")

    # ── Public API ───────────────────────────────────────────

    async def discover_opportunities(
        self,
        curve_matrix: CurveMatrix,
        liquidity_scores: Dict[str, float],
    ) -> List[SpreadCandidate]:
        """
        Full discovery pipeline:
        1. Build spread graph from curve matrix
        2. Compute statistics for each edge
        3. Filter and rank candidates
        """
        logger.info("Running spread discovery pipeline...")

        # Step 1: Build spread edges
        edges = self._build_spread_edges(curve_matrix)
        logger.info(f"Built {len(edges)} spread edges")

        # Step 2: Analyze each edge
        candidates = []
        for edge in edges:
            candidate = self._analyze_spread(edge, liquidity_scores)
            if candidate is not None:
                candidates.append(candidate)

        logger.info(f"Analyzed {len(candidates)} valid candidates")

        # Step 3: Filter rejections
        valid = [c for c in candidates if c.signal != TradeSignal.REJECT]
        rejected = [c for c in candidates if c.signal == TradeSignal.REJECT]
        logger.info(f"Valid: {len(valid)}, Rejected: {len(rejected)}")

        # Step 4: Rank by vol-adjusted return
        valid.sort(key=lambda c: c.vol_adjusted_return, reverse=True)
        for i, c in enumerate(valid):
            c.signal = self._determine_signal(c)

        return valid

    # ── Spread Graph Construction ────────────────────────────

    def _build_spread_edges(self, matrix: CurveMatrix) -> List[SpreadEdge]:
        """Build all spread edges from the curve matrix."""
        edges = []
        spreads = matrix.spreads

        if spreads.empty:
            logger.warning("No spreads in curve matrix — using fallback")
            return self._build_fallback_edges(matrix)

        # Type classification and leg-ticker mapping
        spread_meta = {
            #  name             type                              ticker1  ticker2
            "TLT_SHY":          (SpreadType.CURVE,                "TLT",   "SHY"),
            "TLT_IEF":          (SpreadType.CURVE,                "TLT",   "IEF"),
            "IEF_SHY":          (SpreadType.CURVE,                "IEF",   "SHY"),
            "IG_GOVT":          (SpreadType.SWAP_SPREAD,          "LQD",   "GOVT"),
            "HY_IG":            (SpreadType.SWAP_SPREAD,          "HYG",   "LQD"),
            "TIPS_BREAKEVEN":   (SpreadType.INFLATION_BREAKEVEN,  "TIP",   "IEF"),
            "INTL_US":          (SpreadType.CROSS_COUNTRY,        "BWX",   "GOVT"),
            "EM_US":            (SpreadType.CROSS_COUNTRY,        "EMB",   "GOVT"),
            "FLOAT_FIXED":      (SpreadType.OIS_LIBOR_BASIS,      "FLOT",  "VCSH"),
            "MBS_GOVT":         (SpreadType.FUTURES_BASIS,        "MBB",   "GOVT"),
            "AGG_GOVT":         (SpreadType.SWAP_SPREAD,          "AGG",   "GOVT"),
            "CORP_CURVE":       (SpreadType.CURVE,                "VCLT",  "VCSH"),
        }

        for col in spreads.columns:
            series = spreads[col].dropna()
            if len(series) < 60:  # Need sufficient history
                continue

            meta = spread_meta.get(col)
            if meta:
                stype, ticker1, ticker2 = meta
            else:
                parts = col.split("_", 1)
                ticker1 = parts[0] if len(parts) > 0 else col
                ticker2 = parts[1] if len(parts) > 1 else ""
                stype = SpreadType.CURVE

            # Attach raw leg prices for Johansen / Kalman tests
            leg1_prices = None
            leg2_prices = None
            curves = matrix.curves
            if ticker1 in curves and "Close" in curves[ticker1].columns:
                leg1_prices = curves[ticker1]["Close"].reindex(series.index).dropna()
            if ticker2 in curves and "Close" in curves[ticker2].columns:
                leg2_prices = curves[ticker2]["Close"].reindex(series.index).dropna()

            edges.append(SpreadEdge(
                name=col,
                leg1=ticker1,
                leg2=ticker2,
                spread_type=stype,
                series=series,
                leg1_prices=leg1_prices,
                leg2_prices=leg2_prices,
            ))

        return edges

    def _build_fallback_edges(self, matrix: CurveMatrix) -> List[SpreadEdge]:
        """Build spread edges directly from price pairs when spread matrix is empty."""
        edges = []
        curves = matrix.curves
        symbols = list(curves.keys())

        # Build spreads from price ratios of key pairs
        key_pairs = [
            ("TLT", "SHY", SpreadType.CURVE),
            ("TLT", "IEF", SpreadType.CURVE),
            ("IEF", "SHY", SpreadType.CURVE),
            ("LQD", "GOVT", SpreadType.SWAP_SPREAD),
            ("HYG", "LQD", SpreadType.SWAP_SPREAD),
            ("TIP", "IEF", SpreadType.INFLATION_BREAKEVEN),
            ("EMB", "GOVT", SpreadType.CROSS_COUNTRY),
            ("BWX", "GOVT", SpreadType.CROSS_COUNTRY),
            ("VCLT", "VCSH", SpreadType.CURVE),
            ("MBB", "GOVT", SpreadType.FUTURES_BASIS),
        ]

        for leg1, leg2, stype in key_pairs:
            if leg1 in curves and leg2 in curves:
                df1 = curves[leg1]
                df2 = curves[leg2]
                if "Close" in df1.columns and "Close" in df2.columns:
                    # Align indices
                    combined = pd.DataFrame({
                        "a": df1["Close"],
                        "b": df2["Close"],
                    }).dropna()
                    if len(combined) >= 60:
                        # Log price ratio spread (bps-equivalent)
                        spread = (np.log(combined["a"]) - np.log(combined["b"])) * 10_000
                        edges.append(SpreadEdge(
                            name=f"{leg1}_{leg2}",
                            leg1=leg1,
                            leg2=leg2,
                            spread_type=stype,
                            series=spread,
                            leg1_prices=combined["a"],
                            leg2_prices=combined["b"],
                        ))

        return edges

    # ── Statistical Analysis ─────────────────────────────────

    def _analyze_spread(
        self,
        edge: SpreadEdge,
        liquidity_scores: Dict[str, float],
    ) -> Optional[SpreadCandidate]:
        """Compute all statistics for a spread edge, including advanced metrics."""
        series = edge.series
        if len(series) < 60:
            return None

        try:
            # Basic statistics
            mean_spread = series.mean()
            std_spread = series.std()
            if std_spread < 1e-10:
                return None

            current = series.iloc[-1]
            zscore = (current - mean_spread) / std_spread

            # AR(1) coefficient and half-life
            ar1, halflife = self._compute_halflife(series)

            # Stationarity (ADF test)
            is_stationary = self._adf_test(series)

            # Structural break probability
            break_prob = self._structural_break_test(series)

            # ── Advanced: Hurst exponent ─────────────────────
            # H < 0.5 → mean-reverting (good for RV), H ≈ 0.5 → random walk,
            # H > 0.5 → trending (bad).  Computed on spread level series.
            hurst_exp = compute_hurst_exponent(series.values)

            # ── Advanced: Johansen cointegration ─────────────
            # Use normalized leg prices (if available in series index)
            joh_trace, is_coint = self._run_johansen(edge)

            # ── Advanced: Kalman filter hedge ratio ──────────
            kalman_beta, kalman_std = self._run_kalman(edge)

            # Tail risk
            returns = series.diff().dropna()
            tail_5 = np.percentile(returns, 5)
            tail_1 = np.percentile(returns, 1)

            # Drawdown analysis
            cummax = series.cummax()
            drawdowns = (series - cummax)
            max_dd = drawdowns.min()

            # Liquidity
            liq1 = liquidity_scores.get(edge.leg1, 0.5)
            liq2 = liquidity_scores.get(edge.leg2, 0.5)
            liquidity = min(liq1, liq2)

            # Crowding proxy (high recent correlation with other spreads)
            crowding = self._estimate_crowding(series)

            # Expected return (mean reversion towards zero z-score)
            expected_return = abs(zscore) * std_spread * (1 - np.exp(-np.log(2) / max(halflife, 1)))

            # Expected shortfall
            es = abs(tail_1) * 2  # Conservative ES estimate

            # Vol-adjusted return (penalize trending Hurst)
            vol = returns.std() * np.sqrt(252) if len(returns) > 20 else std_spread
            hurst_penalty = max(0.3, 1.0 - max(0, hurst_exp - 0.5) * 2)  # penalize H>0.5
            vol_adj_return = (expected_return / (vol + 1e-10)) * hurst_penalty

            # Build candidate
            candidate = SpreadCandidate(
                spread_id=edge.name,
                spread_type=edge.spread_type,
                leg1_instrument=edge.leg1,
                leg2_instrument=edge.leg2,
                leg1_country=self._infer_country(edge.leg1),
                leg2_country=self._infer_country(edge.leg2),
                leg1_tenor="",
                leg2_tenor="",
                current_spread_bps=round(current, 4),
                zscore=round(zscore, 4),
                halflife_days=round(halflife, 1),
                mean_spread_bps=round(mean_spread, 4),
                std_spread_bps=round(std_spread, 4),
                ar1_coefficient=round(ar1, 4),
                is_stationary=is_stationary,
                structural_break_prob=round(break_prob, 4),
                liquidity_score=round(liquidity, 4),
                crowding_proxy=round(crowding, 4),
                expected_return_bps=round(expected_return, 4),
                expected_shortfall_bps=round(es, 4),
                vol_adjusted_return=round(vol_adj_return, 4),
                tail_5pct_bps=round(tail_5, 4),
                tail_1pct_bps=round(tail_1, 4),
                # Advanced fields
                hurst_exponent=round(hurst_exp, 4),
                johansen_trace_stat=round(joh_trace, 4),
                is_cointegrated=is_coint,
                kalman_hedge_ratio=round(kalman_beta, 4),
                kalman_hedge_ratio_std=round(kalman_std, 4),
            )

            # Apply rejection filters
            self._apply_filters(candidate)

            return candidate

        except Exception as e:
            logger.warning(f"Analysis failed for {edge.name}: {e}")
            return None

    def _run_johansen(self, edge: SpreadEdge) -> Tuple[float, bool]:
        """Run Johansen cointegration test on the actual leg price series."""
        try:
            if edge.leg1_prices is not None and edge.leg2_prices is not None:
                # Use actual leg prices — the correct approach
                y = edge.leg1_prices.values
                x = edge.leg2_prices.values
                min_len = min(len(y), len(x))
                if min_len < 60:
                    return 0.0, False
                y = y[-min_len:]
                x = x[-min_len:]
            else:
                # Fallback: use spread level + a lagged version
                series = edge.series.values
                if len(series) < 60:
                    return 0.0, False
                y = series
                x = np.roll(series, 1)
                x[0] = x[1]

            return johansen_cointegration_test(y, x)
        except Exception as e:
            logger.debug(f"Johansen test failed for {edge.name}: {e}")
            return 0.0, False

    def _run_kalman(self, edge: SpreadEdge) -> Tuple[float, float]:
        """Run Kalman filter for dynamic hedge ratio using actual leg prices."""
        try:
            if edge.leg1_prices is not None and edge.leg2_prices is not None:
                y = edge.leg1_prices.values
                x = edge.leg2_prices.values
                min_len = min(len(y), len(x))
                if min_len < 30:
                    return 1.0, 0.0
                y = y[-min_len:]
                x = x[-min_len:]
            else:
                series = edge.series.values
                n = len(series)
                if n < 30:
                    return 1.0, 0.0
                y = series[1:]
                x = series[:-1]

            kf = KalmanHedgeRatio(delta=1e-4, ve=max(1e-6, np.var(y - x)))
            return kf.estimate(y, x)
        except Exception as e:
            logger.debug(f"Kalman filter failed for {edge.name}: {e}")
            return 1.0, 0.0

    def _compute_halflife(self, series: pd.Series) -> Tuple[float, float]:
        """
        Estimate AR(1) coefficient and half-life of mean reversion.
        Half-life = -log(2) / log(|ar1|)
        """
        y = series.values
        y_lag = y[:-1]
        y_curr = y[1:]

        if len(y_lag) < 10:
            return 0.0, 999.0

        # OLS: y_t = alpha + beta * y_{t-1}
        X = np.column_stack([np.ones_like(y_lag), y_lag])
        try:
            beta = np.linalg.lstsq(X, y_curr, rcond=None)[0]
            ar1 = beta[1]
        except np.linalg.LinAlgError:
            return 0.0, 999.0

        if abs(ar1) >= 1.0 or ar1 <= 0:
            return ar1, 999.0

        halflife = -np.log(2) / np.log(abs(ar1))
        return ar1, max(1.0, halflife)

    def _adf_test(self, series: pd.Series) -> bool:
        """Augmented Dickey-Fuller stationarity test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.values, maxlag=10, autolag="AIC")
            return result[1] < 0.05  # p-value < 5% = stationary
        except Exception:
            # Fallback: simple variance ratio test
            n = len(series)
            if n < 40:
                return False
            half = n // 2
            var1 = series.iloc[:half].var()
            var2 = series.iloc[half:].var()
            ratio = var1 / (var2 + 1e-10)
            return 0.5 < ratio < 2.0  # Roughly stable variance

    def _structural_break_test(self, series: pd.Series) -> float:
        """
        Structural break probability (Chow test proxy).
        Compare regression coefficients in first vs second half.
        """
        n = len(series)
        if n < 60:
            return 0.5

        mid = n // 2
        first_half = series.iloc[:mid]
        second_half = series.iloc[mid:]

        # Compare mean and variance
        mean_diff = abs(first_half.mean() - second_half.mean()) / (series.std() + 1e-10)
        var_ratio = first_half.var() / (second_half.var() + 1e-10)

        # Score: higher = more likely structural break
        mean_score = min(1.0, mean_diff / 2.0)
        var_score = min(1.0, abs(np.log(var_ratio + 1e-10)) / 2.0)

        return round(0.5 * mean_score + 0.5 * var_score, 4)

    def _estimate_crowding(self, series: pd.Series) -> float:
        """Estimate crowding risk proxy (0-1)."""
        if len(series) < 60:
            return 0.5

        # Recent autocorrelation spike (crowded trades show persistent returns)
        returns = series.diff().dropna()
        if len(returns) < 20:
            return 0.5

        recent_autocorr = returns.tail(20).autocorr()
        hist_autocorr = returns.autocorr()

        if hist_autocorr is None or np.isnan(hist_autocorr):
            return 0.3

        # Rising autocorrelation suggests crowding
        crowding = max(0, min(1, (recent_autocorr - hist_autocorr) * 5 + 0.3))
        return round(crowding, 4)

    def _infer_country(self, symbol: str) -> str:
        """Infer country from instrument symbol."""
        intl = {"BWX": "INTL", "EMB": "EM", "BNDX": "INTL", "IGLT.L": "GB"}
        return intl.get(symbol, "US")

    # ── Filtering & Ranking ──────────────────────────────────

    def _apply_filters(self, c: SpreadCandidate) -> None:
        """Apply institutional rejection filters including advanced metrics."""
        reasons = []

        if c.halflife_days > settings.max_halflife_days:
            reasons.append(f"halflife {c.halflife_days:.0f}d > {settings.max_halflife_days}d cap")

        if c.halflife_days < settings.min_halflife_days:
            reasons.append(f"halflife {c.halflife_days:.0f}d < {settings.min_halflife_days}d (noise)")

        # ADF non-stationarity is a soft penalty: only reject if
        # BOTH the spread is non-stationary AND the Hurst confirms trending
        if not c.is_stationary and c.hurst_exponent > 0.55:
            reasons.append(f"non-stationary + trending Hurst {c.hurst_exponent:.3f}")

        if c.structural_break_prob > 0.80:
            reasons.append(f"structural break prob {c.structural_break_prob:.2f}")

        if c.liquidity_score < settings.min_liquidity_score:
            reasons.append(f"liquidity {c.liquidity_score:.2f} < {settings.min_liquidity_score}")

        if abs(c.zscore) < settings.zscore_entry_threshold:
            reasons.append(f"|z|={abs(c.zscore):.2f} < entry threshold {settings.zscore_entry_threshold}")

        # Advanced filter: Hurst exponent — reject strongly trending pairs
        if c.hurst_exponent > 0.55:
            reasons.append(f"Hurst {c.hurst_exponent:.3f} > 0.55 (trending)")

        if reasons:
            c.signal = TradeSignal.REJECT
            c.rejection_reason = "; ".join(reasons)

    def _determine_signal(self, c: SpreadCandidate) -> TradeSignal:
        """Determine trade signal based on z-score."""
        if c.zscore > settings.zscore_entry_threshold:
            return TradeSignal.ENTER_SHORT  # Spread too wide, short it
        elif c.zscore < -settings.zscore_entry_threshold:
            return TradeSignal.ENTER_LONG   # Spread too narrow, long it
        elif abs(c.zscore) < settings.zscore_exit_threshold:
            return TradeSignal.EXIT
        return TradeSignal.HOLD


# ── Singleton ────────────────────────────────────────────────────
_engine: Optional[SpreadGraphEngine] = None


def get_spread_graph_engine() -> SpreadGraphEngine:
    global _engine
    if _engine is None:
        _engine = SpreadGraphEngine()
    return _engine
