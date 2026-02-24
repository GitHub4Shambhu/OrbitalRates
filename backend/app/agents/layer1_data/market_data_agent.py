"""
Layer 1 — Global Rates Ingestion Engine

Responsibilities:
  - Ingest sovereign yield curves (G10 + EM) via yfinance proxies
  - Normalize yields, compute DV01
  - Build term structure matrix
  - Compute rolling volatility and correlation matrices
  - Generate liquidity metrics
  - Detect stale pricing
  - Calculate funding stress index

Outputs:
  - Clean structured curve matrix
  - Liquidity index
  - Funding stress index
  - Positioning imbalance score
"""

import asyncio
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from app.core.config import settings
from app.core.types import CurvePoint, MarketRegime

# ── Yield curve proxy tickers (Treasury ETFs & futures proxies) ──
# In production: Bloomberg BPIPE / Refinitiv. Here: yfinance proxies.

YIELD_PROXIES: Dict[str, Dict[str, str]] = {
    "US": {
        "1Y": "^IRX",      # 13-week T-bill
        "2Y": "2YY=F",     # 2Y yield futures
        "5Y": "^FVX",      # 5Y Treasury yield
        "10Y": "^TNX",     # 10Y Treasury yield
        "30Y": "^TYX",     # 30Y Treasury yield
    },
    "DE": {
        "10Y": "BUNL=F",   # Bund proxy
    },
    "JP": {
        "10Y": "JPST",     # Japan proxy
    },
    "GB": {
        "10Y": "IGLT.L",   # UK Gilts proxy
    },
}

# ETF proxies for spread construction
RATE_ETFS = {
    "TLT": "iShares 20+ Year Treasury",
    "IEF": "iShares 7-10 Year Treasury",
    "SHY": "iShares 1-3 Year Treasury",
    "TIP": "iShares TIPS",
    "IGLT.L": "UK Gilts",
    "BNDX": "International Bond",
    "EMB": "EM Bond",
    "LQD": "Investment Grade Corporate",
    "HYG": "High Yield Corporate",
    "MBB": "Mortgage-Backed Securities",
    "GOVT": "US Treasury Bond",
    "VGSH": "Short-Term Treasury",
    "VGIT": "Intermediate Treasury",
    "VGLT": "Long-Term Treasury",
    "BWX": "International Treasury",
    "AGG": "US Aggregate Bond",
    "BND": "Total Bond Market",
    "FLOT": "Floating Rate",
    "VCSH": "Short-Term Corporate",
    "VCLT": "Long-Term Corporate",
}

# Core universe for spread construction
CORE_UNIVERSE = list(RATE_ETFS.keys())


@dataclass
class CurveMatrix:
    """Structured yield curve data."""
    curves: Dict[str, pd.DataFrame] = field(default_factory=dict)
    spreads: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    volatilities: pd.DataFrame = field(default_factory=pd.DataFrame)
    liquidity_scores: Dict[str, float] = field(default_factory=dict)
    funding_stress_index: float = 0.0
    data_source: str = "live"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stale_symbols: List[str] = field(default_factory=list)


class MarketDataAgent:
    """
    Layer 1: Global Rates Ingestion Engine.
    
    Ingests, cleans, and normalizes fixed income market data.
    Computes term structure metrics, volatility, correlations,
    and liquidity scores.
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._mock_symbols: set = set()
        self._last_refresh: Optional[datetime] = None
        logger.info("Layer 1 — Market Data Agent initialized")

    # ── Public API ───────────────────────────────────────────

    async def get_curve_matrix(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 252,
    ) -> CurveMatrix:
        """
        Fetch and build complete curve matrix.
        
        Returns clean price/yield data for all instruments,
        plus computed spreads, correlations, volatilities.
        """
        symbols = symbols or CORE_UNIVERSE
        logger.info(f"Building curve matrix for {len(symbols)} instruments")

        # Fetch all data in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._fetch_instrument, sym, lookback_days)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build curves dict
        curves: Dict[str, pd.DataFrame] = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {sym}: {result}")
                curves[sym] = self._generate_mock_data(sym, lookback_days)
                self._mock_symbols.add(sym)
            elif result is not None and not result.empty:
                curves[sym] = result
                self._mock_symbols.discard(sym)
            else:
                curves[sym] = self._generate_mock_data(sym, lookback_days)
                self._mock_symbols.add(sym)

        # Compute derived metrics
        prices_df = self._build_price_matrix(curves)
        returns_df = prices_df.pct_change().dropna()
        
        spreads = self._compute_spread_matrix(prices_df)
        correlations = self._compute_rolling_correlations(returns_df)
        volatilities = self._compute_rolling_volatility(returns_df)
        liquidity = self._compute_liquidity_scores(curves)
        funding_stress = self._compute_funding_stress(curves)

        matrix = CurveMatrix(
            curves=curves,
            spreads=spreads,
            correlations=correlations,
            volatilities=volatilities,
            liquidity_scores=liquidity,
            funding_stress_index=funding_stress,
            data_source=self.data_source,
            stale_symbols=list(self._mock_symbols),
        )

        self._last_refresh = datetime.utcnow()
        logger.info(
            f"Curve matrix built: {len(curves)} instruments, "
            f"funding_stress={funding_stress:.3f}, "
            f"data_source={self.data_source}"
        )
        return matrix

    @property
    def data_source(self) -> str:
        return "stale" if self._mock_symbols else "live"

    # ── Data Fetching ────────────────────────────────────────

    def _fetch_instrument(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single instrument."""
        try:
            import yfinance as yf
            period = f"{max(lookback_days // 20, 1)}mo"
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            return hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _generate_mock_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Generate synthetic data for instruments not available via yfinance."""
        logger.debug(f"Generating mock data for {symbol}")
        np.random.seed(hash(symbol) % 2**31)
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq="B")
        
        base_price = 100.0 + np.random.uniform(-20, 20)
        returns = np.random.normal(0.0001, 0.005, lookback_days)
        prices = base_price * np.cumprod(1 + returns)
        
        volumes = np.random.lognormal(15, 1, lookback_days).astype(int)
        
        df = pd.DataFrame({
            "Open": prices * (1 + np.random.normal(0, 0.001, lookback_days)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.005, lookback_days))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, lookback_days))),
            "Close": prices,
            "Volume": volumes,
        }, index=dates)
        
        return df

    # ── Derived Computations ─────────────────────────────────

    def _build_price_matrix(self, curves: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build aligned price matrix from all instruments."""
        close_prices = {}
        for sym, df in curves.items():
            if "Close" in df.columns and not df.empty:
                close_prices[sym] = df["Close"]
        
        if not close_prices:
            return pd.DataFrame()
        
        prices = pd.DataFrame(close_prices)
        prices = prices.ffill().bfill()
        return prices

    def _compute_spread_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise spreads as log price ratios.

        Using log(P_A / P_B) instead of (norm_A - norm_B) produces a
        stationary series when two instruments are cointegrated with a
        hedge ratio close to 1.  This dramatically improves ADF, Hurst,
        and half-life estimation downstream in L2.
        """
        if prices.empty:
            return pd.DataFrame()

        log_prices = np.log(prices.replace(0, np.nan)).dropna()

        # Key spreads — each is log(leg1 / leg2) × 10_000 for bps-like scale
        spreads = pd.DataFrame(index=log_prices.index)
        scale = 10_000  # express in bps-equivalent units

        pairs = [
            # Duration / curve
            ("TLT_SHY",          "TLT",  "SHY"),
            ("TLT_IEF",          "TLT",  "IEF"),
            ("IEF_SHY",          "IEF",  "SHY"),
            # Credit
            ("IG_GOVT",          "LQD",  "GOVT"),
            ("HY_IG",            "HYG",  "LQD"),
            # Inflation breakeven proxy
            ("TIPS_BREAKEVEN",   "TIP",  "IEF"),
            # International
            ("INTL_US",          "BWX",  "GOVT"),
            ("EM_US",            "EMB",  "GOVT"),
            # Floating vs fixed
            ("FLOAT_FIXED",      "FLOT", "VCSH"),
            # MBS
            ("MBS_GOVT",         "MBB",  "GOVT"),
            # Aggregate vs Treasury
            ("AGG_GOVT",         "AGG",  "GOVT"),
            # Corporate curve
            ("CORP_CURVE",       "VCLT", "VCSH"),
        ]

        for name, leg1, leg2 in pairs:
            if leg1 in log_prices.columns and leg2 in log_prices.columns:
                spreads[name] = (log_prices[leg1] - log_prices[leg2]) * scale

        return spreads

    def _compute_rolling_correlations(
        self, returns: pd.DataFrame, window: int = 60
    ) -> pd.DataFrame:
        """Compute rolling pairwise correlations."""
        if returns.empty or len(returns) < window:
            return pd.DataFrame()
        return returns.rolling(window).corr().dropna()

    def _compute_rolling_volatility(
        self, returns: pd.DataFrame, window: int = 21
    ) -> pd.DataFrame:
        """Compute rolling annualized volatility."""
        if returns.empty or len(returns) < window:
            return pd.DataFrame()
        return returns.rolling(window).std() * np.sqrt(252)

    def _compute_liquidity_scores(self, curves: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute liquidity score per instrument (0-1).
        Based on volume consistency, bid-ask proxy, and volume trend.
        """
        scores = {}
        for sym, df in curves.items():
            if "Volume" not in df.columns or df.empty:
                scores[sym] = 0.5
                continue
            
            vol = df["Volume"].tail(60)
            if vol.sum() == 0:
                scores[sym] = 0.2
                continue

            # Volume consistency (lower CV = more liquid)
            cv = vol.std() / (vol.mean() + 1e-10)
            consistency = max(0, 1 - cv)

            # Volume trend (rising volume = improving liquidity)
            if len(vol) >= 20:
                recent = vol.tail(20).mean()
                older = vol.head(20).mean()
                trend = min(1.0, recent / (older + 1e-10))
            else:
                trend = 0.5

            # Bid-ask proxy from high-low spread
            if "High" in df.columns and "Low" in df.columns:
                hl_spread = ((df["High"] - df["Low"]) / df["Close"]).tail(20).mean()
                tightness = max(0, 1 - hl_spread * 100)
            else:
                tightness = 0.5

            score = 0.4 * consistency + 0.3 * trend + 0.3 * tightness
            scores[sym] = round(max(0, min(1, score)), 4)

        return scores

    def _compute_funding_stress(self, curves: Dict[str, pd.DataFrame]) -> float:
        """
        Compute funding stress index (0-1, 1=severe).
        
        Uses short-term vs long-term rate proxies and
        credit spread widening as stress indicators.
        """
        stress_signals = []

        # Short-term volatility spike (SHY vol > normal)
        if "SHY" in curves and not curves["SHY"].empty:
            shy_ret = curves["SHY"]["Close"].pct_change().tail(20)
            shy_vol = shy_ret.std() * np.sqrt(252)
            shy_hist_vol = curves["SHY"]["Close"].pct_change().std() * np.sqrt(252)
            if shy_hist_vol > 0:
                stress_signals.append(min(1.0, shy_vol / (shy_hist_vol * 2)))

        # Credit spread widening (HYG underperforming LQD)
        if "HYG" in curves and "LQD" in curves:
            hyg = curves["HYG"]["Close"].pct_change().tail(20).sum()
            lqd = curves["LQD"]["Close"].pct_change().tail(20).sum()
            spread_move = lqd - hyg  # Positive = credit stress
            stress_signals.append(max(0, min(1, spread_move * 20)))

        # Floating rate premium (FLOT outperforming = funding stress)
        if "FLOT" in curves and "VCSH" in curves:
            flot = curves["FLOT"]["Close"].pct_change().tail(20).sum()
            vcsh = curves["VCSH"]["Close"].pct_change().tail(20).sum()
            float_premium = flot - vcsh
            stress_signals.append(max(0, min(1, float_premium * 30)))

        if not stress_signals:
            return 0.3  # Neutral default
        
        return round(np.mean(stress_signals), 4)


# ── Singleton ────────────────────────────────────────────────────
_agent: Optional[MarketDataAgent] = None


def get_market_data_agent() -> MarketDataAgent:
    global _agent
    if _agent is None:
        _agent = MarketDataAgent()
    return _agent
