"""
OrbitalRates — Core Configuration

Institutional-grade configuration with environment variable overrides.
All risk parameters are conservative by default.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Global system configuration."""

    # ── Identity ─────────────────────────────────────────────
    app_name: str = "OrbitalRates"
    version: str = "0.1.0"
    environment: str = Field("development", env="ORBITAL_ENV")
    debug: bool = Field(True, env="ORBITAL_DEBUG")

    # ── API ──────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ── Risk Parameters (HARD CONSTRAINTS) ───────────────────
    max_drawdown_pct: float = Field(8.0, description="Max portfolio drawdown %")
    max_single_event_loss_pct: float = Field(3.0, description="Max single event loss % NAV")
    survival_probability_min: float = Field(0.99, description="Min annual survival probability")
    target_sharpe: float = Field(1.5, description="Target Sharpe ratio")
    max_leverage: float = Field(10.0, description="Absolute max leverage cap")
    dynamic_leverage_floor: float = Field(2.0, description="Min leverage in crisis regime")

    # ── Kelly Fraction ───────────────────────────────────────
    kelly_fraction_min: float = 0.2
    kelly_fraction_max: float = 0.4
    kelly_fraction_default: float = 0.3

    # ── Spread Discovery ─────────────────────────────────────
    zscore_entry_threshold: float = Field(1.5, description="Z-score to enter trade")
    zscore_exit_threshold: float = Field(0.5, description="Z-score to exit trade")
    max_halflife_days: int = Field(90, description="Max acceptable half-life in days")
    min_halflife_days: int = Field(3, description="Min half-life (reject noise)")
    structural_break_pvalue: float = Field(0.05, description="Chow test p-value threshold")
    min_liquidity_score: float = Field(0.3, description="Min liquidity score 0-1")

    # ── Regime Thresholds ────────────────────────────────────
    vol_regime_lookback: int = 60
    correlation_expansion_threshold: float = 0.75
    liquidity_contraction_threshold: float = 0.4
    crisis_vol_multiplier: float = 2.0

    # ── Stress Test Scenarios ────────────────────────────────
    max_stress_loss_pct: float = Field(25.0, description="Kill switch on worst-case stress loss %")
    stress_spread_widening_bps: float = 200.0
    stress_vol_multiplier: float = 2.0
    stress_correlation_shock: float = 0.95
    stress_funding_spread_bps: float = 150.0
    stress_liquidity_haircut: float = 0.5

    # ── Execution ────────────────────────────────────────────
    max_slippage_bps: float = 5.0
    market_impact_decay: float = 0.5
    min_order_book_depth: float = 1_000_000.0

    # ── Data Sources ─────────────────────────────────────────
    data_refresh_seconds: int = 300
    curve_lookback_days: int = 252
    correlation_lookback_days: int = 120

    # ── Universe ─────────────────────────────────────────────
    sovereign_countries: List[str] = [
        "US", "DE", "GB", "JP", "AU", "CA", "FR", "IT", "ES", "NZ",
        "CH", "SE", "NO", "KR", "CN", "IN", "BR", "MX", "ZA", "PL",
    ]
    tenors: List[str] = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

    # ── Meta Learning ────────────────────────────────────────
    decay_lookback_window: int = 60
    retraining_trigger_decay_pct: float = 20.0
    crowding_correlation_threshold: float = 0.85

    # ── Infrastructure ───────────────────────────────────────
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    model_config = {"env_prefix": "ORBITAL_", "case_sensitive": False}


settings = Settings()
