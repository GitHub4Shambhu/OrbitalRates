"""
OrbitalRates — Core Type Definitions

Enums and base types used across all agents.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime


# ── Regime Classification ────────────────────────────────────────

class MarketRegime(str, Enum):
    """Market regime states — drives leverage and trade sizing."""
    STABLE_MEAN_REVERTING = "stable_mean_reverting"
    VOLATILE_MEAN_REVERTING = "volatile_mean_reverting"
    LIQUIDITY_TIGHTENING = "liquidity_tightening"
    STRUCTURAL_SHIFT = "structural_shift"
    CRISIS = "crisis"


class TradeSignal(str, Enum):
    """Spread trade signals."""
    ENTER_LONG = "enter_long"    # Spread expected to narrow
    ENTER_SHORT = "enter_short"  # Spread expected to widen
    HOLD = "hold"
    EXIT = "exit"
    REJECT = "reject"


class SpreadType(str, Enum):
    """Types of fixed income spreads."""
    CURVE = "curve"                    # Steepener/flattener
    SWAP_SPREAD = "swap_spread"        # Swap vs treasury
    CROSS_COUNTRY = "cross_country"    # Cross-country basis
    FUTURES_BASIS = "futures_basis"    # Futures vs cash
    INFLATION_BREAKEVEN = "inflation_breakeven"
    REPO_DISLOCATION = "repo_dislocation"
    OIS_LIBOR_BASIS = "ois_libor_basis"


class InstrumentType(str, Enum):
    """Tradable instrument types."""
    SOVEREIGN_BOND = "sovereign_bond"
    INTEREST_RATE_SWAP = "interest_rate_swap"
    FUTURES = "futures"
    OIS = "ois"
    REPO = "repo"
    INFLATION_LINKED = "inflation_linked"


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class CurvePoint:
    """Single point on a yield curve."""
    country: str
    tenor: str
    tenor_years: float
    yield_pct: float
    dv01: float
    timestamp: datetime
    source: str = "live"
    is_stale: bool = False


@dataclass
class SpreadCandidate:
    """A candidate relative value spread trade."""
    spread_id: str
    spread_type: SpreadType
    leg1_instrument: str
    leg2_instrument: str
    leg1_country: str
    leg2_country: str
    leg1_tenor: str
    leg2_tenor: str
    current_spread_bps: float
    zscore: float
    halflife_days: float
    mean_spread_bps: float
    std_spread_bps: float
    ar1_coefficient: float
    is_stationary: bool
    structural_break_prob: float
    liquidity_score: float       # 0-1
    crowding_proxy: float        # 0-1
    expected_return_bps: float
    expected_shortfall_bps: float
    vol_adjusted_return: float
    tail_5pct_bps: float
    tail_1pct_bps: float
    signal: TradeSignal = TradeSignal.HOLD
    rejection_reason: Optional[str] = None
    # Advanced discovery fields (state-of-the-art 2025)
    hurst_exponent: float = 0.5        # <0.5 = mean-reverting, >0.5 = trending
    johansen_trace_stat: float = 0.0   # Johansen cointegration test statistic
    is_cointegrated: bool = False       # Johansen test p<0.05
    kalman_hedge_ratio: float = 1.0    # Dynamic hedge ratio from Kalman filter
    kalman_hedge_ratio_std: float = 0.0 # Uncertainty in hedge ratio
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeState:
    """Current market regime assessment."""
    regime: MarketRegime
    confidence: float           # 0-1
    vol_percentile: float       # Current vol vs history
    correlation_level: float    # Average cross-correlation
    liquidity_index: float      # 0-1 (1=abundant)
    funding_stress: float       # 0-1 (1=severe)
    leverage_cap: float         # Dynamic leverage cap for this regime
    halflife_tolerance: int     # Max acceptable half-life
    regime_duration_days: int   # How long in current regime
    transition_probability: Dict[str, float] = field(default_factory=dict)
    # HMM-derived fields (state-of-the-art 2025)
    hmm_state_probabilities: Dict[str, float] = field(default_factory=dict)
    hmm_confidence: float = 0.0
    heuristic_regime: str = ""
    ensemble_agreement: float = 0.0  # 1.0 if HMM + heuristic agree
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioPosition:
    """Active portfolio position."""
    position_id: str
    spread_id: str
    spread_type: SpreadType
    leg1_instrument: str
    leg2_instrument: str
    direction: str              # "long_spread" or "short_spread"
    notional: float
    dv01_net: float
    convexity_net: float
    entry_spread_bps: float
    current_spread_bps: float
    entry_zscore: float
    current_zscore: float
    unrealized_pnl_bps: float
    weight: float               # Portfolio weight
    leverage_contribution: float
    entry_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskMetrics:
    """Portfolio-level risk metrics."""
    total_dv01: float
    total_convexity: float
    gross_leverage: float
    net_leverage: float
    expected_return_annual_pct: float
    expected_shortfall_99_pct: float
    var_99_pct: float
    max_stress_loss_pct: float
    survival_probability: float
    sharpe_estimate: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    liquidity_weighted_exposure: float
    correlation_risk: float
    crowding_risk: float
    funding_cost_bps: float
    # EVT / advanced tail risk fields (state-of-the-art 2025)
    evt_var_99_pct: float = 0.0        # EVT-based VaR (Generalized Pareto)
    evt_es_99_pct: float = 0.0         # EVT-based Expected Shortfall
    evt_shape_parameter: float = 0.0   # GPD shape (xi): >0 = heavy tail
    tail_dependence_coeff: float = 0.0 # Upper tail dependence (copula)
    regime_vol_multiplier: float = 1.0 # Current regime vol scaling
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StressResult:
    """Result of a single stress scenario."""
    scenario_name: str
    portfolio_loss_pct: float
    margin_call_triggered: bool
    survival: bool
    description: str
    spread_widening_bps: float = 0.0
    vol_shock_multiplier: float = 1.0
    correlation_shock: float = 0.0


@dataclass 
class ExecutionResult:
    """Execution simulation result."""
    spread_id: str
    intended_notional: float
    executed_notional: float
    slippage_bps: float
    market_impact_bps: float
    fill_rate: float            # 0-1
    execution_cost_bps: float
    liquidity_available: bool


@dataclass
class DecayMetrics:
    """Strategy decay monitoring."""
    halflife_drift_pct: float
    edge_decay_pct: float
    regime_frequency_change: float
    correlation_clustering: float
    crowding_increase: float
    retraining_recommended: bool
    parameter_adjustments: Dict[str, float] = field(default_factory=dict)
    # Online learning fields (state-of-the-art 2025)
    cusum_statistic: float = 0.0          # CUSUM change-point detection stat
    cusum_alert: bool = False             # True if CUSUM exceeds threshold
    ewma_edge_estimate: float = 0.0       # Exponentially-weighted moving avg edge
    forgetting_factor: float = 0.95       # Lambda for exponential forgetting
    bayesian_zscore_threshold: float = 2.0 # Adaptively shrunk z-score entry


@dataclass
class AuditEntry:
    """Governance audit log entry."""
    timestamp: datetime
    agent: str
    action: str
    decision: str
    rationale: str
    risk_metrics: Optional[Dict] = None
    approved: bool = True
    override_reason: Optional[str] = None
