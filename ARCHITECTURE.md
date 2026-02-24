# OrbitalRates — Architecture Deep Dive

## System Design Principles

1. **Survival-first**: Every layer prioritizes survival over return
2. **Separation of concerns**: Each agent owns one decision domain
3. **Composability**: Agents are independently testable and replaceable
4. **Regime-awareness**: All layers adapt behavior to the current market regime
5. **Auditability**: Every decision is logged with rationale

## Data Flow

```
Market Data (yfinance ETFs)
        │
        ▼
┌──────────────────┐
│ L1: Market Data  │ → CurveMatrix (spreads, correlations, vol, liquidity, funding)
│     Agent        │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L2: Spread Graph │ → List[SpreadCandidate] with z-scores, halflife, signals
│     Engine       │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L3: Regime       │ → RegimeState (classification, confidence, constraints)
│     Engine       │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L4: Tail Risk    │ → RiskMetrics + StressResults + survival probability
│     Engine       │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L5: Capital      │ → List[PortfolioPosition] with notional sizing
│     Allocator    │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L6: Execution    │ → List[ExecutionResult] with slippage, fill rates
│     Engine       │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ L7: Meta         │ → DecayMetrics + parameter adjustments
│     Learning     │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Governance       │ → Approval/rejection + audit entry + kill switch check
│     Layer        │
└──────────────────┘
```

## Key Data Structures

### CurveMatrix (L1 output)
Contains: price DataFrames, spread DataFrames, correlation matrix, volatilities dict, liquidity scores dict, funding stress index.

### SpreadCandidate (L2 output)
25+ fields including: spread_id, spread_type, leg1, leg2, current_spread_bps, zscore, halflife_days, ar1_coefficient, is_stationary, structural_break_prob, liquidity_score, crowding_proxy, expected_return_bps, expected_shortfall_bps, vol_adjusted_return, signal, rejection_reason.

### RegimeState (L3 output)
Fields: regime (enum), confidence, vol_percentile, correlation_level, liquidity_index, funding_stress, leverage_cap, halflife_tolerance, regime_duration_days, transition_probability.

### RiskMetrics (L4 output)
Fields: total_dv01, total_convexity, gross/net leverage, expected return/shortfall, VaR 99%, max stress loss, survival probability, Sharpe estimate, drawdown, liquidity/correlation/crowding/funding risk.

### PortfolioPosition (L5 output)
Fields: position_id, spread_id, spread_type, legs, direction, notional, dv01_net, weight, entry/current spread & zscore, unrealized P&L.

## Regime Classification Logic

The regime engine runs a priority-based classifier:

```python
if vol_pct > 0.9 and corr > 0.8:
    regime = CRISIS           # Everything correlating, vol exploding
elif funding_stress > 0.7:
    regime = STRUCTURAL       # Funding markets stressed
elif liquidity < 0.4:
    regime = LIQUIDITY        # Liquidity drying up
elif vol_pct > 0.7:
    regime = VOLATILE         # High vol but no systemic stress
else:
    regime = STABLE           # Normal conditions
```

## Risk Simulation

Monte Carlo runs 10,000 paths with:
- Student-t innovations (df=4) for fat tails
- Regime-dependent vol multiplier (1.0× to 3.0×)
- Daily stepping for 252 trading days
- Survival = probability of never hitting fatal drawdown

## Governance Enforcement

The governance layer gates every trade through:
1. **Kill switch check** — is the system halted?
2. **Region exposure check** — would this trade exceed country limits?
3. **Type exposure check** — would this exceed spread type limits?
4. **Risk approval** — does the trade pass all risk constraints?
5. **Audit logging** — record decision + rationale regardless of outcome

## Orchestrator Pattern

The orchestrator uses a singleton pattern and coordinates all agents:

```python
async def run_cycle(nav: float) -> SystemState:
    curve_matrix = self.data_agent.fetch_all()          # L1
    candidates = self.discovery.discover(curve_matrix)   # L2
    regime = self.regime_engine.classify(curve_matrix)   # L3
    risk = self.risk_engine.assess(portfolio, regime)    # L4
    positions = self.allocator.allocate(candidates, ...) # L5
    executions = self.execution.simulate(positions, ...) # L6
    decay = self.meta.analyze(candidates, regime)        # L7
    # Governance checks throughout
    return SystemState(...)
```
