<div align="center">

# 🛰️ OrbitalRates

### AI-Native Global Fixed Income Relative Value Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A 7-layer multi-agent architecture for institutional fixed income relative value trading.**

*Survive every regime. Compound at 15–20% with controlled drawdown.*

[Architecture](#architecture) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Agents](#agent-layers) · [Dashboard](#dashboard) · [Configuration](#configuration)

</div>

---

## Overview

OrbitalRates is a research-grade, AI-native platform for discovering and executing market-neutral fixed income relative value trades across sovereign curves, credit spreads, breakeven inflation, and cross-border opportunities.

The system operates as a **hierarchical multi-agent pipeline** — seven specialized agents process market data through discovery, regime classification, risk simulation, capital allocation, execution modeling, and meta-learning. A separate **governance layer** enforces hard constraints, kill switches, and audit trails.

### Performance Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| **Sharpe Ratio** | > 1.5 | Meta-learning feedback loop |
| **Max Drawdown** | < 8% | Hard governance kill switch |
| **Single Event Loss** | < 3% NAV | Stress test gating |
| **Survival Probability** | > 99% annually | Monte Carlo simulation |
| **Leverage** | ≤ 10× gross | Regime-adaptive caps |

### Design Philosophy

> *"The goal is not to predict the future. The goal is to survive every possible future and compound through the ones that favor you."*

- **Survival over return** — always
- **Fat-tail aware** — never assume Gaussian (Student-t with df=4)
- **No look-ahead / survivorship bias**
- **Realistic funding costs & liquidity constraints**
- **Margin & stress simulation before every trade**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GOVERNANCE LAYER                            │
│   Kill Switch · Region/Type Caps · Audit Trail · Approvals     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ L1: DATA │→│L2: DISCO- │→│L3: REGIME│→│ L4: RISK │       │
│  │  Market  │  │  VERY     │  │  Class.  │  │ Tail/MC  │       │
│  │  Data    │  │  Spreads  │  │  Engine  │  │ Survival │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       │                                          │              │
│       │         ┌──────────┐  ┌──────────┐       │              │
│       └────────→│L5: CAPI- │→│L6: EXEC- │←──────┘              │
│                 │  TAL     │  │  UTION   │                      │
│                 │  Alloc.  │  │  Engine  │                      │
│                 └──────────┘  └──────────┘                      │
│                       │              │                           │
│                 ┌──────────┐         │                           │
│                 │ L7: META │←────────┘                           │
│                 │ Learning │                                     │
│                 └──────────┘                                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ORCHESTRATOR (run_cycle)                     │   │
│  │   Coordinates L1→L2→L3→L4→L5→L6→L7 + Governance         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.11, FastAPI, Pydantic v2 |
| **Quantitative** | NumPy, Pandas, SciPy, statsmodels, arch |
| **ML/Statistical** | scikit-learn, Hurst exponent, ADF tests |
| **Market Data** | yfinance (ETF proxies for rate instruments) |
| **Frontend** | Next.js 15, React 19, TypeScript 5, Tailwind CSS 4 |
| **Logging** | Loguru (structured, leveled) |

---

## Agent Layers

### Layer 1 — Market Data Agent
> `backend/app/agents/layer1_data/market_data_agent.py`

Fetches real-time and historical data for **20 fixed-income ETFs** spanning:
- **US Treasuries**: TLT, IEF, SHY, GOVT, VGLT, VGIT, VGSH
- **TIPS / Inflation**: TIP, STIP
- **Credit**: LQD, HYG, VCSH, VCLT
- **MBS / Agency**: MBB, GNMA
- **International**: BNDX, IGOV, EMB
- **Floating Rate**: FLOT, TFLO

**Constructs:**
- Spread matrix (12 pairs: curve, credit, TIPS breakeven, international, MBS, floating)
- Rolling correlations (60-day window)
- Volatility surfaces (20-day annualized)
- Liquidity scores (volume CV + trend analysis + bid-ask tightness proxy)
- Funding stress index (short-term vol + credit spread + floating rate premium)

Falls back to mock data with `_mock_symbols` tracking when live data is unavailable.

---

### Layer 2 — Spread Graph Engine
> `backend/app/agents/layer2_discovery/spread_graph_engine.py`

Builds a **spread graph** from curve matrix data and runs statistical filters:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Z-Score | > 2.0σ | Dislocation detection |
| Half-life | 5–90 days | Mean-reversion speed |
| ADF p-value | < 0.05 | Stationarity confirmation |
| Structural break | < 0.3 prob | Regime safety |
| Liquidity score | > 0.3 | Executable check |

**Signals produced:** `ENTER_LONG`, `ENTER_SHORT`, `EXIT`, `HOLD`, `REJECT`

Half-life formula: $t_{1/2} = -\frac{\ln 2}{\ln |φ_1|}$ where $φ_1$ is the AR(1) coefficient.

---

### Layer 3 — Regime Classification Engine
> `backend/app/agents/layer3_regime/regime_engine.py`

Classifies the current market into one of **5 regimes**:

| Regime | Leverage Cap | Halflife Tolerance | Trigger |
|--------|-------------|-------------------|---------|
| **Stable** | 1.0× (of max) | 1.0× | Default |
| **Volatile** | 0.7× | 0.8× | Vol %ile > 70 |
| **Liquidity Stress** | 0.5× | 0.6× | Liquidity < 0.4 |
| **Structural Break** | 0.3× | 0.5× | Funding stress > 0.7 |
| **Crisis** | 0.1× | 0.3× | Vol %ile > 90 & Corr > 0.8 |

Classification priority: Crisis > Structural > Liquidity > Volatile > Stable.

---

### Layer 4 — Tail Risk & Survival Engine
> `backend/app/agents/layer4_risk/tail_risk_engine.py`

**Monte Carlo simulation** with fat tails (Student-t, df=4) and regime-adjusted volatility:

| Regime | Vol Multiplier |
|--------|---------------|
| Stable | 1.0× |
| Volatile | 1.5× |
| Liquidity | 2.0× |
| Structural | 2.5× |
| Crisis | 3.0× |

**6 stress scenarios tested every cycle:**
1. 2008-style spread blowout (+300 bps)
2. 2020 liquidity seizure (corr → 0.95, vol × 3)
3. Correlation convergence (all corr → 1)
4. Volatility doubling
5. Funding rate spike (+200 bps)
6. Forced deleveraging (50% liquidation at 3× slippage)

Survival probability: $P_{survival} = (1 - p_{fatal/day})^{252}$

---

### Layer 5 — Capital Allocator
> `backend/app/agents/layer5_capital/capital_allocator.py`

**Adaptive fractional Kelly** sizing:
- Kelly fraction range: 0.2–0.4 (scaled by regime)
- Regime multipliers: Stable 1.0× → Crisis 0.0×
- Liquidity penalty applied
- Crowding penalty applied
- Automatic leverage normalization to regime cap

---

### Layer 6 — Execution Engine
> `backend/app/agents/layer6_execution/execution_engine.py`

Simulates realistic execution:
- Base slippage: $\sqrt{\text{position size}}$
- Regime multiplier (1× stable → 5× crisis)
- Market impact: $\sqrt{\text{participation rate}}$
- Fill rate estimation based on liquidity
- Auto-reduces order if slippage exceeds `max_slippage_bps`

---

### Layer 7 — Meta Learning Agent
> `backend/app/agents/layer7_meta/meta_learning_agent.py`

Monitors strategy decay:
- **Halflife drift** — are mean-reversion speeds changing?
- **Edge decay** — are returns degrading?
- **Regime frequency** — are regimes shifting more often?
- **Correlation clustering** — are correlations rising?
- **Crowding increase** — are strategies becoming crowded?

Triggers retraining recommendation if decay > 20%. Auto-adjusts: max halflife, z-score threshold, leverage, crowding penalty.

---

### Governance Layer
> `backend/app/governance/governance.py`

Hard constraints enforced **before and after** every trade:

| Constraint | Limit |
|-----------|-------|
| US exposure | ≤ 60% |
| EU exposure | ≤ 40% |
| UK exposure | ≤ 25% |
| Japan exposure | ≤ 20% |
| EM exposure | ≤ 15% |
| Curve trades | ≤ 40% |
| Swap spread | ≤ 30% |

**Kill switch** auto-triggers on: drawdown breach, survival probability breach, stress test failure.

Full audit trail for every decision.

---

## Quick Start

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Node.js 18+ (20 recommended)
- Git

### Backend Setup
```bash
cd backend
python -m venv ../.venv
source ../.venv/bin/activate   # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at **http://localhost:3000** and the API at **http://localhost:8000**.

### Environment Variables (Optional)
Create a `.env` file in the project root:
```env
ORBITAL_ENV=development
ORBITAL_DEBUG=true
ORBITAL_MAX_LEVERAGE=10.0
ORBITAL_MAX_DRAWDOWN_PCT=8.0
ORBITAL_SURVIVAL_PROBABILITY_MIN=0.99
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1/orbital`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard` | Run full cycle and return complete system state |
| `GET` | `/dashboard?nav=1000000000` | Run cycle with specific NAV ($1B) |
| `GET` | `/state` | Get last cycle state (no re-run) |
| `POST` | `/kill-switch/activate?reason=...` | Activate emergency halt |
| `POST` | `/kill-switch/deactivate?reason=...` | Deactivate halt |
| `GET` | `/audit` | Get full audit trail |
| `GET` | `/health` | System health check |

### Dashboard Response Schema
```json
{
  "cycle_count": 1,
  "data_source": "live",
  "is_halted": false,
  "cycle_duration_ms": 14523.7,
  "timestamp": "2026-02-23T18:38:00Z",
  "regime": {
    "regime": "stable_mean_reverting",
    "confidence": 0.735,
    "vol_percentile": 0.342,
    "leverage_cap": 10.0,
    "..."
  },
  "opportunities": [...],
  "positions": [...],
  "risk_metrics": { "survival_probability": 0.9987, "..." },
  "stress_results": [...],
  "execution_results": [...],
  "decay_metrics": { "..." },
  "audit_log": [...]
}
```

### Interactive Docs
When the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Dashboard

The Next.js institutional dashboard provides real-time visualization of all 7 agent layers:

| Panel | Agent | Shows |
|-------|-------|-------|
| **Regime Display** | L3 | Current regime, confidence, meter bars for vol/corr/liquidity/funding |
| **Risk Metrics** | L4 | Survival probability, Sharpe estimate, leverage, VaR, ES, drawdown |
| **Spread Discovery** | L2 | Active opportunities table with z-scores, halflife, signals |
| **Active Positions** | L5 | Trade book with P&L, direction, DV01, weights |
| **Stress Tests** | L4 | Pass/fail for all 6 scenarios with loss magnitudes |
| **Execution Sim** | L6 | Slippage, fill rates, market impact per trade |
| **Decay Monitor** | L7 | Strategy health: halflife drift, edge decay, crowding |
| **Governance** | Gov | Kill switch control, audit trail, system status |

---

## Configuration

All parameters are configurable via environment variables or `backend/app/core/config.py`:

### Risk Parameters
| Parameter | Default | Env Var |
|-----------|---------|---------|
| Max Drawdown | 8.0% | `ORBITAL_MAX_DRAWDOWN_PCT` |
| Max Single Event Loss | 3.0% | `ORBITAL_MAX_SINGLE_EVENT_LOSS_PCT` |
| Survival Probability | 99.0% | `ORBITAL_SURVIVAL_PROBABILITY_MIN` |
| Max Leverage | 10.0× | `ORBITAL_MAX_LEVERAGE` |
| Target Sharpe | 1.5 | `ORBITAL_TARGET_SHARPE` |

### Spread Discovery
| Parameter | Default | Description |
|-----------|---------|-------------|
| Z-Score Entry | 2.0σ | Minimum dislocation to enter |
| Min Halflife | 5 days | Fastest acceptable mean-reversion |
| Max Halflife | 90 days | Slowest acceptable mean-reversion |
| ADF p-value | 0.05 | Stationarity significance level |

### Kelly Fraction
| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Fraction | 0.2 | Conservative sizing floor |
| Max Fraction | 0.4 | Aggressive sizing ceiling |
| Default | 0.3 | Starting fraction |

---

## Project Structure

```
OrbitalRates/
├── README.md                       # This file
├── ARCHITECTURE.md                 # Deep technical documentation
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT License
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
│
├── backend/                        # Python FastAPI + 7-Agent Engine
│   ├── requirements.txt            # Python dependencies
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── agents/
│   │   │   ├── layer1_data/        # Market Data Agent
│   │   │   │   └── market_data_agent.py
│   │   │   ├── layer2_discovery/   # Spread Graph Engine
│   │   │   │   └── spread_graph_engine.py
│   │   │   ├── layer3_regime/      # Regime Classification
│   │   │   │   └── regime_engine.py
│   │   │   ├── layer4_risk/        # Tail Risk & Survival
│   │   │   │   └── tail_risk_engine.py
│   │   │   ├── layer5_capital/     # Capital Allocator
│   │   │   │   └── capital_allocator.py
│   │   │   ├── layer6_execution/   # Execution Engine
│   │   │   │   └── execution_engine.py
│   │   │   └── layer7_meta/        # Meta Learning Agent
│   │   │       └── meta_learning_agent.py
│   │   ├── api/
│   │   │   └── orbital.py          # REST API endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic Settings
│   │   │   └── types.py            # Dataclasses & enums
│   │   ├── governance/
│   │   │   └── governance.py       # Kill switch, caps, audit
│   │   ├── orchestrator/
│   │   │   └── orchestrator.py     # Multi-agent coordinator
│   │   └── schemas/
│   │       └── responses.py        # Pydantic response models
│   └── tests/
│       └── test_agents.py          # Agent unit tests
│
├── frontend/                       # Next.js 15 Institutional Dashboard
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── postcss.config.mjs
│   └── src/
│       ├── app/
│       │   ├── globals.css         # Dark theme styles
│       │   ├── layout.tsx          # Root layout
│       │   └── page.tsx            # Main dashboard
│       ├── components/
│       │   ├── RegimeDisplay.tsx    # L3 regime visualization
│       │   ├── RiskMetrics.tsx      # L4 risk gauges
│       │   ├── SpreadOpportunities.tsx # L2 spread table
│       │   ├── Positions.tsx        # L5 trade book
│       │   ├── StressTests.tsx      # L4 stress scenarios
│       │   ├── ExecutionSim.tsx     # L6 execution display
│       │   ├── GovernancePanel.tsx  # Kill switch & audit
│       │   ├── DecayMonitor.tsx     # L7 strategy health
│       │   └── DataSourceBadge.tsx  # Live/mock indicator
│       └── lib/
│           └── api.ts              # TypeScript API client
│
└── .github/
    └── copilot-instructions.md     # AI assistant context
```

---

## Spread Types

The system discovers and trades **7 spread types**:

| Type | Example | Edge |
|------|---------|------|
| **Curve** | TLT/SHY (long/short duration) | Term premium mean-reversion |
| **Credit** | HYG/LQD (high yield vs IG) | Credit spread normalization |
| **TIPS Breakeven** | TIP/IEF (inflation vs nominal) | Breakeven dislocation |
| **International** | BNDX/GOVT (foreign vs domestic) | Cross-border convergence |
| **MBS Spread** | MBB/GOVT (mortgage vs sovereign) | Prepayment/agency premium |
| **Floating** | FLOT/SHY (floating vs fixed short) | Rate expectation arb |
| **Swap Spread** | Various | Swap spread normalization |

---

## Running Tests

```bash
cd backend
source ../.venv/bin/activate
python -m pytest tests/ -v
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for institutional-grade research. Not financial advice.**

*OrbitalRates v0.1.0*

</div>
