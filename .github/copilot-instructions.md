# OrbitalRates — Copilot Instructions

## Project Overview
OrbitalRates is an AI-native institutional fixed income relative value platform with a 7-layer multi-agent architecture, Next.js frontend, and Python FastAPI backend.

## Architecture
- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS 4 in `/frontend`
- **Backend**: Python 3.11, FastAPI, 7-layer agent stack in `/backend`
- **Governance**: Kill switch, audit trail, exposure caps in `/backend/app/governance`
- **Orchestrator**: Multi-agent coordinator in `/backend/app/orchestrator`

## Key Files

### Backend
- `backend/app/main.py` — FastAPI entry point with CORS and lifespan
- `backend/app/core/config.py` — Pydantic Settings with all risk parameters
- `backend/app/core/types.py` — Dataclasses: MarketRegime, SpreadCandidate, RegimeState, etc.
- `backend/app/agents/layer1_data/market_data_agent.py` — Market data fetching (20 ETFs)
- `backend/app/agents/layer2_discovery/spread_graph_engine.py` — Spread discovery, z-scores, halflife
- `backend/app/agents/layer3_regime/regime_engine.py` — 5-regime classifier
- `backend/app/agents/layer4_risk/tail_risk_engine.py` — Monte Carlo, stress tests, survival
- `backend/app/agents/layer5_capital/capital_allocator.py` — Fractional Kelly sizing
- `backend/app/agents/layer6_execution/execution_engine.py` — Slippage, fill rate, market impact
- `backend/app/agents/layer7_meta/meta_learning_agent.py` — Strategy decay, parameter drift
- `backend/app/governance/governance.py` — Kill switch, region/type caps, audit
- `backend/app/orchestrator/orchestrator.py` — L1→L7 coordinator
- `backend/app/api/orbital.py` — REST endpoints
- `backend/app/schemas/responses.py` — Pydantic response models

### Frontend
- `frontend/src/app/page.tsx` — Main dashboard page
- `frontend/src/lib/api.ts` — TypeScript API client with all types
- `frontend/src/components/RegimeDisplay.tsx` — L3 regime visualization
- `frontend/src/components/RiskMetrics.tsx` — L4 risk gauges
- `frontend/src/components/SpreadOpportunities.tsx` — L2 spread table
- `frontend/src/components/Positions.tsx` — L5 trade book
- `frontend/src/components/StressTests.tsx` — L4 stress scenarios
- `frontend/src/components/ExecutionSim.tsx` — L6 execution display
- `frontend/src/components/GovernancePanel.tsx` — Kill switch & audit
- `frontend/src/components/DecayMonitor.tsx` — L7 strategy health
- `frontend/src/components/DataSourceBadge.tsx` — Live/mock data indicator

## Development Commands

### Start Backend
```bash
cd backend && source ../.venv/bin/activate && uvicorn app.main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend && npm run dev
```

## API Endpoints
- `GET /api/v1/orbital/dashboard` — Full cycle run + system state
- `GET /api/v1/orbital/state` — Last state (no re-run)
- `POST /api/v1/orbital/kill-switch/activate?reason=...` — Emergency halt
- `POST /api/v1/orbital/kill-switch/deactivate?reason=...` — Resume
- `GET /api/v1/orbital/audit` — Audit trail
- `GET /api/v1/orbital/health` — Health check

## Data Types
- `MarketRegime` — 5 states: stable, volatile, liquidity, structural, crisis
- `SpreadCandidate` — 25+ field trade opportunity
- `RegimeState` — Classification + constraints
- `RiskMetrics` — Full risk snapshot
- `StressResult` — Scenario test result
- `PortfolioPosition` — Active trade
- `ExecutionResult` — Simulated execution
- `DecayMetrics` — Strategy health metrics

## Risk Parameters (Hard Constraints)
- Max Drawdown: 8%
- Max Single Event Loss: 3% NAV
- Survival Probability: > 99%
- Max Leverage: 10×
- Target Sharpe: 1.5

## Spread Types
Curve, Credit, TIPS Breakeven, International, MBS, Floating, Swap Spread

## 7-Layer Pipeline
L1 Data → L2 Discovery → L3 Regime → L4 Risk → L5 Capital → L6 Execution → L7 Meta + Governance
