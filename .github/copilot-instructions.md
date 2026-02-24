# OrbitalRates — Copilot Instructions

## Architecture Overview

OrbitalRates is a **7-layer multi-agent fixed income relative value platform**. Backend (Python 3.11/FastAPI) runs a sequential agent pipeline; frontend (Next.js 15/React 19) is a single-page dashboard consuming one main endpoint.

**Data flow:** `Orchestrator.run_cycle()` in `backend/app/orchestrator/orchestrator.py` calls L1→L3→L2→L5→L4→Governance→L6→L7 (note: L3 runs *before* L2 to affect filtering; L5 runs *before* L4 for risk validation of sized positions).

**Key invariant:** Governance gates *every* trade. The kill switch (`GovernanceLayer.is_halted`) short-circuits the pipeline after L3. Auto-triggers fire on drawdown > 8%, survival < 99%, or stress loss > 8% — see `backend/app/governance/governance.py::check_auto_triggers`.

## Project-Specific Patterns

### Backend singleton pattern
Every agent and governance layer uses a module-level singleton with a `get_*()` factory. When adding a new service, follow this exact pattern at the bottom of the module:
```python
_instance: Optional[MyAgent] = None
def get_my_agent() -> MyAgent:
    global _instance
    if _instance is None:
        _instance = MyAgent()
    return _instance
```

### Triple-layer type mirroring
Adding a new data field requires updates in **three** places, plus the converter:
1. **Domain dataclass** in `backend/app/core/types.py` — use `@dataclass` with `field(default_factory=...)` for mutables; enums inherit `(str, Enum)` for JSON serialization
2. **Pydantic response** in `backend/app/schemas/responses.py` — `BaseModel` mirroring the dataclass
3. **TypeScript interface** in `frontend/src/lib/api.ts` — mirrors the response schema
4. **Converter** `_state_to_dashboard()` in `backend/app/api/orbital.py` — manual field-by-field mapping from `SystemState` → `DashboardResponse`

### Configuration
All risk parameters live in `backend/app/core/config.py::Settings` (pydantic-settings). Env vars use `ORBITAL_` prefix (e.g., `ORBITAL_MAX_LEVERAGE`). Risk limits are **hard constraints** — never relax without governance review.

### Frontend patterns
- **Single state owner:** `frontend/src/app/page.tsx` holds all state, passes slices as props to child components
- **1:1 component-to-layer mapping:** `RegimeDisplay.tsx`→L3, `RiskMetrics.tsx`→L4, `SpreadOpportunities.tsx`→L2, `Positions.tsx`→L5, `StressTests.tsx`→L4, `ExecutionSim.tsx`→L6, `DecayMonitor.tsx`→L7, `GovernancePanel.tsx`→Governance
- **Mock fallback:** Frontend auto-falls back to `frontend/src/lib/mockData.ts` when backend is unavailable. Keep `MOCK_DASHBOARD` realistic and matching `DashboardData`
- **Styling:** Tailwind CSS 4 with CSS custom properties in `globals.css` (`var(--card-bg)`, `var(--bg-primary)`). Use `.card` utility class for panel containers. Dark theme only

## Development Commands

```bash
# Backend (from repo root)
cd backend && source ../.venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Frontend (from repo root)
cd frontend && npm run dev

# Tests (from backend/)
cd backend && python -m pytest tests/ -v
```

Backend: `:8000` (Swagger at `/docs`). Frontend: `:3000`. CORS pre-configured for both. API prefix: `/api/v1`.

## API Surface

All endpoints in `backend/app/api/orbital.py` under `/api/v1/orbital/`:
| Endpoint | Method | Effect |
|----------|--------|--------|
| `/dashboard?nav=100000000` | GET | Runs full 7-layer cycle, returns `DashboardResponse` |
| `/state` | GET | Returns last state without re-running |
| `/kill-switch/activate?reason=...` | POST | Emergency halt |
| `/kill-switch/deactivate?reason=...` | POST | Resume trading |
| `/audit` | GET | Last 50 governance audit entries |
| `/health` | GET | System status + cycle count |

## Critical Constraints (Never Violate)

| Parameter | Limit | Enforced In |
|-----------|-------|-------------|
| Max drawdown | 8% | `governance.py::check_auto_triggers` |
| Single event loss | 3% NAV | `config.py` |
| Survival probability | ≥ 99% | `governance.py::check_auto_triggers` |
| Max leverage | 10× | `config.py` + regime-adaptive caps in L3 |
| Single position weight | ≤ 20% | `governance.py::_validate_position` |
| Crisis regime | No new trades | `governance.py::_validate_position` |

## Adding a New Agent Layer

1. Create `backend/app/agents/layerN_name/` with `__init__.py` and engine module
2. Add class with async methods, singleton `get_*()` at module bottom
3. Add output `@dataclass` to `types.py`, response `BaseModel` to `responses.py`, TS interface to `api.ts`
4. Wire into `Orchestrator.__init__()` and `run_cycle()` in correct pipeline position
5. Update `_state_to_dashboard()` converter in `orbital.py`
6. Create frontend component in `frontend/src/components/`, add to `page.tsx` grid layout
7. Update mock data in `frontend/src/lib/mockData.ts`
