# Contributing to OrbitalRates

Thank you for your interest in contributing to OrbitalRates!

## Development Setup

1. Fork and clone the repository
2. Set up the backend:
   ```bash
   cd backend
   python -m venv ../.venv
   source ../.venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```
4. Start both servers:
   ```bash
   # Terminal 1 — Backend
   cd backend && uvicorn app.main:app --reload --port 8000

   # Terminal 2 — Frontend
   cd frontend && npm run dev
   ```

## Project Structure

- `backend/app/agents/` — 7-layer agent stack (one directory per layer)
- `backend/app/governance/` — Kill switch, audit, exposure caps
- `backend/app/orchestrator/` — Multi-agent coordinator
- `backend/app/api/` — FastAPI REST endpoints
- `backend/app/core/` — Configuration, types, constants
- `backend/app/schemas/` — Pydantic response models
- `frontend/src/components/` — Dashboard components
- `frontend/src/lib/` — API client and utilities

## Code Style

### Python
- Use type hints everywhere
- Follow PEP 8 with 100-char line length
- Use `loguru` for logging (not stdlib `logging`)
- Pydantic v2 for all data validation
- Dataclasses for internal structures, Pydantic for API boundaries

### TypeScript
- Strict mode enabled
- Use named exports
- Components are in PascalCase with `.tsx` extension
- Use Tailwind CSS for styling (no CSS modules)

## Adding a New Agent Layer

1. Create a directory: `backend/app/agents/layerN_name/`
2. Add `__init__.py`
3. Implement the agent class following the pattern in existing layers
4. Register it in the orchestrator: `backend/app/orchestrator/orchestrator.py`
5. Add response schema in `backend/app/schemas/responses.py`
6. Add frontend component in `frontend/src/components/`
7. Add tests in `backend/tests/`

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a PR with a clear description

## Reporting Issues

Please include:
- Steps to reproduce
- Expected vs actual behavior
- Python/Node.js version
- OS information
