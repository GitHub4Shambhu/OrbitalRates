"""
OrbitalRates — FastAPI Application Entry Point

AI-Native Global Fixed Income Relative Value Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.api.orbital import router as orbital_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle."""
    logger.info("🛰️  OrbitalRates starting up...")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Max leverage: {settings.max_leverage}x")
    logger.info(f"   Max drawdown: {settings.max_drawdown_pct}%")
    logger.info(f"   Survival target: {settings.survival_probability_min*100}%")
    logger.info(f"   Target Sharpe: {settings.target_sharpe}")
    yield
    logger.info("👋 OrbitalRates shutting down...")


app = FastAPI(
    title="OrbitalRates",
    description=(
        "AI-Native Global Fixed Income Relative Value Platform. "
        "Multi-agent system for institutional-grade market-neutral "
        "fixed income trading."
    ),
    version=settings.version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(orbital_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "name": "OrbitalRates",
        "version": settings.version,
        "description": "AI-Native Fixed Income Relative Value Platform",
        "docs": "/docs",
        "dashboard": f"{settings.api_prefix}/orbital/dashboard",
    }
