"""
Basic tests for OrbitalRates agents and orchestrator.
"""

import pytest
import asyncio
from app.agents.layer1_data.market_data_agent import MarketDataAgent
from app.agents.layer2_discovery.spread_graph_engine import SpreadGraphEngine
from app.agents.layer3_regime.regime_engine import RegimeClassificationEngine
from app.agents.layer4_risk.tail_risk_engine import TailRiskEngine
from app.agents.layer5_capital.capital_allocator import CapitalAllocator
from app.agents.layer6_execution.execution_engine import ExecutionEngine
from app.agents.layer7_meta.meta_learning_agent import MetaLearningAgent
from app.core.types import MarketRegime


class TestMarketDataAgent:
    def test_init(self):
        agent = MarketDataAgent()
        assert agent.data_source == "live"  # No mocks yet

    @pytest.mark.asyncio
    async def test_get_curve_matrix(self):
        agent = MarketDataAgent()
        matrix = await agent.get_curve_matrix(symbols=["TLT", "SHY"], lookback_days=60)
        assert len(matrix.curves) == 2
        assert matrix.data_source in ["live", "stale"]


class TestSpreadGraphEngine:
    def test_init(self):
        engine = SpreadGraphEngine()
        assert engine is not None


class TestRegimeEngine:
    def test_init(self):
        engine = RegimeClassificationEngine()
        assert engine.current_regime is None


class TestTailRiskEngine:
    def test_init(self):
        engine = TailRiskEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_empty_portfolio(self):
        engine = TailRiskEngine()
        from app.core.types import RegimeState
        regime = RegimeState(
            regime=MarketRegime.STABLE_MEAN_REVERTING,
            confidence=0.9,
            vol_percentile=0.3,
            correlation_level=0.4,
            liquidity_index=0.7,
            funding_stress=0.1,
            leverage_cap=8.0,
            halflife_tolerance=90,
            regime_duration_days=30,
        )
        metrics = await engine.compute_risk_metrics([], regime)
        assert metrics.survival_probability == 1.0
        assert metrics.gross_leverage == 0


class TestCapitalAllocator:
    def test_init(self):
        allocator = CapitalAllocator()
        assert allocator is not None


class TestExecutionEngine:
    def test_init(self):
        engine = ExecutionEngine()
        assert engine is not None


class TestMetaLearningAgent:
    def test_init(self):
        agent = MetaLearningAgent()
        assert agent is not None
