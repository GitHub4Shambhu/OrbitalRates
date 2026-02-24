"""
Multi-Agent Orchestrator — The Autonomous Hedge Fund Brain

Coordinates all 7 agent layers in the correct sequence:
  L1 → Data ingestion
  L2 → Dislocation discovery
  L3 → Regime classification
  L4 → Risk validation
  L5 → Capital allocation
  L6 → Execution simulation
  L7 → Meta learning / decay monitoring

Plus governance overlay on every decision.
"""

from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

from app.core.types import (
    SpreadCandidate, RegimeState, PortfolioPosition,
    RiskMetrics, StressResult, ExecutionResult, DecayMetrics,
)
from app.agents.layer1_data.market_data_agent import (
    MarketDataAgent, CurveMatrix, get_market_data_agent,
)
from app.agents.layer2_discovery.spread_graph_engine import (
    SpreadGraphEngine, get_spread_graph_engine,
)
from app.agents.layer3_regime.regime_engine import (
    RegimeClassificationEngine, get_regime_engine,
)
from app.agents.layer4_risk.tail_risk_engine import (
    TailRiskEngine, get_tail_risk_engine,
)
from app.agents.layer5_capital.capital_allocator import (
    CapitalAllocator, get_capital_allocator,
)
from app.agents.layer6_execution.execution_engine import (
    ExecutionEngine, get_execution_engine,
)
from app.agents.layer7_meta.meta_learning_agent import (
    MetaLearningAgent, get_meta_learning_agent,
)
from app.governance.governance import GovernanceLayer, get_governance


@dataclass
class SystemState:
    """Complete system state snapshot."""
    # Layer outputs
    curve_matrix: Optional[CurveMatrix] = None
    candidates: List[SpreadCandidate] = field(default_factory=list)
    regime: Optional[RegimeState] = None
    risk_metrics: Optional[RiskMetrics] = None
    stress_results: List[StressResult] = field(default_factory=list)
    positions: List[PortfolioPosition] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    decay_metrics: Optional[DecayMetrics] = None

    # Governance
    approved_positions: List[PortfolioPosition] = field(default_factory=list)
    rejections: List[str] = field(default_factory=list)
    audit_log: List[Dict] = field(default_factory=list)

    # Meta
    data_source: str = "live"
    is_halted: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cycle_duration_ms: float = 0.0


class Orchestrator:
    """
    The Autonomous Hedge Fund Brain.
    
    Runs the full agent pipeline and maintains system state.
    """

    def __init__(self):
        self.data_agent: MarketDataAgent = get_market_data_agent()
        self.spread_engine: SpreadGraphEngine = get_spread_graph_engine()
        self.regime_engine: RegimeClassificationEngine = get_regime_engine()
        self.risk_engine: TailRiskEngine = get_tail_risk_engine()
        self.allocator: CapitalAllocator = get_capital_allocator()
        self.execution_engine: ExecutionEngine = get_execution_engine()
        self.meta_agent: MetaLearningAgent = get_meta_learning_agent()
        self.governance: GovernanceLayer = get_governance()

        self._state: Optional[SystemState] = None
        self._cycle_count: int = 0

        logger.info("🧠 Orchestrator initialized — all 7 agents + governance ready")

    # ── Full Pipeline ────────────────────────────────────────

    async def run_cycle(self, nav: float = 100_000_000.0) -> SystemState:
        """
        Run one full agent cycle.
        
        L1 → L2 → L3 → L4 → L5 → L6 → L7 + Governance
        """
        start = datetime.utcnow()
        self._cycle_count += 1
        logger.info(f"═══ Cycle {self._cycle_count} START ═══")

        state = SystemState()

        # ── L1: Market Data ──────────────────────────────────
        logger.info("L1 — Fetching market data...")
        state.curve_matrix = await self.data_agent.get_curve_matrix()
        state.data_source = state.curve_matrix.data_source
        logger.info(f"L1 complete: {len(state.curve_matrix.curves)} instruments, data_source={state.data_source}")

        # ── L3: Regime Classification (before L2, affects filtering) ─
        logger.info("L3 — Classifying regime...")
        state.regime = await self.regime_engine.classify_regime(state.curve_matrix)
        logger.info(f"L3 complete: regime={state.regime.regime.value}, confidence={state.regime.confidence:.2f}")

        # ── Check governance halt ────────────────────────────
        if self.governance.is_halted:
            state.is_halted = True
            state.audit_log = self.governance.recent_audit
            logger.warning("System HALTED — skipping trading pipeline")
            self._state = state
            return state

        # ── L2: Dislocation Discovery ────────────────────────
        logger.info("L2 — Discovering dislocations...")
        state.candidates = await self.spread_engine.discover_opportunities(
            state.curve_matrix,
            state.curve_matrix.liquidity_scores,
        )
        logger.info(f"L2 complete: {len(state.candidates)} valid opportunities")

        # ── L5: Capital Allocation ───────────────────────────
        logger.info("L5 — Allocating capital...")
        state.positions = await self.allocator.allocate(
            state.candidates,
            state.regime,
            nav=nav,
        )
        logger.info(f"L5 complete: {len(state.positions)} positions")

        # ── L4: Risk Validation ──────────────────────────────
        logger.info("L4 — Computing risk metrics...")
        state.risk_metrics = await self.risk_engine.compute_risk_metrics(
            state.positions, state.regime, state.candidates, nav=nav,
        )
        state.stress_results = self.risk_engine.run_stress_tests(
            state.positions, state.regime,
        )
        logger.info(
            f"L4 complete: survival={state.risk_metrics.survival_probability:.4f}, "
            f"sharpe={state.risk_metrics.sharpe_estimate:.2f}"
        )

        # ── Governance Check ─────────────────────────────────
        self.governance.check_auto_triggers(state.risk_metrics, state.regime)
        state.approved_positions, state.rejections = self.governance.approve_trades(
            state.positions, state.risk_metrics, state.regime,
        )
        state.is_halted = self.governance.is_halted

        # ── L6: Execution Simulation ─────────────────────────
        if state.approved_positions:
            logger.info("L6 — Simulating execution...")
            state.execution_results = await self.execution_engine.simulate_execution(
                state.approved_positions,
                state.regime,
                state.curve_matrix.liquidity_scores,
            )
            logger.info(f"L6 complete: {len(state.execution_results)} executions simulated")

        # ── L7: Meta Learning ────────────────────────────────
        logger.info("L7 — Assessing strategy decay...")
        state.decay_metrics = await self.meta_agent.assess_decay(
            state.candidates, state.regime,
        )
        logger.info(f"L7 complete: decay={state.decay_metrics.edge_decay_pct:.1f}%")

        # ── Finalize ─────────────────────────────────────────
        state.audit_log = self.governance.recent_audit
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        state.cycle_duration_ms = round(elapsed, 1)
        state.timestamp = datetime.utcnow()

        self._state = state
        logger.info(
            f"═══ Cycle {self._cycle_count} COMPLETE ({elapsed:.0f}ms) ═══\n"
            f"  Regime: {state.regime.regime.value}\n"
            f"  Opportunities: {len(state.candidates)}\n"
            f"  Positions: {len(state.approved_positions)}\n"
            f"  Survival: {state.risk_metrics.survival_probability:.4f}\n"
            f"  Data Source: {state.data_source}"
        )

        return state

    @property
    def current_state(self) -> Optional[SystemState]:
        return self._state

    @property
    def cycle_count(self) -> int:
        return self._cycle_count


# ── Singleton ────────────────────────────────────────────────────
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
