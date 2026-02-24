"""
Governance Layer — Institutional Risk Guardrails

No AI autonomy without guardrails.
This is what separates hedge fund from trading bot.

Implements:
  - Kill switch (halt all trading immediately)
  - Exposure caps by region
  - Exposure caps by instrument type
  - Hard drawdown limits
  - Audit log of all decisions
"""

from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.types import (
    AuditEntry, PortfolioPosition, RiskMetrics, RegimeState,
    MarketRegime,
)


class GovernanceLayer:
    """
    Institutional Governance.
    
    Every trade decision passes through here.
    Kill switch overrides everything.
    """

    def __init__(self):
        self._kill_switch: bool = False
        self._audit_log: List[AuditEntry] = []
        self._region_caps: Dict[str, float] = {
            "US": 0.60,    # Max 60% in US
            "EU": 0.40,
            "GB": 0.25,
            "JP": 0.20,
            "EM": 0.15,
            "INTL": 0.30,
        }
        self._type_caps: Dict[str, float] = {
            "curve": 0.40,
            "swap_spread": 0.30,
            "cross_country": 0.25,
            "inflation_breakeven": 0.20,
            "futures_basis": 0.15,
        }
        logger.info("Governance Layer initialized")

    # ── Kill Switch ──────────────────────────────────────────

    def activate_kill_switch(self, reason: str) -> None:
        """HALT ALL TRADING IMMEDIATELY."""
        self._kill_switch = True
        self._log_decision(
            agent="governance",
            action="KILL_SWITCH_ACTIVATED",
            decision="halt_all_trading",
            rationale=reason,
            approved=True,
        )
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self, reason: str) -> None:
        """Resume trading (requires explicit reason)."""
        self._kill_switch = False
        self._log_decision(
            agent="governance",
            action="KILL_SWITCH_DEACTIVATED",
            decision="resume_trading",
            rationale=reason,
            approved=True,
        )
        logger.warning(f"Kill switch deactivated: {reason}")

    @property
    def is_halted(self) -> bool:
        return self._kill_switch

    # ── Trade Approval ───────────────────────────────────────

    def approve_trades(
        self,
        positions: List[PortfolioPosition],
        risk_metrics: RiskMetrics,
        regime: RegimeState,
    ) -> tuple[List[PortfolioPosition], List[str]]:
        """
        Final approval gate for all trades.
        
        Returns (approved_positions, rejection_reasons).
        """
        if self._kill_switch:
            reason = "KILL SWITCH ACTIVE — all trades rejected"
            self._log_decision("governance", "trade_approval", "rejected_all", reason)
            return [], [reason]

        approved = []
        rejections = []

        for pos in positions:
            ok, reason = self._validate_position(pos, risk_metrics, regime)
            if ok:
                approved.append(pos)
                self._log_decision(
                    "governance", "trade_approval", "approved",
                    f"{pos.spread_id}: {reason}",
                    risk_metrics={"leverage": risk_metrics.gross_leverage},
                )
            else:
                rejections.append(f"{pos.spread_id}: {reason}")
                self._log_decision(
                    "governance", "trade_approval", "rejected",
                    f"{pos.spread_id}: {reason}",
                    approved=False,
                )

        # Check portfolio-level constraints
        portfolio_issues = self._check_portfolio_constraints(approved, risk_metrics)
        if portfolio_issues:
            rejections.extend(portfolio_issues)
            # May need to trim positions
            approved = self._trim_to_constraints(approved)

        logger.info(f"Governance: {len(approved)} approved, {len(rejections)} rejected")
        return approved, rejections

    # ── Auto Kill-Switch Triggers ────────────────────────────

    def check_auto_triggers(self, risk_metrics: RiskMetrics, regime: RegimeState) -> None:
        """Check if auto kill-switch should be triggered."""
        # Drawdown breach
        if risk_metrics.current_drawdown_pct > settings.max_drawdown_pct:
            self.activate_kill_switch(
                f"Drawdown {risk_metrics.current_drawdown_pct:.2f}% > "
                f"limit {settings.max_drawdown_pct}%"
            )

        # Survival probability breach
        if risk_metrics.survival_probability < settings.survival_probability_min:
            self.activate_kill_switch(
                f"Survival probability {risk_metrics.survival_probability:.4f} < "
                f"minimum {settings.survival_probability_min}"
            )

        # Max stress loss breach (separate from running drawdown)
        if risk_metrics.max_stress_loss_pct > settings.max_stress_loss_pct:
            self.activate_kill_switch(
                f"Stress loss {risk_metrics.max_stress_loss_pct:.2f}% > "
                f"stress limit {settings.max_stress_loss_pct}%"
            )

    # ── Validation ───────────────────────────────────────────

    def _validate_position(
        self,
        pos: PortfolioPosition,
        risk: RiskMetrics,
        regime: RegimeState,
    ) -> tuple[bool, str]:
        """Validate a single position against governance rules."""
        # Crisis mode: no new trades
        if regime.regime == MarketRegime.CRISIS:
            return False, "CRISIS regime — no new positions allowed"

        # Leverage check
        if risk.gross_leverage > regime.leverage_cap:
            return False, f"leverage {risk.gross_leverage:.1f}x > cap {regime.leverage_cap}x"

        # Position size check (no single position > 20% of NAV)
        if pos.weight > 0.20:
            return False, f"position weight {pos.weight:.1%} > 20% cap"

        return True, "passed all governance checks"

    def _check_portfolio_constraints(
        self,
        positions: List[PortfolioPosition],
        risk: RiskMetrics,
    ) -> List[str]:
        """Check portfolio-level constraints."""
        issues = []

        # Region concentration
        region_exposure: Dict[str, float] = {}
        for pos in positions:
            region = self._infer_region(pos.leg1_instrument)
            region_exposure[region] = region_exposure.get(region, 0) + abs(pos.weight)

        for region, exposure in region_exposure.items():
            cap = self._region_caps.get(region, 0.50)
            if exposure > cap:
                issues.append(f"Region {region}: exposure {exposure:.1%} > cap {cap:.1%}")

        # Type concentration
        type_exposure: Dict[str, float] = {}
        for pos in positions:
            stype = pos.spread_type.value
            type_exposure[stype] = type_exposure.get(stype, 0) + abs(pos.weight)

        for stype, exposure in type_exposure.items():
            cap = self._type_caps.get(stype, 0.30)
            if exposure > cap:
                issues.append(f"Type {stype}: exposure {exposure:.1%} > cap {cap:.1%}")

        return issues

    def _trim_to_constraints(self, positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
        """Trim positions to meet all constraints (proportional reduction)."""
        # Simple: scale all positions down by 20% if constraints breached
        return [
            PortfolioPosition(
                position_id=p.position_id,
                spread_id=p.spread_id,
                spread_type=p.spread_type,
                leg1_instrument=p.leg1_instrument,
                leg2_instrument=p.leg2_instrument,
                direction=p.direction,
                notional=p.notional * 0.8,
                dv01_net=p.dv01_net * 0.8,
                convexity_net=p.convexity_net * 0.8,
                entry_spread_bps=p.entry_spread_bps,
                current_spread_bps=p.current_spread_bps,
                entry_zscore=p.entry_zscore,
                current_zscore=p.current_zscore,
                unrealized_pnl_bps=p.unrealized_pnl_bps,
                weight=p.weight * 0.8,
                leverage_contribution=p.leverage_contribution * 0.8,
            )
            for p in positions
        ]

    def _infer_region(self, symbol: str) -> str:
        """Infer region from instrument symbol."""
        intl = {"BWX": "INTL", "BNDX": "INTL", "EMB": "EM", "IGLT.L": "GB"}
        return intl.get(symbol, "US")

    # ── Audit Log ────────────────────────────────────────────

    def _log_decision(
        self,
        agent: str,
        action: str,
        decision: str,
        rationale: str,
        approved: bool = True,
        risk_metrics: Optional[Dict] = None,
    ) -> None:
        """Log every decision for audit trail."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            agent=agent,
            action=action,
            decision=decision,
            rationale=rationale,
            risk_metrics=risk_metrics,
            approved=approved,
        )
        self._audit_log.append(entry)

    @property
    def audit_log(self) -> List[AuditEntry]:
        return self._audit_log

    @property
    def recent_audit(self) -> List[Dict]:
        """Last 50 audit entries as dicts."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "agent": e.agent,
                "action": e.action,
                "decision": e.decision,
                "rationale": e.rationale,
                "approved": e.approved,
            }
            for e in self._audit_log[-50:]
        ]


# ── Singleton ────────────────────────────────────────────────────
_governance: Optional[GovernanceLayer] = None


def get_governance() -> GovernanceLayer:
    global _governance
    if _governance is None:
        _governance = GovernanceLayer()
    return _governance
