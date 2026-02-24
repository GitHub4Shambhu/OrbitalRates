"use client";

import { useState } from "react";
import { AuditEntry, activateKillSwitch, deactivateKillSwitch } from "@/lib/api";

export default function GovernancePanel({
  isHalted,
  auditLog,
  onRefresh,
}: {
  isHalted: boolean;
  auditLog: AuditEntry[];
  onRefresh: () => void;
}) {
  const [reason, setReason] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleKillSwitch() {
    if (!reason.trim()) return;
    setLoading(true);
    try {
      if (isHalted) {
        await deactivateKillSwitch(reason);
      } else {
        await activateKillSwitch(reason);
      }
      onRefresh();
    } finally {
      setLoading(false);
      setReason("");
    }
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold tracking-wider text-neutral-400 uppercase">
          Governance & Audit
        </h2>
        <span className={`text-xs px-3 py-1 rounded-full font-bold ${
          isHalted
            ? "bg-red-500/20 border border-red-500/50 text-red-400 crisis-pulse"
            : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400"
        }`}>
          {isHalted ? "⚠ HALTED" : "● LIVE"}
        </span>
      </div>

      {/* Kill Switch */}
      <div className="p-4 rounded-lg bg-neutral-900/50 border border-neutral-800 mb-5">
        <p className="text-xs text-neutral-400 mb-2">Manual Kill Switch</p>
        <div className="flex gap-2">
          <input
            type="text"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason for action..."
            className="flex-1 px-3 py-1.5 rounded bg-neutral-800 border border-neutral-700 text-xs text-neutral-200 placeholder-neutral-600 focus:border-neutral-500 focus:outline-none"
          />
          <button
            onClick={handleKillSwitch}
            disabled={loading || !reason.trim()}
            className={`px-4 py-1.5 rounded text-xs font-bold transition-colors disabled:opacity-40 ${
              isHalted
                ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                : "bg-red-600 hover:bg-red-500 text-white"
            }`}
          >
            {loading ? "..." : isHalted ? "RESUME" : "HALT"}
          </button>
        </div>
      </div>

      {/* Audit Log */}
      <div>
        <p className="text-xs text-neutral-500 mb-2">Recent Audit Trail</p>
        <div className="max-h-60 overflow-y-auto space-y-1.5 scrollbar-thin">
          {auditLog.length === 0 ? (
            <p className="text-xs text-neutral-600 text-center py-4">No audit entries</p>
          ) : (
            auditLog.slice(-20).reverse().map((entry, i) => (
              <div key={i} className={`p-2 rounded text-[10px] border ${
                entry.approved
                  ? "border-neutral-800/50 bg-neutral-900/20"
                  : "border-red-500/20 bg-red-500/5"
              }`}>
                <div className="flex justify-between mb-0.5">
                  <span className="text-neutral-400">{entry.agent}</span>
                  <span className="text-neutral-600">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="text-neutral-300">{entry.action}: {entry.decision}</p>
                {entry.rationale && (
                  <p className="text-neutral-500 mt-0.5">{entry.rationale}</p>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
