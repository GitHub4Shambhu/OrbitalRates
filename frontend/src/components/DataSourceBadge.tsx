"use client";

export default function DataSourceBadge({ source }: { source: string }) {
  const isLive = source === "live";

  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded-full border ${
      isLive
        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
        : "bg-amber-500/10 border-amber-500/30 text-amber-400"
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
      {isLive ? "LIVE DATA" : "MOCK DATA"}
    </span>
  );
}
