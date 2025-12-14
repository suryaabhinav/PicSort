import { useState } from "react";
import { useRunStore } from "../state/runStore";
import { post } from "../api/client";

export default function RunPanel() {
    const { runId, stage, progress, events, stopListening } = useRunStore();
    const [stopping, setStopping] = useState(false);

    async function stop() {
        if (!runId) return;
        setStopping(true);
        try {
            await post(`/api/pipeline/stop/${runId}`);
        } catch {
            // error handling if needed
        } finally {
            stopListening();
            setStopping(false);
        }
    }

    // Circular Progress Component
    const size = 120;
    const strokeWidth = 8;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (progress * circumference);

    return (
        <div className="space-y-6">
            {!runId && <div className="text-sm text-gray-500 italic text-center py-8">Ready to start new run...</div>}

            {runId && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="flex items-start gap-6">
                        {/* Circular Progress */}
                        <div className="relative flex-shrink-0">
                            <svg width={size} height={size} className="transform -rotate-90">
                                <circle
                                    stroke="currentColor"
                                    strokeWidth={strokeWidth}
                                    fill="transparent"
                                    r={radius}
                                    cx={size / 2}
                                    cy={size / 2}
                                    className="text-white/5"
                                />
                                <circle
                                    stroke="currentColor"
                                    strokeWidth={strokeWidth}
                                    fill="transparent"
                                    r={radius}
                                    cx={size / 2}
                                    cy={size / 2}
                                    strokeDasharray={circumference}
                                    strokeDashoffset={offset}
                                    strokeLinecap="round"
                                    className="text-blue-500 transition-all duration-300 ease-out drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]"
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-2xl font-bold bg-gradient-to-br from-white to-gray-400 bg-clip-text text-transparent">
                                    {Math.round(progress * 100)}%
                                </span>
                                <span className="text-[10px] uppercase tracking-wider text-gray-500 mt-0.5">Progress</span>
                            </div>
                        </div>

                        {/* Stats & Actions */}
                        <div className="flex-1 space-y-4">
                            <div className="grid grid-cols-2 gap-3">
                                <div className="kpi">
                                    <div className="text-[11px] uppercase tracking-wider text-gray-500 mb-1">Current Stage</div>
                                    <div className="font-medium text-blue-200 truncate">{stage || "Initializing..."}</div>
                                </div>
                                <div className="kpi">
                                    <div className="text-[11px] uppercase tracking-wider text-gray-500 mb-1">Run ID</div>
                                    <div className="font-mono text-xs text-gray-400 truncate" title={runId}>{runId}</div>
                                </div>
                            </div>

                            <button
                                onClick={stop}
                                disabled={!stage || stage === "done" || stopping}
                                className="w-full relative group overflow-hidden rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-red-500 transition-all hover:bg-red-500/20 hover:border-red-500/40 hover:shadow-[0_0_20px_rgba(239,68,68,0.2)] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <span className="relative z-10 flex items-center justify-center gap-2 font-medium">
                                    {stopping ? (
                                        <>
                                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                                            Stopping...
                                        </>
                                    ) : (
                                        <>
                                            <div className="h-2 w-2 rounded-full bg-red-500 shadow-[0_0_8px_currentColor]" />
                                            Stop Process
                                        </>
                                    )}
                                </span>
                            </button>
                        </div>
                    </div>

                    {/* Terminal Log */}
                    <div className="mt-6 rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-xs shadow-inner">
                        <div className="flex items-center gap-2 mb-3 border-b border-white/5 pb-2">
                            <div className="flex gap-1.5">
                                <div className="h-2.5 w-2.5 rounded-full bg-red-500/20" />
                                <div className="h-2.5 w-2.5 rounded-full bg-yellow-500/20" />
                                <div className="h-2.5 w-2.5 rounded-full bg-green-500/20" />
                            </div>
                            <span className="text-gray-500 ml-2">Process Logs</span>
                        </div>
                        <div className="h-48 overflow-y-auto space-y-1 pr-2 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                            {events.length === 0 ? (
                                <span className="text-gray-600 italic">Waiting for logs...</span>
                            ) : (
                                events.map((e, i) => (
                                    <div key={i} className="flex gap-3 text-gray-300 animate-in fade-in duration-300">
                                        <span className="text-gray-600 select-none w-6 text-right">{i + 1}</span>
                                        <span>
                                            <span className="text-blue-400">[{e.event}]</span>{" "}
                                            {(e.event === "stage" || e.event === "progress") ? (
                                                <>
                                                    {e.stage} <span className="text-gray-500">|</span> {(e.progress * 100).toFixed(1)}%
                                                </>
                                            ) : (
                                                JSON.stringify(e)
                                            )}
                                        </span>
                                    </div>
                                ))
                            )}
                            <div id="log-end" />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
