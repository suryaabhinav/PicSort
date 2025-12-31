import { create } from "zustand";
import type { RunEvent } from "../api/sse";

type RunState = {
    runId?: string;
    stage?: string;
    progress: number;
    analytics?: any;
    events: RunEvent[];
    es?: EventSource;
    setRunId: (id?: string) => void;
    handleEvent: (ev: RunEvent) => void;
    setES: (es?: EventSource) => void;
    stopListening: () => void;
    reset: () => void;
};

export const useRunStore = create<RunState>((set, get) => ({
    progress: 0,
    events: [],
    setRunId: (id) => set({ runId: id, progress: 0, stage: undefined, analytics: undefined, events: [] }),
    setES: (es) => set({ es }),
    handleEvent: (ev) => {
        const prev = get();
        const next: Partial<RunState> = { events: [...prev.events.slice(-199), ev] };
        if (ev.event === "stage" || ev.event === "progress") {
            next.stage = ev.stage; next.progress = ev.progress;
        }
        if (ev.event === "analytics") next.analytics = ev.data;
        if (ev.event === "result") next.analytics = ev.analytics;
        if (ev.event === "done") next.analytics = ev.analytics ?? prev.analytics;
        set(next);
    },
    stopListening: () => {
        const es = get().es; es?.close();
        set({ es: undefined });
    },
    reset: () => {
        const es = get().es; es?.close();
        set({ runId: undefined, stage: undefined, progress: 0, analytics: undefined, events: [], es: undefined });
    },
}));
