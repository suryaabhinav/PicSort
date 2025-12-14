export type RunEvent =
    | { event: "hello"; run_id: string }
    | { event: "stage" | "progress"; stage: string; progress: number; msg?: string }
    | { event: "analytics"; data: any }
    | { event: "done"; ok: boolean; analytics?: any }
    | { event: "error"; error: string }

export function openRunEvents(runId: string, onEvent: (ev: RunEvent) => void) {
    const es = new EventSource(`/api/runs/${runId}/events`);
    const handler = (type: string) => (e: MessageEvent) => onEvent({ ...(JSON.parse(e.data)), event: type } as any);
    ["hello", "stage", "progress", "analytics", "done", "error"].forEach(t => es.addEventListener(t, handler(t)));
    return es
}