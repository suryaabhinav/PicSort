import { http, HttpResponse } from "msw";
import { API_BASE } from "../api/client";

let LAST_RUN_ID = "mock-run-001";
let started = false;

export const handlers = [
    http.post(`${API_BASE}/api/pipeline/start`, async () => {
        started = true;
        LAST_RUN_ID = `mock-run-${Math.random().toString(36).slice(2, 7)}`;
        return HttpResponse.json({ run_id: LAST_RUN_ID });
    }),

    http.get(`${API_BASE}/api/runs/:id/status`, async ({ params }) => {
        const prog = started ? Math.min(1, (Date.now() % 20000) / 20000) : 0;
        return HttpResponse.json({
            run_id: params.id,
            stage: prog < 1 ? "stage_b" : "done",
            progress: prog,
            message: prog < 1 ? "Running mock..." : "All done",
            error: null,
            timings: {},
            paths: { final: `/runs/${params.id}/final.parquet` },
            root: "/mock/root"
        });
    }),

    http.post(`${API_BASE}/api/move/:id`, async () => {
        return HttpResponse.json({ ok: true, moved: { "10_People/Person_1": 12, "20_Landscape/Scene_0": 4 } });
    }),

    http.post(`${API_BASE}/api/pipeline/stop/:id`, async () => {
        started = false;
        return HttpResponse.json({ ok: true });
    }),
]