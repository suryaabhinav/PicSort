import { useState } from "react";
import { post } from "../api/client";
import { openRunEvents } from "../api/sse";
import { useRunStore } from "../state/runStore";
import { MOCK } from "../config";

export default function ConfigForm() {
    const [root, setRoot] = useState("/path/to/images");
    const [batchSize, setBatch] = useState(16);
    const [runFaces, setRunFaces] = useState(true);

    const { setRunId, handleEvent, setES, reset } = useRunStore();

    async function start() {
        reset();
        const { run_id } = await post<{ run_id: string }>("/api/pipeline/start", {
            root, batch_size: batchSize, run_faces: runFaces,
        });
        setRunId(run_id);

        if (MOCK) {
            const startTime = Date.now();
            const DURATION = 15000;
            const es = { close() { clearInterval((es as any)._t); } } as unknown as EventSource;

            (es as any)._t = setInterval(() => {
                const elapsed = Date.now() - startTime;
                const p = Math.min(1, elapsed / DURATION);

                if (p < 1) {
                    handleEvent({ event: "progress", stage: "Processing...", progress: p });
                } else {
                    handleEvent({ event: "progress", stage: "Finalizing", progress: 1 });
                    handleEvent({ event: "analytics", data: { total: { images: 30, with_people: 20, without_people: 10 }, final_groups: { "10_People/Person_1": 12, "20_Landscape/Scene_0": 4 } } });
                    handleEvent({ event: "done", ok: true });
                    clearInterval((es as any)._t);
                }
            }, 500);
            setES(es);
        } else {
            const es = openRunEvents(run_id, handleEvent);
            setES(es);
        }
    }

    return (
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); start(); }}>
            <div className="space-y-1">
                <label className="text-sm text-gray-300">Root Folder</label>
                <input className="input" value={root} onChange={e => setRoot(e.target.value)} placeholder="/Volumes/Photos" />
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                    <label className="text-sm text-gray-300">Batch Size</label>
                    <input type="number" className="input" value={batchSize} onChange={e => setBatch(Number(e.target.value))} />
                </div>
                <div className="flex items-end gap-2">
                    <input id="faces" type="checkbox" className="checkbox" checked={runFaces} onChange={e => setRunFaces(e.target.checked)} />
                    <label htmlFor="faces" className="text-sm text-gray-300">Run Faces</label>
                </div>
            </div>

            <div className="flex items-center gap-2">
                <button type="submit" className="btn btn-primary">Start Run</button>
                <span className="text-xs text-gray-400">Sends request to /api/pipeline/start</span>
            </div>
        </form>
    );
}
