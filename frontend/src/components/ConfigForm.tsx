import { useState } from "react";
import { post } from "../api/client";
import { openRunEvents } from "../api/sse";
import { useRunStore } from "../state/runStore";
import { MOCK } from "../config";

export default function ConfigForm() {
    const [root, setRoot] = useState("/path/to/images");
    const [batchSize, setBatch] = useState(16);
    const [runFaces, setRunFaces] = useState(true);

    const [error, setError] = useState<string | null>(null);
    const [createdRunId, setCreatedRunId] = useState<string | null>(null);

    const { setRunId, handleEvent, setES, reset } = useRunStore();

    const [showAdvanced, setShowAdvanced] = useState(false);
    const [params, setParams] = useState({
        focus_t_subj: 10.0,
        focus_t_bg: 8.0,
        yolo_person_conf: 0.6,
        face_conf: 0.6,
        face_sim_tresh: 0.65,
    });

    const updateParam = (key: keyof typeof params, val: any) => setParams(p => ({ ...p, [key]: Number(val) }));

    async function start() {
        reset();
        setError(null);
        setCreatedRunId(null);

        try {
            const { run_id } = await post<{ run_id: string }>("/api/pipeline/start", {
                root,
                batch_size: batchSize,
                run_faces: runFaces,
                ...params // Spread advanced params
            });
            setRunId(run_id);
            setCreatedRunId(run_id);

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
        } catch (e: any) {
            setError(e.message || "Failed to start run");
        }
    }

    return (
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); start(); }}>
            {error && (
                <div role="alert" className="rounded border-s-4 border-red-500 bg-red-50 p-4 dark:border-red-600 dark:bg-red-900/30">
                    <strong className="block font-medium text-red-800 dark:text-red-100"> Something went wrong </strong>
                    <p className="mt-2 text-sm text-red-700 dark:text-red-200">{error}</p>
                </div>
            )}

            {createdRunId && (
                <div role="alert" className="rounded border-s-4 border-green-500 bg-green-50 p-4 dark:border-green-600 dark:bg-green-900/30">
                    <strong className="block font-medium text-green-800 dark:text-green-100"> Run Started Successfully </strong>
                    <p className="mt-2 text-sm text-green-700 dark:text-green-200">Run ID: <span className="font-mono">{createdRunId}</span></p>
                </div>
            )}

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

            <div className="pt-2">
                <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-2 text-xs text-gray-400 hover:text-gray-200 transition-colors">
                    <span className={showAdvanced ? "transform rotate-90" : ""}>▶</span>
                    Advanced Analysis Settings
                </button>

                {showAdvanced && (
                    <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-xl bg-white/5 animate-in fade-in slide-in-from-top-2">
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <label className="text-gray-300">Subject Sharpness</label>
                                <span className="text-gray-500">{params.focus_t_subj}</span>
                            </div>
                            <input type="range" min="1" max="50" step="0.5" className="range range-xs range-primary" value={params.focus_t_subj} onChange={e => updateParam('focus_t_subj', e.target.value)} />
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <label className="text-gray-300">Background Sharpness</label>
                                <span className="text-gray-500">{params.focus_t_bg}</span>
                            </div>
                            <input type="range" min="1" max="50" step="0.5" className="range range-xs range-secondary" value={params.focus_t_bg} onChange={e => updateParam('focus_t_bg', e.target.value)} />
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <label className="text-gray-300">Person Confirm. Thresh</label>
                                <span className="text-gray-500">{params.yolo_person_conf}</span>
                            </div>
                            <input type="range" min="0.1" max="0.95" step="0.05" className="range range-xs range-accent" value={params.yolo_person_conf} onChange={e => updateParam('yolo_person_conf', e.target.value)} />
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                                <label className="text-gray-300">Face Confirm. Thresh</label>
                                <span className="text-gray-500">{params.face_conf}</span>
                            </div>
                            <input type="range" min="0.1" max="0.95" step="0.05" className="range range-xs range-info" value={params.face_conf} onChange={e => updateParam('face_conf', e.target.value)} />
                        </div>
                        <div className="col-span-1 md:col-span-2 space-y-1">
                            <div className="flex justify-between text-xs">
                                <label className="text-gray-300">Face Identity Similarity (Strictness)</label>
                                <span className="text-gray-500">{params.face_sim_tresh}</span>
                            </div>
                            <input type="range" min="0.3" max="0.9" step="0.01" className="range range-xs range-warning" value={params.face_sim_tresh} onChange={e => updateParam('face_sim_tresh', e.target.value)} />
                            <div className="flex justify-between text-[10px] text-gray-500 px-1">
                                <span>Loose (Merge more)</span>
                                <span>Strict (Split more)</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="flex items-center gap-2 pt-2 border-t border-white/5">
                <button type="submit" className="btn btn-primary">Start Run</button>
                <div className="text-xs text-gray-400">
                    {MOCK && <span className="mr-2 rounded bg-yellow-100 px-2 py-0.5 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300">MOCK MODE</span>}
                    Sends request to /api/pipeline/start
                </div>
            </div>
        </form>
    );
}
