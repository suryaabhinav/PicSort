import { post } from "../api/client";
import { useRunStore } from "../state/runStore";
import { useState } from "react";

export default function MoveBar() {
    const { runId, stage } = useRunStore();
    const [dryRun, setDry] = useState(true);
    const [resp, setResp] = useState<any>(null);

    const disabled = !runId || stage !== "done";

    async function doMove() {
        if (!runId) return;
        const r = await post(`/api/move/${runId}?dry_run=${dryRun}`);
        setResp(r);
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" className="checkbox" checked={dryRun} onChange={e => setDry(e.target.checked)} />
                    Dry run
                </label>
                <button className="btn btn-primary disabled:opacity-50" disabled={disabled} onClick={doMove}>
                    Move files
                </button>
            </div>
            {resp && (
                <div className="rounded-xl border border-white/5 bg-white/5 p-3 text-xs text-gray-200">
                    <div className="mb-1 text-gray-400">Move result</div>
                    <pre className="whitespace-pre-wrap">{JSON.stringify(resp, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}
