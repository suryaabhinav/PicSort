import { post } from "../api/client";
import { useRunStore } from "../state/runStore";
import { useState } from "react";

export default function MoveBar() {
    const { runId } = useRunStore();
    const [dryRun, setDry] = useState(true);
    const [resp, setResp] = useState<any>(null);

    const [error, setError] = useState<string | null>(null);

    async function doMove() {
        if (!runId) return;
        setError(null);
        setResp(null);
        try {
            const r = await post(`/api/move/${runId}?dry_run=${dryRun}`);
            setResp(r);
        } catch (e: any) {
            setError(e.message || "Failed to move files");
        }
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" className="checkbox" checked={dryRun} onChange={e => setDry(e.target.checked)} />
                    Dry run
                </label>
                <button className="btn btn-primary" onClick={doMove}>
                    Move files
                </button>
            </div>
            {error && (
                <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-xs text-red-200">
                    <div className="font-bold mb-1">Error</div>
                    {error}
                </div>
            )}
            {resp && (
                <div className="rounded-xl border border-white/5 bg-white/5 p-3 text-xs text-gray-200">
                    <div className="mb-1 text-gray-400">Move result</div>
                    <pre className="whitespace-pre-wrap">{JSON.stringify(resp, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}
