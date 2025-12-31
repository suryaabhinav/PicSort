import ReactECharts from "echarts-for-react";
import { useRunStore } from "../state/runStore";

export default function AnalyticsCharts() {
    const analytics = useRunStore(s => s.analytics);
    if (!analytics) return <div className="text-sm text-gray-500">Run to see analytics.</div>;

    const fg = analytics.final_groups ?? {};
    const fgNames = Object.keys(fg);
    const fgVals = Object.values(fg);

    const totals = analytics.total ?? {};

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-3 gap-3">
                <div className="kpi">
                    <div className="text-[11px] text-gray-400">Images</div>
                    <div className="text-sm">{totals.images ?? "-"}</div>
                </div>
                <div className="kpi">
                    <div className="text-[11px] text-gray-400">With People</div>
                    <div className="text-sm">{totals.with_people ?? "-"}</div>
                </div>
                <div className="kpi">
                    <div className="text-[11px] text-gray-400">Without People</div>
                    <div className="text-sm">{totals.without_people ?? "-"}</div>
                </div>
            </div>

            <ReactECharts
                option={{
                    title: { text: "Final Groups", left: "center", textStyle: { color: "#e5e7eb", fontSize: 14 } },
                    tooltip: {},
                    grid: { left: 24, right: 12, bottom: 24, top: 48 },
                    xAxis: { type: "category", data: fgNames, axisLabel: { color: "#9ca3af", rotate: 20 } },
                    yAxis: { type: "value", axisLabel: { color: "#9ca3af" }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } } },
                    series: [{ type: "bar", data: fgVals, barWidth: "50%" }],
                }}
                style={{ height: 320 }}
            />
        </div>
    );
}
