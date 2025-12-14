import ConfigForm from "./components/ConfigForm";
import RunPanel from "./components/RunPanel";
import AnalyticsCharts from "./components/AnalyticsCharts";
import MoveBar from "./components/MoveBar";

export default function App() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-white/5 bg-surface/80 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-white/90" />
            <h1 className="text-lg font-semibold">PicSort</h1>
            <span className="ml-3 text-xs rounded-lg bg-white/10 px-2 py-1 text-gray-300">v1</span>
          </div>
          <div className="text-xs text-gray-400">FastAPI • SSE • React</div>
        </div>
        {/* <button
          onClick={() => document.documentElement.classList.toggle('dark')}
          className="btn btn-ghost text-xs"
        >
          Toggle theme
        </button> */}
      </header>

      {/* Content grid */}
      <main className="mx-auto max-w-7xl px-4 py-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-1">
            <section className="card">
              <div className="card-body">
                <div className="section-title mb-3">Configuration</div>
                <ConfigForm />
              </div>
            </section>

            <section className="card">
              <div className="card-body">
                <div className="section-title mb-3">Run</div>
                <RunPanel />
              </div>
            </section>
          </div>

          <div className="space-y-6 lg:col-span-2">
            <section className="card">
              <div className="card-body">
                <div className="section-title mb-3">Analytics</div>
                <AnalyticsCharts />
              </div>
            </section>

            <section className="card">
              <div className="card-body">
                <div className="section-title mb-3">Actions</div>
                <MoveBar />
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
