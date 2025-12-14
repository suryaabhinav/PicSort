import asyncio
import time
import uuid
from typing import Any, Dict, Optional


class RunState:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.error: Optional[str] = None
        self.timings: Dict[str, float] = {}
        self.paths: Dict[str, str] = {}
        self.root: Optional[str] = None
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.events: asyncio.Queue = asyncio.Queue()
        self.bg_task: Optional[asyncio.Task] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def emit_threadsafe(self, event: Dict[str, Any]) -> None:
        if not self.loop:
            raise RuntimeError("RunState.loop not initialized")

        def _put():
            self.events.put_nowait(event)

        self.loop.call_soon_threadsafe(_put)

    async def emit(self, event: Dict[str, Any]) -> None:
        await self.events.put(event)


class RunRegistry:
    def __init__(self):
        self._runs: Dict[str, RunState] = {}

    def create(self) -> RunState:
        run = RunState()
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)


registry = RunRegistry()
