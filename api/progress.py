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
        self.data_paths: Dict[str, str] = {}
        self.events: asyncio.Queue = asyncio.Queue()

    async def emit(self, event: Dict[str, Any]):
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
