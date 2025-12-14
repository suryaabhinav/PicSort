from pydantic import BaseModel


class StartRunRequest(BaseModel):
    root: str
    batch_size: int | None = None
    dry_run: bool = True
    run_faces: bool = True
