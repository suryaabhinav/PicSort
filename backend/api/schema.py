from pydantic import BaseModel


class StartRunRequest(BaseModel):
    root: str
    batch_size: int | None = None
    dry_run: bool = True
    run_faces: bool = True

    # Analysis Parameters
    focus_t_subj: float | None = None
    focus_t_bg: float | None = None
    yolo_person_conf: float | None = None
    face_conf: float | None = None
    face_sim_tresh: float | None = None
