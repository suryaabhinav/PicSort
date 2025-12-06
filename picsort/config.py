from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeContext:
    device_str: str


@dataclass
class Models:
    yolo_seg_model: Optional[object] = "yolo11m-seg.pt"
    yolo_face_model: Optional[object] = None
    retina_detect_fn: Optional[object] = None
    retina_model: Optional[object] = None
    mtcnn: Optional[object] = None
    mtcnn_cpu: Optional[object] = None
    facenet: Optional[object] = None
    openclip_model: Optional[object] = None
    openclip_preprocess: Optional[object] = None


@dataclass
class FocusConfig:
    t_subj: float = 32.0
    t_bg: float = 8.0
    closeness: float = 5.0
    multiscale_level: int = 3


@dataclass
class YoloConfig:
    imgsz: int = 640
    batch_size: int = 32
    seg_conf: float = 0.4
    person_conf: float = 0.6
    device: str = "mps"
    # seg_model: str = "yolo11m-seg.pt"


@dataclass
class FaceConfig:
    sim_tresh: float = 0.65
    knn_k: int = 10
    min_cluster_size: int = 2
    fast_no_identity: bool = False
    batch_size: int = 32
    weights_path: Optional[str] = "./yolov8n-face-lindevs.pt"
    conf: float = 0.6
    max_det: int = 300
    iou: float = 0.5


@dataclass
class AppConfig:
    root: str
    focus: FocusConfig = FocusConfig()
    yolo: YoloConfig = YoloConfig()
    face: FaceConfig = FaceConfig()
