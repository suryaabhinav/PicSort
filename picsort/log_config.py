import logging
import sys
from pathlib import Path


def make_logger(script_path: Path) -> logging.Logger:
    log = logging.getLogger("picsort")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(ch)
        fh = logging.FileHandler(script_path.parent / "picsort_pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s")
        )
        log.addHandler(fh)
    return log
