# api/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name="picsort", log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(
        log_dir / "picsort.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Let uvicorn/fastapi still log to their own loggers
    logger.propagate = False
    return logger


log = get_logger()
