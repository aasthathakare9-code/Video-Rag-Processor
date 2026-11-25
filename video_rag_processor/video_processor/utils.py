"""
Shared utilities: logging setup, timestamp formatting, filename sanitizing.
"""
import logging
import re
import sys
from pathlib import Path


def setup_logger(name: str = "video_processor", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS (zero-padded, second precision)."""
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, _ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def seconds_to_timestamp_ms(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS.mmm (millisecond precision)."""
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def sanitize_filename(name: str) -> str:
    """Turn a video filename into a safe folder/name component."""
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return stem or "video"
