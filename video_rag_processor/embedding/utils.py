"""
Utility helpers: logging setup, timestamp formatting, metadata.json loading,
frame discovery (with timestamp linkage), and safe image loading.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Matches filenames like: frame_0000030_t00-00-01.jpg
FRAME_FILENAME_RE = re.compile(
    r"frame_(?P<frame_number>\d+)_t(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_seconds(hms: str) -> float:
    h, m, s = (int(part) for part in hms.split(":"))
    return h * 3600 + m * 60 + s


def parse_timestamp_from_filename(filename: str) -> Optional[Dict]:
    """Fallback parser: extract frame_number + timestamp straight from
    filenames like 'frame_0000030_t00-00-01.jpg'. Used only when metadata.json
    is missing or doesn't cover a given frame."""
    match = FRAME_FILENAME_RE.search(filename)
    if not match:
        return None
    frame_number = int(match.group("frame_number"))
    h, m, s = int(match.group("h")), int(match.group("m")), int(match.group("s"))
    timestamp_seconds = h * 3600 + m * 60 + s
    return {
        "frame_number": frame_number,
        "timestamp_seconds": float(timestamp_seconds),
        "timestamp_hms": f"{h:02d}:{m:02d}:{s:02d}",
    }


def load_video_metadata(metadata_path: str) -> Optional[dict]:
    """Load metadata.json produced by the Video Processing Module, if present."""
    if not metadata_path or not os.path.isfile(metadata_path):
        logger.warning(
            f"metadata.json not found at '{metadata_path}'. "
            f"Falling back to parsing timestamps from frame filenames."
        )
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def _index_metadata_frames(metadata: Optional[dict]) -> Dict[str, dict]:
    """
    Build a filename -> frame-info lookup from metadata.json.

    Tolerant of a few reasonable schema shapes, since the exact structure
    produced by the Video Processing Module isn't fixed here:
      - a top-level list of frame dicts
      - {"frames": [...]} / {"keyframes": [...]} / {"extracted_frames": [...]}
    Each frame dict is expected to carry a filename plus frame_number /
    timestamp fields under one of a few common key names.
    """
    if not metadata:
        return {}

    frames = None
    if isinstance(metadata, list):
        frames = metadata
    elif isinstance(metadata, dict):
        for key in ("frames", "keyframes", "extracted_frames"):
            if key in metadata and isinstance(metadata[key], list):
                frames = metadata[key]
                break

    if frames is None:
        return {}

    index = {}
    for entry in frames:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename") or entry.get("frame_filename") or entry.get("file_name")
        if not filename:
            continue
        filename = os.path.basename(filename)
        index[filename] = entry
    return index


def discover_frames(frames_dir: str, metadata_path: Optional[str] = None) -> List[Dict]:
    """
    Walk the frames directory and build a sorted list of frame records:
        {filename, path, frame_number, timestamp_seconds, timestamp_hms}

    Timestamp info is taken from metadata.json when available (the
    authoritative source from the Video Processing Module); otherwise it is
    parsed from the frame filename as a fallback. Timestamps are never
    recalculated from the image itself.
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    metadata = load_video_metadata(metadata_path) if metadata_path else None
    meta_index = _index_metadata_frames(metadata)

    records = []
    for filename in sorted(os.listdir(frames_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        path = os.path.join(frames_dir, filename)
        info = None

        if filename in meta_index:
            entry = meta_index[filename]
            frame_number = entry.get("frame_number", entry.get("frame_index"))
            # Prefer the numeric seconds field; only fall back to "timestamp"
            # if it isn't a string like "00:00:00".
            timestamp_seconds = entry.get("timestamp_seconds", entry.get("timestamp_sec"))
            if timestamp_seconds is None and not isinstance(entry.get("timestamp"), str):
                timestamp_seconds = entry.get("timestamp")
            timestamp_hms = entry.get("timestamp_hms", entry.get("timestamp"))

            if frame_number is not None and timestamp_seconds is not None:
                if not timestamp_hms:
                    timestamp_hms = seconds_to_hms(float(timestamp_seconds))
                info = {
                    "frame_number": int(frame_number),
                    "timestamp_seconds": float(timestamp_seconds),
                    "timestamp_hms": timestamp_hms,
                }

        if info is None:
            # Fall back to parsing the timestamp encoded in the filename.
            info = parse_timestamp_from_filename(filename)

        if info is None:
            logger.warning(f"Could not determine timestamp for '{filename}'. Skipping.")
            continue

        records.append({
            "filename": filename,
            "path": path,
            **info,
        })

    records.sort(key=lambda r: r["frame_number"])
    logger.info(f"Discovered {len(records)} frame(s) in '{frames_dir}'.")
    return records


def safe_load_image(path: str) -> Optional[Image.Image]:
    """Load an image and convert to RGB. Returns None (and logs a warning)
    for unreadable/corrupted files instead of raising, so a single bad frame
    doesn't kill an entire batch run."""
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Skipping unreadable/corrupted image '{path}': {e}")
        return None
