"""
Video input validation and basic metadata extraction.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2

from .utils import setup_logger

logger = setup_logger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class VideoMetadata:
    filename: str
    filepath: str
    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


class InvalidVideoError(Exception):
    """Raised when the input file is missing, unsupported, or unreadable."""


def validate_video_file(video_path: str) -> Path:
    """Cheap existence/extension checks before we try to actually open it."""
    path = Path(video_path)
    if not path.exists():
        raise InvalidVideoError(f"File does not exist: {video_path}")
    if not path.is_file():
        raise InvalidVideoError(f"Not a file: {video_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Extension '%s' is not in the known list %s — will still try to open it.",
            path.suffix, sorted(SUPPORTED_EXTENSIONS),
        )
    return path


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InvalidVideoError(f"OpenCV could not open video (corrupt or unsupported codec): {video_path}")
    return cap


def extract_metadata(video_path: Path, cap: cv2.VideoCapture) -> VideoMetadata:
    """Pull duration / fps / resolution / frame count out of an opened capture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        logger.warning("FPS reported as 0 by OpenCV; defaulting to 25.0 for duration/timestamp math.")
        fps = 25.0

    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0 or width <= 0 or height <= 0:
        raise InvalidVideoError(
            f"Video appears corrupted or unreadable (frames={total_frames}, size={width}x{height})."
        )

    return VideoMetadata(
        filename=video_path.name,
        filepath=str(video_path.resolve()),
        duration_sec=round(duration_sec, 3),
        fps=round(fps, 3),
        width=width,
        height=height,
        total_frames=total_frames,
    )
