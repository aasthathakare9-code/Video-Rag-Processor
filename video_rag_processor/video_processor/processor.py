"""
VideoProcessor: end-to-end video -> keyframes + metadata pipeline.

    Video -> Read metadata -> Sample frames (time-based + scene-change)
          -> Remove near-duplicates -> Save frames + timestamps + metadata.json

The output of this module (frames on disk + metadata.json with exact
timestamps) is designed to be consumed directly by a downstream
CLIP/SigLIP embedding stage.
"""
import json
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import ProcessingConfig
from .metadata_extractor import (
    validate_video_file,
    open_video_capture,
    extract_metadata,
    VideoMetadata,
)
from .scene_detector import get_scene_detector
from .dedup import FrameDeduplicator
from .utils import setup_logger, seconds_to_timestamp, seconds_to_timestamp_ms, sanitize_filename


class VideoProcessor:
    """
    Reusable, configurable class that extracts representative keyframes
    (with precise timestamps and metadata) from a single video.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = setup_logger("VideoProcessor", self.config.log_level)
        self.scene_detector = get_scene_detector(
            self.config.scene_detector_method, self.config.scene_change_threshold
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, video_path: str) -> dict:
        """Run the full pipeline on one video. Returns the same dict that is
        written to <output_root>/<video_name>/metadata.json."""
        start_time = time.time()

        path = validate_video_file(video_path)
        video_name = sanitize_filename(path.name)
        self.logger.info("Processing video: %s", path)

        cap = open_video_capture(path)
        try:
            video_meta = extract_metadata(path, cap)
            self.logger.info(
                "Metadata: duration=%.2fs fps=%.2f size=%dx%d frames=%d",
                video_meta.duration_sec, video_meta.fps,
                video_meta.width, video_meta.height, video_meta.total_frames,
            )

            frames_dir = self.config.frames_dir_for(video_name)
            frames_dir.mkdir(parents=True, exist_ok=True)

            frame_records = self._extract_frames(cap, video_meta, frames_dir)
        finally:
            cap.release()

        elapsed = time.time() - start_time
        result = {
            "video_metadata": video_meta.to_dict(),
            "processing_config": {
                "sampling_interval_sec": self.config.sampling_interval_sec,
                "scene_change_threshold": self.config.scene_change_threshold,
                "scene_detector_method": self.config.scene_detector_method,
                "dedup_hamming_threshold": self.config.dedup_hamming_threshold,
                "max_frame_width": self.config.max_frame_width,
            },
            "num_frames_extracted": len(frame_records),
            "processing_time_sec": round(elapsed, 3),
            "frames": frame_records,
        }

        self._write_metadata(video_name, result)
        self.logger.info(
            "Done. Extracted %d keyframes in %.2fs -> %s",
            len(frame_records), elapsed, self.config.output_dir_for(video_name),
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal: main frame-scanning loop
    # ------------------------------------------------------------------ #
    def _extract_frames(
        self,
        cap: cv2.VideoCapture,
        video_meta: VideoMetadata,
        frames_dir: Path,
    ) -> List[dict]:
        fps = video_meta.fps
        interval_frames = max(1, int(round(self.config.sampling_interval_sec * fps)))
        min_scene_gap_frames = max(1, int(round(self.config.min_scene_gap_sec * fps)))

        deduplicator = FrameDeduplicator(
            hash_size=self.config.dedup_hash_size,
            hamming_threshold=self.config.dedup_hamming_threshold,
            compare_last_n=self.config.dedup_compare_last_n,
        )

        frame_records: List[dict] = []
        prev_signature = None
        last_saved_frame_idx = -min_scene_gap_frames
        next_interval_target = 0
        consecutive_failures = 0
        frame_idx = 0
        saved_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                # Distinguish "corrupted frame" from "reached end of video"
                if frame_idx >= video_meta.total_frames - 1:
                    break
                consecutive_failures += 1
                self.logger.warning("Unreadable/corrupted frame at index %d (skipping).", frame_idx)
                frame_idx += 1
                if consecutive_failures >= self.config.max_consecutive_read_failures:
                    self.logger.error(
                        "Too many consecutive unreadable frames (%d). Stopping early.",
                        consecutive_failures,
                    )
                    break
                continue
            consecutive_failures = 0

            try:
                signature = self.scene_detector.compute_signature(frame)
                is_scene_change = self.scene_detector.is_scene_change(prev_signature, signature)
                prev_signature = signature

                trigger = None
                if frame_idx >= next_interval_target:
                    trigger = "interval"
                    next_interval_target = frame_idx + interval_frames
                elif is_scene_change and (frame_idx - last_saved_frame_idx) >= min_scene_gap_frames:
                    trigger = "scene_change"

                if trigger is not None:
                    frame_hash = deduplicator.check_and_register(frame)
                    if frame_hash is None:
                        self.logger.debug("Frame %d skipped as near-duplicate.", frame_idx)
                    else:
                        record = self._save_frame(frame, frame_idx, fps, frames_dir, trigger, frame_hash)
                        frame_records.append(record)
                        last_saved_frame_idx = frame_idx
                        saved_count += 1
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
                self.logger.error("Error processing frame %d: %s", frame_idx, exc)

            frame_idx += 1

        self.logger.info("Scanned %d frames, saved %d keyframes.", frame_idx, saved_count)
        return frame_records

    # ------------------------------------------------------------------ #
    # Internal: saving frames + writing metadata.json
    # ------------------------------------------------------------------ #
    def _save_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        frames_dir: Path,
        trigger: str,
        frame_hash,
    ) -> dict:
        timestamp_sec = frame_idx / fps
        timestamp_str = seconds_to_timestamp(timestamp_sec)
        timestamp_ms_str = seconds_to_timestamp_ms(timestamp_sec)

        out_frame = self._resize_if_needed(frame)

        safe_ts = timestamp_str.replace(":", "-")
        filename = f"frame_{frame_idx:07d}_t{safe_ts}.jpg"
        out_path = frames_dir / filename

        cv2.imwrite(
            str(out_path), out_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )

        return {
            "frame_filename": filename,
            "frame_path": str(out_path),
            "frame_index": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "timestamp": timestamp_str,          # HH:MM:SS  (e.g. for "found at 01:23:42")
            "timestamp_precise": timestamp_ms_str,  # HH:MM:SS.mmm
            "trigger": trigger,                  # "interval" | "scene_change"
            "width": out_frame.shape[1],
            "height": out_frame.shape[0],
            "phash": str(frame_hash),
        }

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        max_w = self.config.max_frame_width
        if not max_w:
            return frame
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / w
        new_size = (max_w, int(round(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _write_metadata(self, video_name: str, result: dict) -> None:
        meta_path = self.config.metadata_path_for(video_name)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        self.logger.info("Metadata written to %s", meta_path)
