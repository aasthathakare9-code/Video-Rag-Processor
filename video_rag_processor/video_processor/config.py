"""
Configuration for the Video Processing & Keyframe Extraction module.

All tunable knobs live here so behaviour can be changed without touching
pipeline code.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    # --- Time-based sampling ---
    sampling_interval_sec: float = 1.0          # extract a frame at least this often
    min_scene_gap_sec: float = 0.5              # min gap between two scene-change saves

    # --- Scene / shot change detection ---
    scene_change_threshold: float = 0.55        # histogram correlation cutoff (0..1)
    scene_detector_method: str = "histogram"    # pluggable: only "histogram" for now

    # --- Duplicate / near-duplicate filtering ---
    dedup_hash_size: int = 8                    # perceptual hash grid size
    dedup_hamming_threshold: int = 5            # <= this distance => treated as duplicate
    dedup_compare_last_n: int = 3               # compare new frame against last N saved

    # --- Output layout ---
    output_root: str = "processed_videos"
    frame_subdir: str = "frames"
    metadata_filename: str = "metadata.json"

    # --- Saved frame quality ---
    max_frame_width: int = 720                  # resize saved frames (aspect preserved); None disables
    jpeg_quality: int = 92

    # --- Robustness ---
    max_consecutive_read_failures: int = 30     # abort early if this many unreadable frames in a row

    # --- Logging ---
    log_level: str = "INFO"

    def output_dir_for(self, video_name: str) -> Path:
        return Path(self.output_root) / video_name

    def frames_dir_for(self, video_name: str) -> Path:
        return self.output_dir_for(video_name) / self.frame_subdir

    def metadata_path_for(self, video_name: str) -> Path:
        return self.output_dir_for(video_name) / self.metadata_filename
