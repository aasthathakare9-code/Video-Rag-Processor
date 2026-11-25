"""
Smoke tests for the video processing pipeline.

Generates a short synthetic video (so no external sample file is needed)
made of a few abrupt color segments simulating scene changes, runs
VideoProcessor on it, and checks that:
  - metadata.json is created and well-formed
  - keyframes are extracted and saved to disk
  - timestamps are correctly formatted and monotonically non-decreasing
  - scene-change detection fires even with a large sampling interval
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_processor import VideoProcessor, ProcessingConfig


def make_synthetic_video(path: str, fps: int = 10, seconds_per_segment: float = 1.5):
    """Create a short video made of solid-color segments (simulating scene changes)."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    colors = [(30, 30, 30), (200, 200, 200), (30, 30, 200), (30, 200, 30)]
    for color in colors:
        base = np.full((height, width, 3), color, dtype=np.uint8)
        n_frames = int(seconds_per_segment * fps)
        for _ in range(n_frames):
            noise = np.random.randint(0, 5, base.shape, dtype=np.uint8)
            writer.write(cv2.add(base, noise))
    writer.release()


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.video_path = str(Path(self.tmp_dir) / "synthetic.mp4")
        make_synthetic_video(self.video_path)
        self.output_root = str(Path(self.tmp_dir) / "processed_videos")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_pipeline_end_to_end(self):
        config = ProcessingConfig(
            sampling_interval_sec=1.0,
            output_root=self.output_root,
            scene_change_threshold=0.6,
        )
        result = VideoProcessor(config).process(self.video_path)

        self.assertGreater(result["num_frames_extracted"], 0)
        self.assertEqual(result["video_metadata"]["filename"], "synthetic.mp4")

        video_name = "synthetic"
        meta_path = Path(self.output_root) / video_name / "metadata.json"
        frames_dir = Path(self.output_root) / video_name / "frames"

        self.assertTrue(meta_path.exists())
        self.assertTrue(frames_dir.exists())
        saved_frames = list(frames_dir.glob("*.jpg"))
        self.assertEqual(len(saved_frames), result["num_frames_extracted"])

        prev_ts = -1.0
        for frame_info in result["frames"]:
            self.assertRegex(frame_info["timestamp"], r"^\d{2}:\d{2}:\d{2}$")
            self.assertGreaterEqual(frame_info["timestamp_sec"], prev_ts)
            prev_ts = frame_info["timestamp_sec"]
            self.assertTrue(Path(frame_info["frame_path"]).exists())

    def test_detects_scene_changes_beyond_uniform_sampling(self):
        # Large interval => scene-change detection is what must catch the cuts.
        config = ProcessingConfig(
            sampling_interval_sec=10.0,
            output_root=self.output_root,
            scene_change_threshold=0.6,
            min_scene_gap_sec=0.2,
        )
        result = VideoProcessor(config).process(self.video_path)
        triggers = {f["trigger"] for f in result["frames"]}
        self.assertIn("scene_change", triggers)


if __name__ == "__main__":
    unittest.main()
