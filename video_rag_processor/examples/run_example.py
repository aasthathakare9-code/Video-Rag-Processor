"""
Example / CLI entry point for the Video Processing module.

Usage:
    python examples/run_example.py --video path/to/video.mp4 --interval 1.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_processor import VideoProcessor, ProcessingConfig


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes + metadata from a video.")
    parser.add_argument("--video", required=True, help="Path to input video (mp4/avi/mov).")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--output", default="processed_videos", help="Output root directory.")
    parser.add_argument("--max-width", type=int, default=720, help="Resize saved frames to this max width.")
    parser.add_argument(
        "--scene-threshold", type=float, default=0.55,
        help="Histogram correlation cutoff for scene-change detection (lower = more sensitive).",
    )
    args = parser.parse_args()

    config = ProcessingConfig(
        sampling_interval_sec=args.interval,
        output_root=args.output,
        max_frame_width=args.max_width,
        scene_change_threshold=args.scene_threshold,
    )

    processor = VideoProcessor(config)
    result = processor.process(args.video)

    print("\n=== Summary ===")
    print(f"Video: {result['video_metadata']['filename']}")
    print(f"Duration: {result['video_metadata']['duration_sec']}s")
    print(f"Keyframes extracted: {result['num_frames_extracted']}")
    print(f"Processing time: {result['processing_time_sec']}s")
    if result["frames"]:
        first = result["frames"][0]
        print(f"First keyframe at {first['timestamp']} -> {first['frame_path']}")


if __name__ == "__main__":
    main()
