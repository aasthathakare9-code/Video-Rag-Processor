#!/usr/bin/env python3
"""
Command-line entry point for the CLIP/SigLIP Embedding Module.

Examples:
    # Simplest form — point at the video's output folder
    python run_embedding.py --video-dir processed_videos/my_video --model-type clip

    # SigLIP, larger batch size, forced CPU
    python run_embedding.py --video-dir processed_videos/my_video \
        --model-type siglip --batch-size 64 --device cpu

    # Fully explicit paths instead of --video-dir
    python run_embedding.py \
        --frames-dir processed_videos/my_video/frames \
        --metadata-path processed_videos/my_video/metadata.json \
        --output-dir processed_videos/my_video/embeddings \
        --model-type clip
"""
import argparse

from embedding import EmbeddingConfig, EmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CLIP/SigLIP embeddings for extracted video frames."
    )
    parser.add_argument("--video-dir", type=str, default=None,
                         help="Path to processed_videos/<video_name> (expects frames/ and metadata.json inside).")
    parser.add_argument("--frames-dir", type=str, default=None,
                         help="Path to frames/ directory. Overrides the path derived from --video-dir.")
    parser.add_argument("--metadata-path", type=str, default=None,
                         help="Path to metadata.json. Overrides the path derived from --video-dir.")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Where to write embeddings.npy + embedding_metadata.json. "
                              "Overrides the path derived from --video-dir.")
    parser.add_argument("--model-type", type=str, default="clip", choices=["clip", "siglip"])
    parser.add_argument("--model-name", type=str, default=None,
                         help="HF model id/path. Defaults to a sensible checkpoint per --model-type.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="Defaults to CUDA if available, else CPU.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.video_dir and not (args.frames_dir and args.metadata_path and args.output_dir):
        raise SystemExit(
            "Provide either --video-dir, or all of --frames-dir / --metadata-path / --output-dir."
        )

    config = EmbeddingConfig(
        model_type=args.model_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        video_dir=args.video_dir,
        frames_dir=args.frames_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )

    generator = EmbeddingGenerator(config)
    summary = generator.run()

    print("\n=== Embedding Generation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
