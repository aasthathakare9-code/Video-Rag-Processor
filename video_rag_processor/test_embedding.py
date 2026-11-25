#!/usr/bin/env python3
"""
Example/test script for the CLIP/SigLIP Embedding Module.

Loads the output of the existing Video Processing Module, generates image
embeddings for every extracted frame, generates a text embedding for a
sample query, and verifies the output files were created correctly.

Usage:
    python test_embedding.py --video-dir processed_videos/my_video --model-type clip
    python test_embedding.py --video-dir processed_videos/my_video --model-type siglip --batch-size 16
"""
import argparse
import os

import numpy as np

from embedding import EmbeddingConfig, EmbeddingGenerator

SAMPLE_QUERY = "a person standing near a whiteboard"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True,
                         help="Path to processed_videos/<video_name> (output of the Video Processing Module).")
    parser.add_argument("--model-type", type=str, default="clip", choices=["clip", "siglip"])
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    config = EmbeddingConfig(
        video_dir=args.video_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
    )

    generator = EmbeddingGenerator(config)

    print(f"\nGenerating image embeddings for frames in: {config.frames_dir}")
    summary = generator.run()

    print(f"\nGenerating text embedding for query: '{SAMPLE_QUERY}'")
    query_vector = generator.embed_query(SAMPLE_QUERY)

    print("\n=== Results ===")
    print(f"Model: {summary['model_type']} ({summary['model_name']})")
    print(f"Embedding dimension: {summary['embedding_dim']}")
    print(f"Frames processed: {summary['frames_embedded']} / {summary['video_frames_found']}")
    print(f"Frames skipped (unreadable): {summary['frames_skipped']}")
    print(f"Query embedding shape: {query_vector.shape}")
    print(f"Query embedding norm (should be ~1.0): {np.linalg.norm(query_vector):.4f}")

    npy_path = os.path.join(config.output_dir, "embeddings.npy")
    meta_path = os.path.join(config.output_dir, "embedding_metadata.json")
    assert os.path.isfile(npy_path), f"Missing {npy_path}"
    assert os.path.isfile(meta_path), f"Missing {meta_path}"

    saved = np.load(npy_path)
    print(f"\nembeddings.npy shape: {saved.shape}")
    print(f"Output files confirmed at: {config.output_dir}")


if __name__ == "__main__":
    main()
