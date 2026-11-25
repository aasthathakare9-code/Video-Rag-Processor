"""
EmbeddingGenerator: orchestrates reading frames + timestamps, running batch
inference through a configured embedder, validating results, and writing
embeddings.npy + embedding_metadata.json to disk. Also exposes a reusable
text-embedding method for the future search/retrieval module.
"""
import json
import logging
import os
import time
from typing import List, Optional

import numpy as np

from .clip_embedder import CLIPEmbedder
from .config import EmbeddingConfig
from .embedding_model import BaseEmbedder
from .siglip_embedder import SigLIPEmbedder
from .utils import discover_frames, safe_load_image, setup_logging

logger = logging.getLogger(__name__)


def build_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Factory: returns the embedder implementation matching config.model_type.
    This is the one place that needs to know about concrete embedder classes —
    everything else in the pipeline talks to the BaseEmbedder interface."""
    if config.model_type == "clip":
        return CLIPEmbedder(model_name=config.model_name, device=config.device)
    elif config.model_type == "siglip":
        return SigLIPEmbedder(model_name=config.model_name, device=config.device)
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")


class EmbeddingGenerator:
    """
    Runs the full image-embedding pipeline for one video's extracted frames,
    and provides embed_query() for turning a text search string into a
    vector in the same embedding space.
    """

    def __init__(self, config: EmbeddingConfig, embedder: Optional[BaseEmbedder] = None):
        self.config = config
        setup_logging(config.log_level)
        # Allow an already-loaded embedder to be reused across multiple
        # videos instead of reloading the model each time.
        self.embedder = embedder or build_embedder(config)

    def run(self) -> dict:
        """
        Full pipeline: discover frames -> batch embed -> validate -> save.
        Returns a summary dict.
        """
        start = time.time()
        frame_records = discover_frames(self.config.frames_dir, self.config.metadata_path)
        if not frame_records:
            raise RuntimeError(f"No valid frames found in '{self.config.frames_dir}'.")

        all_vectors: List[np.ndarray] = []
        all_meta: List[dict] = []
        skipped = 0

        batch_size = self.config.batch_size
        total = len(frame_records)

        for start_idx in range(0, total, batch_size):
            batch_records = frame_records[start_idx:start_idx + batch_size]

            images = []
            valid_records = []
            for record in batch_records:
                img = safe_load_image(record["path"])
                if img is None:
                    skipped += 1
                    continue
                images.append(img)
                valid_records.append(record)

            if not images:
                continue

            vectors = self.embedder.embed_images(images)

            for record, vector in zip(valid_records, vectors):
                all_vectors.append(vector)
                all_meta.append({
                    "frame_filename": record["filename"],
                    "frame_number": record["frame_number"],
                    "timestamp_seconds": record["timestamp_seconds"],
                    "timestamp_hms": record["timestamp_hms"],
                    "model_type": self.embedder.model_type,
                    "model_name": self.embedder.model_name,
                    "embedding_dim": self.embedder.embedding_dim,
                })

            done = min(start_idx + batch_size, total)
            logger.info(f"Processed {done}/{total} frames ({skipped} skipped so far)...")

        if not all_vectors:
            raise RuntimeError("No embeddings were generated (are all frames unreadable?).")

        embeddings = np.stack(all_vectors).astype(np.float32)
        self._validate(embeddings, all_meta)
        self._save(embeddings, all_meta)

        elapsed = time.time() - start
        summary = {
            "video_frames_found": total,
            "frames_embedded": len(all_meta),
            "frames_skipped": skipped,
            "embedding_dim": self.embedder.embedding_dim,
            "model_type": self.embedder.model_type,
            "model_name": self.embedder.model_name,
            "output_dir": self.config.output_dir,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(f"Done. {summary}")
        return summary

    # --- Validation ----------------------------------------------------

    def _validate(self, embeddings: np.ndarray, meta: List[dict]) -> None:
        n_vectors = embeddings.shape[0]
        n_meta = len(meta)

        assert n_vectors == n_meta, (
            f"Mismatch between number of embeddings ({n_vectors}) and metadata records ({n_meta})."
        )

        dims = {embeddings.shape[1]}
        dims.update(m["embedding_dim"] for m in meta)
        assert len(dims) == 1, f"Inconsistent embedding dimensions found: {dims}"
        assert embeddings.shape[1] == self.embedder.embedding_dim, "Embedding dim mismatch vs model config."

        assert np.isfinite(embeddings).all(), "Embeddings contain NaN/Inf values."

        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit-normalized."

        model_names = {m["model_name"] for m in meta}
        model_types = {m["model_type"] for m in meta}
        assert model_names == {self.embedder.model_name}, "Inconsistent model_name across records."
        assert model_types == {self.embedder.model_type}, "Inconsistent model_type across records."

        for i, m in enumerate(meta):
            assert m["frame_filename"], f"Missing frame_filename at index {i}"
            assert m["timestamp_seconds"] is not None, f"Missing timestamp at index {i}"

        logger.info("Validation passed: counts, dimensions, ordering, and normalization all consistent.")

    # --- Saving ----------------------------------------------------------

    def _save(self, embeddings: np.ndarray, meta: List[dict]) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

        npy_path = os.path.join(self.config.output_dir, "embeddings.npy")
        meta_path = os.path.join(self.config.output_dir, "embedding_metadata.json")

        # Row i of embeddings.npy corresponds exactly to meta["frames"][i].
        np.save(npy_path, embeddings)

        payload = {
            "model_type": self.embedder.model_type,
            "model_name": self.embedder.model_name,
            "embedding_dim": self.embedder.embedding_dim,
            "num_frames": len(meta),
            "normalized": self.config.normalize,
            "frames": meta,
        }
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved embeddings -> {npy_path}")
        logger.info(f"Saved metadata   -> {meta_path}")

    # --- Text embedding (reusable for the search module) -----------------

    def embed_query(self, text: str) -> np.ndarray:
        """Generate a normalized text embedding, in the same space as the
        image embeddings, for use as a search query against the vector DB."""
        return self.embedder.embed_text(text)
