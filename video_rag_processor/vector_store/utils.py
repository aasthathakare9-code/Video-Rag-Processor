"""
Utility helpers: logging, loading embeddings/metadata produced by the
embedding module, validation, and deterministic point-ID generation for
the Qdrant vector store module.
"""
import json
import logging
import os
import uuid
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Fixed namespace so point IDs are stable and reproducible across runs:
# re-indexing the same video+frame always yields the same point ID, so
# re-running the indexer safely upserts (updates in place) instead of
# creating duplicate points.
POINT_ID_NAMESPACE = uuid.UUID("d6f0a7d0-8f1a-4b8b-9e0e-5c1e9f6a2b31")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def stable_point_id(video_name: str, frame_filename: str) -> str:
    """Deterministic UUID derived from video_name + frame_filename, so the
    same frame always maps to the same Qdrant point ID."""
    key = f"{video_name}:{frame_filename}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, key))


def load_embeddings(embeddings_path: str) -> np.ndarray:
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"embeddings.npy not found at: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array, got shape {embeddings.shape}")
    return embeddings


def load_embedding_metadata(metadata_path: str) -> dict:
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"embedding_metadata.json not found at: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def validate_embeddings_and_metadata(embeddings: np.ndarray, metadata: dict) -> List[dict]:
    """
    Validate that embeddings.npy and embedding_metadata.json line up, and
    return the per-frame metadata records in the same order as the vectors.
    """
    frames = metadata.get("frames")
    if not frames:
        raise ValueError("embedding_metadata.json has no 'frames' entries.")

    n_vectors = embeddings.shape[0]
    n_meta = len(frames)
    if n_vectors != n_meta:
        raise ValueError(
            f"Mismatch between embeddings ({n_vectors}) and metadata records ({n_meta})."
        )

    declared_dim = metadata.get("embedding_dim")
    if declared_dim is not None and embeddings.shape[1] != declared_dim:
        raise ValueError(
            f"Embedding dimension mismatch: array has {embeddings.shape[1]}, "
            f"metadata declares {declared_dim}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN/Inf values.")

    # Duplicate frame filenames would collide on point ID within one video.
    filenames = [f.get("frame_filename") for f in frames]
    if len(filenames) != len(set(filenames)):
        dupes = {f for f in filenames if filenames.count(f) > 1}
        raise ValueError(f"Duplicate frame_filename entries found in metadata: {dupes}")

    return frames


def normalize_if_needed(embeddings: np.ndarray, atol: float = 1e-3) -> np.ndarray:
    """Re-normalize vectors to unit length if they aren't already, so cosine
    similarity search behaves correctly regardless of upstream guarantees."""
    norms = np.linalg.norm(embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=atol):
        return embeddings.astype(np.float32)
    logger.warning("Embeddings are not unit-normalized; normalizing before indexing.")
    norms = np.where(norms == 0, 1e-12, norms)
    return (embeddings / norms[:, None]).astype(np.float32)
