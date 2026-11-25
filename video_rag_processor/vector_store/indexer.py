"""
index_video(): loads one video's embeddings + metadata (produced by the
embedding module) and indexes them into the appropriate Qdrant collection.
"""
import logging
import os
from typing import Optional

from .config import QdrantConfig
from .qdrant_store import QdrantStore
from .utils import (
    load_embedding_metadata,
    load_embeddings,
    normalize_if_needed,
    setup_logging,
    stable_point_id,
    validate_embeddings_and_metadata,
)

logger = logging.getLogger(__name__)


def index_video(
    video_name: str,
    embeddings_path: str,
    metadata_path: str,
    config: Optional[QdrantConfig] = None,
    store: Optional[QdrantStore] = None,
) -> dict:
    """
    Index one video's frame embeddings into Qdrant.

    Returns a summary dict:
        {video_name, collection, points_indexed, embedding_dim, model_type}
    """
    config = config or QdrantConfig()
    setup_logging(config.log_level)
    store = store or QdrantStore(config)

    logger.info(f"Indexing video '{video_name}' from '{embeddings_path}'")

    embeddings = load_embeddings(embeddings_path)
    metadata = load_embedding_metadata(metadata_path)
    frames = validate_embeddings_and_metadata(embeddings, metadata)

    model_type = metadata.get("model_type", "unknown")
    embeddings = normalize_if_needed(embeddings)

    collection_name = config.collection_name(model_type)
    dim = embeddings.shape[1]
    store.ensure_collection(collection_name, dim)

    ids = [stable_point_id(video_name, f["frame_filename"]) for f in frames]
    payloads = [
        {
            "video_name": video_name,
            "frame_filename": f["frame_filename"],
            "frame_number": f.get("frame_number"),
            "timestamp_seconds": f.get("timestamp_seconds"),
            "timestamp": f.get("timestamp_hms"),
            "model_type": model_type,
            "model_name": metadata.get("model_name"),
        }
        for f in frames
    ]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate point IDs generated for this video's frames; aborting index.")

    n_indexed = store.upsert_points(collection_name, ids, embeddings, payloads)

    summary = {
        "video_name": video_name,
        "collection": collection_name,
        "points_indexed": n_indexed,
        "embedding_dim": dim,
        "model_type": model_type,
    }
    logger.info(f"Done. {summary}")
    return summary


def index_video_dir(
    video_dir: str,
    model_type: str,
    config: Optional[QdrantConfig] = None,
    store: Optional[QdrantStore] = None,
) -> dict:
    """
    Convenience wrapper: given processed_videos/<video_name>/, derives the
    embeddings.npy / embedding_metadata.json paths for the specified
    model_type and the video name from the folder name, then calls
    index_video(). model_type is required since CLIP and SigLIP embeddings
    for the same video are stored in separate subfolders.
    """
    video_name = os.path.basename(os.path.normpath(video_dir))
    embeddings_path = os.path.join(video_dir, "embeddings", model_type, "embeddings.npy")
    metadata_path = os.path.join(video_dir, "embeddings", model_type, "embedding_metadata.json")
    return index_video(video_name, embeddings_path, metadata_path, config=config, store=store)