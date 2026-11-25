"""
Configuration for the Qdrant vector storage & retrieval module.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class QdrantConfig:
    """
    Collection strategy: one collection per model_type (CLIP and SigLIP have
    different vector dimensions, and a Qdrant collection has a fixed
    dimension), shared across every indexed video. Each point's payload
    carries "video_name", so searches can be scoped to a single video via a
    payload filter, or left unscoped to search across all indexed videos.
    This keeps the number of collections small and avoids one video's data
    ever overwriting another's.
    """

    # Local persistent on-disk storage (embedded mode, no server required).
    # Qdrant keeps a real index on disk here and reloads it automatically on
    # the next run, so indexed data survives across program restarts.
    storage_path: str = "vector_store_data/qdrant_local"

    # Optional: point at a running Qdrant server instead of embedded local
    # storage (e.g. Docker or Qdrant Cloud later). If host is set, it takes
    # priority over storage_path.
    host: Optional[str] = None
    port: int = 6333
    api_key: Optional[str] = None

    collection_prefix: str = "video_frames"

    log_level: str = "INFO"

    def collection_name(self, model_type: str) -> str:
        return f"{self.collection_prefix}_{model_type.lower().strip()}"

    def __post_init__(self):
        if not self.host:
            os.makedirs(self.storage_path, exist_ok=True)
