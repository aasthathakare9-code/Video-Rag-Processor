"""
Low-level wrapper around the Qdrant client: collection lifecycle, upserts,
and cosine-similarity search. This is the only file that talks to the
qdrant-client library directly, so version-specific quirks are isolated
here.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import QdrantConfig

logger = logging.getLogger(__name__)


class QdrantStore:
    """Thin wrapper around Qdrant collection lifecycle + point I/O."""

    # Shared across instances within a process: Qdrant's embedded local-disk
    # mode locks the storage directory, so opening it twice in one process
    # would fail. Reusing one client avoids that.
    _client: Optional[QdrantClient] = None
    _client_storage_path: Optional[str] = None

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = self._get_client(config)

    @classmethod
    def _get_client(cls, config: QdrantConfig) -> QdrantClient:
        if config.host:
            logger.info(f"Connecting to Qdrant server at {config.host}:{config.port}")
            return QdrantClient(host=config.host, port=config.port, api_key=config.api_key)

        if cls._client is not None and cls._client_storage_path == config.storage_path:
            return cls._client

        logger.info(f"Using local persistent Qdrant storage at '{config.storage_path}'")
        cls._client = QdrantClient(path=config.storage_path)
        cls._client_storage_path = config.storage_path
        return cls._client

    def collection_exists(self, collection_name: str) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return collection_name in existing

    def ensure_collection(self, collection_name: str, dim: int) -> None:
        """Create the collection if it doesn't exist yet. If it exists with
        a different vector size, raise rather than silently corrupting it."""
        if self.collection_exists(collection_name):
            try:
                info = self.client.get_collection(collection_name)
                existing_dim = info.config.params.vectors.size
                if existing_dim != dim:
                    raise ValueError(
                        f"Collection '{collection_name}' already exists with dim={existing_dim}, "
                        f"but new vectors have dim={dim}. This usually means two different "
                        f"models are being indexed into the same collection."
                    )
            except AttributeError:
                logger.warning(
                    f"Could not verify existing dimension for '{collection_name}'; proceeding."
                )
            return

        logger.info(f"Creating Qdrant collection '{collection_name}' (dim={dim}, cosine distance)")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def upsert_points(
        self,
        collection_name: str,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict],
        batch_size: int = 256,
    ) -> int:
        """Upsert points in batches. Re-running with the same IDs updates
        existing points in place rather than creating duplicates."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = [
                qmodels.PointStruct(
                    id=ids[i],
                    vector=vectors[i].tolist(),
                    payload=payloads[i],
                )
                for i in range(start, end)
            ]
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            logger.info(f"Upserted {end}/{total} points into '{collection_name}'")
        return total

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        video_name: Optional[str] = None,
    ) -> List[Dict]:
        query_filter = None
        if video_name:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="video_name", match=qmodels.MatchValue(value=video_name))]
            )

        # qdrant-client >=1.10 prefers query_points(); older versions only
        # have search(). Support both so this module isn't pinned to one
        # exact client version.
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            raw_results = response.points
        else:
            raw_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        formatted = []
        for point in raw_results:
            payload = point.payload or {}
            formatted.append({
                "score": float(point.score),
                "video_name": payload.get("video_name"),
                "frame_filename": payload.get("frame_filename"),
                "frame_number": payload.get("frame_number"),
                "timestamp_seconds": payload.get("timestamp_seconds"),
                "timestamp": payload.get("timestamp"),
            })
        return formatted

    def count(self, collection_name: str, video_name: Optional[str] = None) -> int:
        if not self.collection_exists(collection_name):
            return 0
        query_filter = None
        if video_name:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="video_name", match=qmodels.MatchValue(value=video_name))]
            )
        result = self.client.count(collection_name=collection_name, count_filter=query_filter, exact=True)
        return result.count
