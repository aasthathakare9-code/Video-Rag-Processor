"""
Reusable semantic search against an indexed Qdrant collection.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from .config import QdrantConfig
from .qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class VideoSearcher:
    """
    Wraps a QdrantStore scoped to one model_type's collection, exposing a
    simple search(query_embedding, top_k) interface for text -> frame
    search. Reuse one instance across multiple queries to avoid re-opening
    the Qdrant client each time.
    """

    def __init__(
        self,
        model_type: str,
        config: Optional[QdrantConfig] = None,
        store: Optional[QdrantStore] = None,
    ):
        self.config = config or QdrantConfig()
        self.store = store or QdrantStore(self.config)
        self.collection_name = self.config.collection_name(model_type)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        video_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        query_embedding: a normalized 1D vector produced by the same
        CLIP/SigLIP model (and therefore the same dimension/space) used to
        index the frames in this collection.
        video_name: optionally restrict the search to a single video.
        """
        if not self.store.collection_exists(self.collection_name):
            logger.warning(f"Collection '{self.collection_name}' does not exist yet. Returning no results.")
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        results = self.store.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k,
            video_name=video_name,
        )
        logger.info(f"Search returned {len(results)} result(s) from '{self.collection_name}'")
        return results


def search(
    query_embedding: np.ndarray,
    model_type: str,
    top_k: int = 5,
    video_name: Optional[str] = None,
    config: Optional[QdrantConfig] = None,
) -> List[Dict]:
    """Functional convenience wrapper around VideoSearcher.search(), for
    one-off queries where reusing a searcher instance isn't necessary."""
    searcher = VideoSearcher(model_type=model_type, config=config)
    return searcher.search(query_embedding, top_k=top_k, video_name=video_name)
