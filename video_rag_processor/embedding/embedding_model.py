"""
Abstract base class defining the common embedding interface. Both the CLIP
and SigLIP embedders implement this, so the rest of the pipeline (batching,
saving, validation, text search later on) never needs to know which model
is actually loaded.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from PIL import Image


class BaseEmbedder(ABC):
    """Common interface every embedder implementation must follow."""

    model_type: str = "base"

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("Embedding dimension not set. Model may not be loaded yet.")
        return self._embedding_dim

    # --- Images --------------------------------------------------------

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images -> (N, D) normalized float32 array."""
        raise NotImplementedError

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image -> (D,) normalized float32 array."""
        return self.embed_images([image])[0]

    # --- Text ------------------------------------------------------------

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings -> (N, D) normalized float32 array."""
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string -> (D,) normalized float32 array."""
        return self.embed_texts([text])[0]

    # --- Shared helpers --------------------------------------------------

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so cosine similarity reduces to a dot product."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return vectors / norms
