"""
CLIP embedder implementation, built on Hugging Face Transformers.
"""
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .embedding_model import BaseEmbedder

logger = logging.getLogger(__name__)


def _extract_embedding_tensor(output):
    """
    transformers >=4.5x changed get_image_features()/get_text_features() to
    return a BaseModelOutputWithPooling instead of a plain tensor. Handle
    both so this module works across transformers versions.
    """
    if torch.is_tensor(output):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(output, attr) and getattr(output, attr) is not None:
            return getattr(output, attr)
    raise TypeError(
        f"Unrecognized output type from get_image_features/get_text_features: {type(output)}"
    )


class CLIPEmbedder(BaseEmbedder):
    """CLIP image/text embedder. Image and text embeddings share the same
    projection space, so cosine similarity between them is meaningful."""

    model_type = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__(model_name, device)
        logger.info(f"Loading CLIP model '{model_name}' on device '{device}'...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._embedding_dim = self.model.config.projection_dim
        logger.info(f"CLIP model loaded. Embedding dim = {self._embedding_dim}")

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        features = _extract_embedding_tensor(features)
        features = features.float().cpu().numpy()
        return self.normalize(features).astype(np.float32)

    @torch.inference_mode()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        features = _extract_embedding_tensor(features)
        features = features.float().cpu().numpy()
        return self.normalize(features).astype(np.float32)