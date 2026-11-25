"""
SigLIP embedder implementation, built on Hugging Face Transformers.
"""
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

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


class SigLIPEmbedder(BaseEmbedder):
    """SigLIP image/text embedder. Same interface as CLIPEmbedder so the two
    are interchangeable everywhere else in the pipeline."""

    model_type = "siglip"

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "cpu"):
        super().__init__(model_name, device)
        logger.info(f"Loading SigLIP model '{model_name}' on device '{device}'...")
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Some SigLIP configs expose projection_dim directly; others only via
        # the vision sub-config. Handle both defensively.
        self._embedding_dim = getattr(
            self.model.config,
            "projection_dim",
            self.model.config.vision_config.hidden_size,
        )
        logger.info(f"SigLIP model loaded. Embedding dim = {self._embedding_dim}")

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
        # SigLIP's tokenizer expects fixed-length padding (max_length), unlike
        # CLIP's dynamic padding.
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        features = _extract_embedding_tensor(features)
        features = features.float().cpu().numpy()
        return self.normalize(features).astype(np.float32)