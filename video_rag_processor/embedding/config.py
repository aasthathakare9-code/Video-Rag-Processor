"""
Configuration for the CLIP/SigLIP Embedding Module.
"""
import os
from dataclasses import dataclass
from typing import Optional

import torch

# Known default checkpoints for each supported model family.
DEFAULT_MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EmbeddingConfig:
    """
    All knobs for a single embedding-generation run.

    Typical usage — point it at a video's output folder and let it derive
    frames_dir / metadata_path / output_dir automatically:

        config = EmbeddingConfig(video_dir="processed_videos/my_video", model_type="siglip")

    Or wire every path yourself for more control:

        config = EmbeddingConfig(
            frames_dir="processed_videos/my_video/frames",
            metadata_path="processed_videos/my_video/metadata.json",
            output_dir="processed_videos/my_video/embeddings",
        )
    """

    model_type: str = "clip"                  # "clip" or "siglip"
    model_name: Optional[str] = None           # HF model id/path; defaults per model_type
    batch_size: int = 32
    device: Optional[str] = None               # "cuda" / "cpu"; auto-detected if None

    video_dir: Optional[str] = None            # processed_videos/<video_name>
    frames_dir: Optional[str] = None           # defaults to <video_dir>/frames
    metadata_path: Optional[str] = None        # defaults to <video_dir>/metadata.json
    output_dir: Optional[str] = None           # defaults to <video_dir>/embeddings

    normalize: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        self.model_type = self.model_type.lower().strip()
        if self.model_type not in DEFAULT_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. Choose one of: "
                f"{list(DEFAULT_MODEL_NAMES.keys())}"
            )

        if not self.model_name:
            self.model_name = DEFAULT_MODEL_NAMES[self.model_type]

        if not self.device:
            self.device = _auto_device()

        if self.video_dir:
            if not self.frames_dir:
                self.frames_dir = os.path.join(self.video_dir, "frames")
            if not self.metadata_path:
                self.metadata_path = os.path.join(self.video_dir, "metadata.json")
            if not self.output_dir:
                self.output_dir = os.path.join(self.video_dir, "embeddings", self.model_type)

        if not self.frames_dir:
            raise ValueError("frames_dir could not be determined. Pass video_dir or frames_dir explicitly.")
        if not self.output_dir:
            raise ValueError("output_dir could not be determined. Pass video_dir or output_dir explicitly.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
