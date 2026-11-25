"""
Near-duplicate frame filtering using perceptual hashing (pHash).
"""
from collections import deque
from typing import Optional
import imagehash
from PIL import Image
import numpy as np


class FrameDeduplicator:
    """
    Keeps a small rolling window of perceptual hashes of recently *saved*
    frames and rejects new candidates that are too visually similar to any
    of them (Hamming distance <= threshold).
    """

    def __init__(self, hash_size: int = 8, hamming_threshold: int = 5, compare_last_n: int = 3):
        self.hash_size = hash_size
        self.hamming_threshold = hamming_threshold
        self.recent_hashes = deque(maxlen=compare_last_n)

    @staticmethod
    def _to_pil(frame_bgr: np.ndarray) -> Image.Image:
        rgb = frame_bgr[:, :, ::-1]  # BGR (OpenCV) -> RGB (PIL)
        return Image.fromarray(rgb)

    def compute_hash(self, frame_bgr: np.ndarray) -> imagehash.ImageHash:
        return imagehash.phash(self._to_pil(frame_bgr), hash_size=self.hash_size)

    def is_duplicate(self, frame_hash: imagehash.ImageHash) -> bool:
        return any((frame_hash - prev) <= self.hamming_threshold for prev in self.recent_hashes)

    def register(self, frame_hash: imagehash.ImageHash) -> None:
        self.recent_hashes.append(frame_hash)

    def check_and_register(self, frame_bgr: np.ndarray) -> Optional[imagehash.ImageHash]:
        """
        Returns the pHash if the frame is unique enough to keep (and registers
        it for future comparisons), or None if it's a near-duplicate.
        """
        h = self.compute_hash(frame_bgr)
        if self.is_duplicate(h):
            return None
        self.register(h)
        return h
