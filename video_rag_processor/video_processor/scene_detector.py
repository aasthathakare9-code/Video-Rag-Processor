"""
Modular scene/shot-change detection.

`BaseSceneChangeDetector` defines the interface so the detection method can
later be swapped for something more advanced (SSIM, deep-feature distance,
PySceneDetect, etc.) without touching the rest of the pipeline.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np


class BaseSceneChangeDetector(ABC):
    @abstractmethod
    def compute_signature(self, frame: np.ndarray) -> object:
        """Compute a lightweight representation of a frame used for comparison."""

    @abstractmethod
    def is_scene_change(self, prev_signature: object, curr_signature: object) -> bool:
        """Return True if curr differs enough from prev to count as a scene change."""


class HistogramSceneChangeDetector(BaseSceneChangeDetector):
    """
    Compares grayscale histograms of consecutive frames via correlation.
    Cheap, dependency-free, and a solid baseline for a student project.
    Correlation close to 1.0 -> very similar frames; low/negative -> likely cut.
    """

    def __init__(self, threshold: float = 0.55, hist_bins: int = 64):
        self.threshold = threshold
        self.hist_bins = hist_bins

    def compute_signature(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def is_scene_change(self, prev_signature, curr_signature) -> bool:
        if prev_signature is None:
            return False
        correlation = cv2.compareHist(prev_signature, curr_signature, cv2.HISTCMP_CORREL)
        return correlation < self.threshold


def get_scene_detector(method: str, threshold: float) -> BaseSceneChangeDetector:
    if method == "histogram":
        return HistogramSceneChangeDetector(threshold=threshold)
    raise ValueError(f"Unknown scene_detector_method: {method!r}")
