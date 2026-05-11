from __future__ import annotations
import cv2
import numpy as np
from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, DENOISE_KERNEL_SIZE


class Preprocessor:
    """
    Converts a raw BGR ROI into a lighting-normalised grayscale image
    ready for structural comparison.

    Pipeline: BGR → Gray → Gaussian denoise → CLAHE
    CLAHE compensates for up to ~20 % lighting variation while preserving
    the fine contrast needed to detect micro-level print defects.
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        )
        # Validated odd kernel size; 1 = no blur
        ks = DENOISE_KERNEL_SIZE if DENOISE_KERNEL_SIZE % 2 == 1 else DENOISE_KERNEL_SIZE + 1
        self._blur_ksize = (ks, ks) if ks > 1 else None

    def process(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return a preprocessed uint8 grayscale image."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if self._blur_ksize:
            gray = cv2.GaussianBlur(gray, self._blur_ksize, 0)
        gray = self._clahe.apply(gray)
        return gray
