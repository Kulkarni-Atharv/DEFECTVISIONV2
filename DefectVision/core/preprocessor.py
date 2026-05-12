from __future__ import annotations
import cv2
import numpy as np
from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_KERNEL_SIZE,
    TOPHAT_BG_SIGMA,
)


class Preprocessor:
    """
    Converts a raw BGR ROI into a lighting-normalised grayscale image
    ready for structural comparison.

    Pipeline:
        BGR → Gray → Gaussian denoise → Background normalisation → CLAHE

    Background normalisation step:
        Estimates the slow-varying illumination gradient (bottle-surface
        texture, curved surface, lighting angle) by blurring the image with
        a large Gaussian kernel.  Dividing by this background estimate and
        rescaling to a mean of 128 removes the gradient while preserving the
        sharp ink strokes.  This is far more effective than CLAHE alone for
        handling uneven lighting across the ROI.
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        )
        ks = DENOISE_KERNEL_SIZE if DENOISE_KERNEL_SIZE % 2 == 1 else DENOISE_KERNEL_SIZE + 1
        self._blur_ksize = (ks, ks) if ks > 1 else None

        # Pre-compute background estimation kernel size from sigma.
        # ksize must be odd and large enough to span several characters so
        # only the slow background gradient is captured, not the text itself.
        if TOPHAT_BG_SIGMA > 0:
            ksize = int(6 * TOPHAT_BG_SIGMA) | 1  # next odd number ≥ 6σ
            self._bg_ksize = (ksize, ksize)
            self._bg_sigma = float(TOPHAT_BG_SIGMA)
        else:
            self._bg_ksize = None
            self._bg_sigma = 0.0

    def process(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return a preprocessed uint8 grayscale image."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Denoise
        if self._blur_ksize:
            gray = cv2.GaussianBlur(gray, self._blur_ksize, 0)

        # Background normalisation — remove illumination gradient.
        # Result is centred at 128: background ≈ 128, ink deviates from it.
        if self._bg_ksize is not None:
            gray = self._normalise_background(gray)

        # Local contrast enhancement
        gray = self._clahe.apply(gray)
        return gray

    def _normalise_background(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape

        # Cap sigma so the kernel never exceeds ROI_dimension / 4.
        # A larger kernel would span the whole ROI, include text pixels in
        # the background estimate, and produce a wrong divide result.
        sigma = min(self._bg_sigma, min(h, w) / 4.0)
        if sigma < 2.0:
            return gray  # ROI too small — skip normalisation

        ksize = int(6 * sigma) | 1   # next odd integer ≥ 6σ
        ksize = max(ksize, 3)

        bg = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        # scale * src1 / src2 — uint8 output is auto-clipped to [0, 255]
        return cv2.divide(gray, bg, scale=128.0)
