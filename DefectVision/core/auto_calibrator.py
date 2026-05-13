from __future__ import annotations
import cv2
import numpy as np
from core.inspector import Inspector
from config import (
    AUTO_CAL_N_CONFIRM,
    AUTO_CAL_MIN_INK_RATIO,
    AUTO_CAL_MIN_BLUR_VAR,
)


class AutoCalibrator:
    """
    Watches incoming preprocessed frames and locks a clean reference
    the first time text is clearly and stably visible — no user input
    required for normal operation.

    State machine
    ─────────────
    SEARCHING → no text found yet (or last check failed)
    PENDING   → text found, accumulating confirmation frames
    LOCKED    → reference built, ready for inspection

    Clarity gates (all must pass before a frame is counted)
    ───────────────────────────────────────────────────────
    1. Text bounding box found via binarisation
    2. Ink pixel ratio within that box ≥ AUTO_CAL_MIN_INK_RATIO
       (rules out faint/partial text entering the field of view)
    3. Laplacian variance ≥ AUTO_CAL_MIN_BLUR_VAR
       (rules out motion blur during print transit)
    4. Bounding-box area ≥ 10% of ROI area
       (rules out tiny noise blobs at the frame edge)
    """

    SEARCHING = "SEARCHING"
    PENDING   = "PENDING"
    LOCKED    = "LOCKED"

    def __init__(self) -> None:
        self._n     = AUTO_CAL_N_CONFIRM
        self._ink   = AUTO_CAL_MIN_INK_RATIO
        self._blur  = AUTO_CAL_MIN_BLUR_VAR
        self._buf:  list[np.ndarray] = []
        self._ref:  np.ndarray | None = None
        self.state  = self.SEARCHING

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def reference(self) -> np.ndarray | None:
        return self._ref

    @property
    def is_locked(self) -> bool:
        return self.state == self.LOCKED

    @property
    def progress(self) -> tuple[int, int]:
        """(frames_accumulated, frames_needed)"""
        return len(self._buf), self._n

    def update(self, gray: np.ndarray, bin_mask: np.ndarray) -> bool:
        """
        Feed one preprocessed frame.
        Returns True the moment the reference locks (fires once).
        """
        if self.state == self.LOCKED:
            return False

        if not self._is_clear(gray, bin_mask):
            self._buf.clear()
            self.state = self.SEARCHING
            return False

        self._buf.append(gray.copy())
        self.state = self.PENDING

        if len(self._buf) >= self._n:
            self._ref = np.uint8(np.mean(self._buf, axis=0))
            self._buf.clear()
            self.state = self.LOCKED
            return True

        return False

    def reset(self) -> None:
        self._buf.clear()
        self._ref  = None
        self.state = self.SEARCHING

    # ------------------------------------------------------------------
    # Clarity gate
    # ------------------------------------------------------------------
    def _is_clear(self, gray: np.ndarray, bin_mask: np.ndarray) -> bool:
        bbox = Inspector._text_bbox(bin_mask)
        if bbox is None:
            return False

        x, y, w, h = bbox
        roi_area  = gray.shape[0] * gray.shape[1]

        # Text box must cover at least 10% of the ROI
        if (w * h) < roi_area * 0.10:
            return False

        # Sufficient ink within the text box
        ink_ratio = int(np.count_nonzero(bin_mask[y:y + h, x:x + w])) / max(w * h, 1)
        if ink_ratio < self._ink:
            return False

        # Not motion-blurred
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self._blur:
            return False

        return True
