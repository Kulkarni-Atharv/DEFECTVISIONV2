from __future__ import annotations
import cv2
import numpy as np
from config import (
    ALIGN_ENABLED,
    ALIGN_MAX_SHIFT_RATIO,
    ALIGN_MAX_ROTATION_DEG,
    ALIGN_ECC_MAX_ITER,
    ALIGN_ECC_EPSILON,
)


class Aligner:
    """
    Compensates for translation and slight rotation between the reference
    and live ROI caused by conveyor vibration or minor positional variance.

    Primary method — ECC (Enhanced Correlation Coefficient):
        Finds the rigid-body transform (translation + rotation) that
        maximises the normalised cross-correlation between the two images.
        Handles ±5° rotation and ±25% shift without needing feature
        keypoints, so it works reliably on small text ROIs.

    Fallback — Phase correlation:
        Used when ECC diverges (e.g., very low-texture region, large
        initial misalignment).  Handles translation only.
    """

    def __init__(self) -> None:
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            ALIGN_ECC_MAX_ITER,
            ALIGN_ECC_EPSILON,
        )

    def align(self, reference: np.ndarray, live: np.ndarray) -> np.ndarray:
        if not ALIGN_ENABLED:
            return live

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        h, w = reference.shape
        max_dx = w * ALIGN_MAX_SHIFT_RATIO
        max_dy = h * ALIGN_MAX_SHIFT_RATIO

        ref_f  = np.float32(reference)
        live_f = np.float32(live)

        # ---- Primary: ECC with EUCLIDEAN motion (translation + rotation) --
        try:
            M = np.eye(2, 3, dtype=np.float32)
            _, M = cv2.findTransformECC(
                ref_f, live_f, M,
                cv2.MOTION_EUCLIDEAN,
                self._criteria,
            )
            dx    = float(M[0, 2])
            dy    = float(M[1, 2])
            angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

            if (abs(dx) <= max_dx
                    and abs(dy) <= max_dy
                    and abs(angle) <= ALIGN_MAX_ROTATION_DEG):
                return cv2.warpAffine(
                    live, M,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        except cv2.error:
            pass

        # ---- Fallback: phase correlation (translation only) ---------------
        try:
            shift, _ = cv2.phaseCorrelate(ref_f, live_f)
            dx, dy   = float(shift[0]), float(shift[1])

            if abs(dx) <= max_dx and abs(dy) <= max_dy:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                return cv2.warpAffine(
                    live, M,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        except cv2.error:
            pass

        return live
