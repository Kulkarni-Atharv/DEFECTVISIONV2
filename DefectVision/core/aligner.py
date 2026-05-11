from __future__ import annotations
import cv2
import numpy as np
from config import ALIGN_ENABLED, ALIGN_MAX_SHIFT_RATIO


class Aligner:
    """
    Compensates for sub-pixel translation between the reference and live
    ROI caused by conveyor vibration or minor bottle positional variance.

    Uses phase correlation (FFT-based, ~1–3 ms per call) which is fast
    enough for 30+ fps on a Raspberry Pi CM5.  If the estimated shift
    exceeds ALIGN_MAX_SHIFT_RATIO of the ROI dimension the alignment is
    considered unreliable and the raw live frame is returned unchanged.
    """

    def align(self, reference: np.ndarray, live: np.ndarray) -> np.ndarray:
        if not ALIGN_ENABLED:
            return live

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        try:
            shift, _response = cv2.phaseCorrelate(
                np.float32(reference),
                np.float32(live),
            )
            dx, dy = shift

            max_dx = reference.shape[1] * ALIGN_MAX_SHIFT_RATIO
            max_dy = reference.shape[0] * ALIGN_MAX_SHIFT_RATIO
            if abs(dx) > max_dx or abs(dy) > max_dy:
                return live  # unreliable shift; skip alignment

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(
                live, M,
                (live.shape[1], live.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned

        except cv2.error:
            return live
