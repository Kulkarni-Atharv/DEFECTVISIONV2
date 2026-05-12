from __future__ import annotations
import cv2
import numpy as np
from config import (
    POSITION_LOCK_THRESHOLD,
    POSITION_LOCK_SEARCH_MARGIN,
    POSITION_LOCK_BLUR_THRESHOLD,
)


class PositionLock:
    """
    Locates the print region in each frame via normalised cross-correlation
    template matching, replacing the fixed-ROI crop used in the static setup.

    Strategy
    --------
    First call (or after losing track): full-frame search — expensive but rare.
    Subsequent calls: restricted ±SEARCH_MARGIN window around last known
    position — typically completes in < 5 ms on CM5 at 1456×1088.

    Focused-template support
     ------------------------
    When the drawn ROI is large (contains a lot of background), the raw ROI
    template is dominated by featureless surface and gives low match confidence.
    Pass ``roi_offset`` and ``full_roi_size`` so the position lock can work on
    a tight text-region crop while returning full-ROI coordinates to the caller.

    Gates
    -----
    • Match confidence < POSITION_LOCK_THRESHOLD  → no match, skip frame.
    • Laplacian variance of matched crop < POSITION_LOCK_BLUR_THRESHOLD
      → motion-blurred frame, skip.  Set threshold to 0 to disable.
    """

    def __init__(
        self,
        template_gray: np.ndarray,
        roi_offset: tuple[int, int] = (0, 0),
        full_roi_size: tuple[int, int] | None = None,
    ) -> None:
        self._tpl: np.ndarray = template_gray
        self._th, self._tw   = template_gray.shape[:2]
        self._margin: int    = POSITION_LOCK_SEARCH_MARGIN
        self._match_thr: float = POSITION_LOCK_THRESHOLD
        self._blur_thr:  float = POSITION_LOCK_BLUR_THRESHOLD
        self._last_pos: tuple[int, int] | None = None

        # Offset of focused template top-left within the full drawn ROI.
        # (0, 0) means the template IS the full ROI (no adjustment needed).
        self._roi_offset:   tuple[int, int]          = roi_offset
        self._full_roi_size: tuple[int, int] | None  = full_roi_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_template(
        self,
        template_gray: np.ndarray,
        roi_offset: tuple[int, int] = (0, 0),
        full_roi_size: tuple[int, int] | None = None,
    ) -> None:
        """Replace the template (called on reference recapture)."""
        self._tpl         = template_gray
        self._th, self._tw = template_gray.shape[:2]
        self._roi_offset  = roi_offset
        self._full_roi_size = full_roi_size
        self._last_pos    = None

    def find(
        self, frame_gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """
        Search for the template in *frame_gray* (full-frame raw grayscale).

        Returns
        -------
        ((x, y, w, h), confidence)  in full-frame pixel coordinates
        (adjusted to the full drawn ROI when a focused template is used), or
        None if the print is not found or the frame is too blurry.
        """
        fh, fw = frame_gray.shape

        # --- choose search region ----------------------------------------
        if self._last_pos is not None:
            lx, ly = self._last_pos
            x1 = max(0, lx - self._margin)
            y1 = max(0, ly - self._margin)
            x2 = min(fw, lx + self._tw + self._margin)
            y2 = min(fh, ly + self._th + self._margin)
            region = frame_gray[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            region = frame_gray
            ox, oy = 0, 0

        if region.shape[0] < self._th or region.shape[1] < self._tw:
            self._last_pos = None
            return None

        # --- template match ----------------------------------------------
        result = cv2.matchTemplate(region, self._tpl, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(result)

        if conf < self._match_thr:
            self._last_pos = None
            return None

        mx = loc[0] + ox
        my = loc[1] + oy

        # Clamp to frame bounds
        mx = int(np.clip(mx, 0, fw - self._tw))
        my = int(np.clip(my, 0, fh - self._th))

        # --- blur gate ---------------------------------------------------
        if self._blur_thr > 0:
            crop = frame_gray[my: my + self._th, mx: mx + self._tw]
            lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if lap_var < self._blur_thr:
                return None

        self._last_pos = (mx, my)

        # --- expand back to full ROI coordinates -------------------------
        ox_roi, oy_roi = self._roi_offset
        if (ox_roi, oy_roi) != (0, 0) and self._full_roi_size is not None:
            full_w, full_h = self._full_roi_size
            rx = max(0, mx - ox_roi)
            ry = max(0, my - oy_roi)
            rx = int(np.clip(rx, 0, fw - full_w))
            ry = int(np.clip(ry, 0, fh - full_h))
            return (rx, ry, full_w, full_h), float(conf)

        return (mx, my, self._tw, self._th), float(conf)

    def reset(self) -> None:
        """Call when a new reference is captured so the next find() does a
        full-frame search rather than searching around a stale position."""
        self._last_pos = None
