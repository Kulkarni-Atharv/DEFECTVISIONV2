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
    Locates the print region in each frame via normalized cross-correlation
    template matching, replacing the fixed-ROI crop used in the static setup.

    Strategy
    --------
    First call (or after losing track): full-frame search — expensive but rare.
    Subsequent calls: restricted ±SEARCH_MARGIN window around last known
    position — typically completes in < 5 ms on CM5 at 1456×1088.

    The template is raw grayscale (no CLAHE) so it stays stable across
    moderate lighting variation and matches against the equally raw grayscale
    of each incoming frame.  The CLAHE step is applied *after* the crop is
    extracted, inside the normal preprocessor pipeline.

    Gates
    -----
    • Match confidence < POSITION_LOCK_THRESHOLD  → no match, skip frame.
    • Laplacian variance of matched crop < POSITION_LOCK_BLUR_THRESHOLD
      → motion-blurred frame, skip.  Set threshold to 0 to disable.
    """

    def __init__(self, template_gray: np.ndarray) -> None:
        self._tpl: np.ndarray = template_gray
        self._th, self._tw = template_gray.shape[:2]
        self._margin: int   = POSITION_LOCK_SEARCH_MARGIN
        self._match_thr: float = POSITION_LOCK_THRESHOLD
        self._blur_thr: float  = POSITION_LOCK_BLUR_THRESHOLD
        self._last_pos: tuple[int, int] | None = None   # (x, y) top-left

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(
        self, frame_gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """
        Search for the template in *frame_gray* (full-frame raw grayscale).

        Returns
        -------
        ((x, y, w, h), confidence)  in full-frame pixel coordinates, or
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

        # Clamp to frame bounds (safety)
        mx = int(np.clip(mx, 0, fw - self._tw))
        my = int(np.clip(my, 0, fh - self._th))

        # --- blur gate ---------------------------------------------------
        if self._blur_thr > 0:
            crop = frame_gray[my: my + self._th, mx: mx + self._tw]
            lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if lap_var < self._blur_thr:
                return None   # motion-blurred; don't update last_pos either

        self._last_pos = (mx, my)
        return (mx, my, self._tw, self._th), float(conf)

    def reset(self) -> None:
        """Call when a new reference is captured so the next find() does a
        full-frame search rather than searching around a stale position."""
        self._last_pos = None
