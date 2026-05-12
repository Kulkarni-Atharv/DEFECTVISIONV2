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

    Multi-template support
    ----------------------
    Pass a list of templates (one per reference angle) so that different
    print orientations are all matched.  The template with the highest NCC
    score wins each frame.  All templates must have the same height/width
    (they are normalised to a common size by capture_reference_video).

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
        templates: np.ndarray | list,
        roi_offset: tuple[int, int] = (0, 0),
        full_roi_size: tuple[int, int] | None = None,
    ) -> None:
        if isinstance(templates, np.ndarray):
            templates = [templates]
        self._templates: list[np.ndarray] = list(templates)
        self._th, self._tw = self._templates[0].shape[:2]
        self._margin: int      = POSITION_LOCK_SEARCH_MARGIN
        self._match_thr: float = POSITION_LOCK_THRESHOLD
        self._blur_thr:  float = POSITION_LOCK_BLUR_THRESHOLD
        self._last_pos: tuple[int, int] | None = None

        # Offset of focused template top-left within the full drawn ROI.
        # (0, 0) means the template IS the full ROI (no adjustment needed).
        self._roi_offset:    tuple[int, int]         = roi_offset
        self._full_roi_size: tuple[int, int] | None  = full_roi_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_template(
        self,
        templates: np.ndarray | list,
        roi_offset: tuple[int, int] = (0, 0),
        full_roi_size: tuple[int, int] | None = None,
    ) -> None:
        """Replace the templates (called on reference recapture)."""
        if isinstance(templates, np.ndarray):
            templates = [templates]
        self._templates     = list(templates)
        self._th, self._tw  = self._templates[0].shape[:2]
        self._roi_offset    = roi_offset
        self._full_roi_size = full_roi_size
        self._last_pos      = None

    def find(
        self, frame_gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int], float, int] | None:
        """
        Search for any of the stored templates in *frame_gray*.

        Returns
        -------
        ((x, y, w, h), confidence, template_index)  in frame pixel coordinates
        (adjusted to the full drawn ROI when roi_offset is set), or
        None if the print is not found or the frame is too blurry.

        *template_index* is the index into self._templates that gave the best
        match — callers can use it to check the corresponding reference first
        in the inspection loop, avoiding redundant ECC alignment calls.
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

        # --- match all templates, keep the best NCC ----------------------
        best_conf  = -1.0
        best_mx, best_my = 0, 0
        best_tw, best_th = self._tw, self._th
        best_idx   = 0

        rh, rw = region.shape[:2]

        for i, tpl in enumerate(self._templates):
            th_t, tw_t = tpl.shape[:2]

            # If template is larger than the search region in any dimension,
            # take the centre crop that fits — at 90 % of the region size so
            # there is still a meaningful matching window.  Centre-cropping
            # preserves original pixel resolution (text stroke widths stay the
            # same) so NCC stays high, unlike down-scaling which blurs strokes.
            if th_t > rh or tw_t > rw:
                target_h = min(th_t, max(5, int(rh * 0.90)))
                target_w = min(tw_t, max(5, int(rw * 0.90)))
                if target_h < 5 or target_w < 5:
                    continue
                y0 = (th_t - target_h) // 2
                x0 = (tw_t - target_w) // 2
                tpl   = tpl[y0: y0 + target_h, x0: x0 + target_w]
                th_t, tw_t = target_h, target_w

            if th_t > rh or tw_t > rw:
                continue

            res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, conf, _, loc = cv2.minMaxLoc(res)
            if conf > best_conf:
                best_conf = conf
                best_mx   = loc[0] + ox
                best_my   = loc[1] + oy
                best_tw   = tw_t
                best_th   = th_t
                best_idx  = i

        if best_conf < self._match_thr:
            self._last_pos = None
            return None

        mx = int(np.clip(best_mx, 0, fw - best_tw))
        my = int(np.clip(best_my, 0, fh - best_th))

        # --- blur gate ---------------------------------------------------
        if self._blur_thr > 0:
            crop = frame_gray[my: my + best_th, mx: mx + best_tw]
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
            return (rx, ry, full_w, full_h), float(best_conf), best_idx

        return (mx, my, best_tw, best_th), float(best_conf), best_idx

    def reset(self) -> None:
        """Call when a new reference is captured so the next find() does a
        full-frame search rather than searching around a stale position."""
        self._last_pos = None
