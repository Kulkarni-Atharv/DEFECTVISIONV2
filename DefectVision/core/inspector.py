from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from config import (
    TEXT_BH_SIZE,
    TEXT_MIN_COMPONENT_AREA,
    TEXT_TOLERANCE_RECALL_PX,
    TEXT_TOLERANCE_PURITY_PX,
    DEBRIS_MIN_COMPONENT_AREA,
    RECALL_WEIGHT,
    PURITY_WEIGHT,
    NCC_WEIGHT,
    DEFECT_SCORE_THRESHOLD,
    TEXT_CROP_MARGIN,
    DEBRIS_TEXT_ZONE_PX,
    MISMATCH_SYM_THRESH,
)


@dataclass
class InspectionResult:
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0
    is_defect: bool = False

    ssim_map: np.ndarray | None = None
    diff_map: np.ndarray | None = None
    edge_diff_map: np.ndarray | None = None

    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    is_angle_mismatch: bool = False
    is_text_found: bool = True       # False when no ink blobs found in live ROI
    is_recognized_angle: bool = True # False when no reference matched this angle


# ---- Morphology kernels (built once at import) --------------------------
_CLOSE_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_OPEN_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_TOL_RECALL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (TEXT_TOLERANCE_RECALL_PX * 2 + 1, TEXT_TOLERANCE_RECALL_PX * 2 + 1),
)
_TOL_PURITY = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (TEXT_TOLERANCE_PURITY_PX * 2 + 1, TEXT_TOLERANCE_PURITY_PX * 2 + 1),
)
# Zone around reference strokes where extra ink is allowed to be flagged.
# Extra ink OUTSIDE this zone is a background/margin artifact — ignored.
_ZONE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (DEBRIS_TEXT_ZONE_PX * 2 + 1, DEBRIS_TEXT_ZONE_PX * 2 + 1),
)
# Wider zone used by debris-only mode to define "near text" for component search.
_DEBRIS_ZONE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

_MIN_CONTOUR_AREA = 15


class Inspector:
    """
    Text-structure inspection engine — two operating modes.

    MODE 1 — Reference comparison  (NCC ≥ NCC_MATCH_THRESHOLD)
    ────────────────────────────────────────────────────────────
    The text is located independently in the reference and live image
    via binarisation + bounding-box.  Both are cropped to their text
    regions and the crops are compared shape-for-shape (position-invariant).

    Purity (extra-ink) is restricted to a zone around the reference text
    strokes (DEBRIS_TEXT_ZONE_PX dilation).  Background artifacts at the
    crop margin — surface creases, lighting edges — that are not near any
    character stroke are silently ignored.

    MODE 2 — Debris-only  (NCC < NCC_MATCH_THRESHOLD)
    ───────────────────────────────────────────────────
    Called when no captured reference angle matches the current frame
    well enough for reliable shape comparison.  No reference is used.

    Algorithm:
      1. Binarise live frame → all ink blobs
      2. Classify blobs: large ones (≥ 30 % of largest) = characters
      3. Dilate character mask to define "near text" zone
      4. Small blobs (≥ DEBRIS_MIN_COMPONENT_AREA) inside the zone = debris
      5. Any such blob → DEFECT

    This gives the system independent judgement: it detects debris ON the
    text without caring where on the cylinder the text currently sits.
    Missing-character detection is not performed in this mode (can't
    distinguish angle-mismatch from genuinely missing ink).
    """

    def __init__(self) -> None:
        self._ref_bin:  np.ndarray | None = None
        self._ref_area: int = 0
        self._polarity: str = 'dark'
        self._ref_bbox: tuple | None = None
        self._cache: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        key = (ref_gray.ctypes.data, ref_gray.nbytes)
        if key not in self._cache:
            polarity = self._detect_polarity(ref_gray)
            ref_bin  = self._binarize(ref_gray, polarity)
            ref_area = max(int(np.count_nonzero(ref_bin)), 1)
            ref_bbox = Inspector._text_bbox(ref_bin, TEXT_CROP_MARGIN)
            self._cache[key] = (ref_bin, ref_area, polarity, ref_bbox)
        self._ref_bin, self._ref_area, self._polarity, self._ref_bbox = \
            self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Mode 1 — Reference comparison
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()
        H, W   = reference.shape[:2]

        if reference.shape != live.shape:
            live = cv2.resize(live, (W, H))

        if self._ref_bin is None or self._ref_bin.shape != reference.shape:
            self.set_reference(reference)

        ref_bin  = self._ref_bin
        live_bin = self._binarize(live, self._polarity)

        # ---- Text-centric crop (position-invariant) --------------------
        ref_bbox  = self._ref_bbox
        live_bbox = Inspector._text_bbox(live_bin, TEXT_CROP_MARGIN)

        if ref_bbox is not None and live_bbox is not None:
            rx, ry, rw, rh = ref_bbox
            lx, ly, lw, lh = live_bbox
            cmp_ref  = ref_bin[ry:ry + rh, rx:rx + rw]
            cmp_live = live_bin[ly:ly + lh, lx:lx + lw]
            if cmp_live.shape != cmp_ref.shape:
                cmp_live = cv2.resize(
                    cmp_live, (rw, rh), interpolation=cv2.INTER_NEAREST
                )
            cmp_area = max(int(np.count_nonzero(cmp_ref)), 1)
            ox, oy   = lx, ly
        else:
            cmp_ref  = ref_bin
            cmp_live = live_bin
            cmp_area = self._ref_area
            ox, oy   = 0, 0

        cph, cpw = cmp_ref.shape[:2]

        # ---- Signal 1: Recall — missing ink ----------------------------
        live_tol_recall = cv2.dilate(cmp_live, _TOL_RECALL)
        ref_tol_purity  = cv2.dilate(cmp_ref,  _TOL_PURITY)
        missing = cv2.bitwise_and(cmp_ref, cv2.bitwise_not(live_tol_recall))
        recall  = 1.0 - float(np.count_nonzero(missing)) / cmp_area

        # ---- Signal 2: Purity — extra ink, zone-restricted -------------
        # Only flag extra pixels that fall within DEBRIS_TEXT_ZONE_PX of a
        # reference stroke.  Margin/background artifacts are ignored.
        text_zone   = cv2.dilate(cmp_ref, _ZONE_KERNEL)
        extra_raw   = cv2.bitwise_and(cmp_live, cv2.bitwise_not(ref_tol_purity))
        extra       = cv2.bitwise_and(extra_raw, text_zone)
        purity      = 1.0 - float(np.count_nonzero(extra)) / cmp_area

        # ---- Signal 3: NCC — structural shape --------------------------
        ref_f  = cmp_ref.astype(np.float64)
        live_f = cmp_live.astype(np.float64)
        ref_m  = ref_f  - ref_f.mean()
        live_m = live_f - live_f.mean()
        denom  = np.sqrt(np.sum(ref_m ** 2) * np.sum(live_m ** 2))
        ncc    = float(np.clip(np.sum(ref_m * live_m) / (denom + 1e-8), 0.0, 1.0))

        # ---- Symmetric mismatch check ----------------------------------
        # Angle mismatch signature: strokes shift → they appear BOTH
        # missing (from reference POV) AND extra (in live but offset).
        # When both recall and purity are simultaneously degraded, that is
        # angle mismatch, NOT a real defect.
        #
        # Asymmetric signals = real defects:
        #   only purity high  → debris/cross-line added (text still present)
        #   only recall high  → text faded or character missing
        #
        # When mismatch is detected: return a neutral result so
        # _run_detection() tries the next reference or falls back to
        # debris-only mode.  The hard override is intentionally skipped
        # because shifted strokes would also form large "extra" components.
        missing_frac = 1.0 - recall
        extra_frac   = float(np.clip(1.0 - purity, 0.0, 1.0))
        if missing_frac > MISMATCH_SYM_THRESH and extra_frac > MISMATCH_SYM_THRESH:
            result.ssim_score        = ncc
            result.edge_diff_score   = missing_frac
            result.pixel_diff_score  = extra_frac
            result.is_angle_mismatch = True
            result.diff_map          = np.zeros((H, W), dtype=np.uint8)
            result.edge_diff_map     = np.zeros((H, W), dtype=np.uint8)
            result.ssim_map          = np.ones((H, W),  dtype=np.float64)
            return result

        # ---- Composite score -------------------------------------------
        defect_score = (
            RECALL_WEIGHT * missing_frac +
            PURITY_WEIGHT * extra_frac +
            NCC_WEIGHT    * (1.0 - ncc)
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect    = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Debris hard override (zone-restricted) --------------------
        if not result.is_defect and np.count_nonzero(extra) > 0:
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(
                extra, connectivity=8
            )
            max_comp = int(max(
                (stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lbl)),
                default=0,
            ))
            if max_comp >= DEBRIS_MIN_COMPONENT_AREA:
                result.is_defect    = True
                result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- Full-size maps for visualiser -----------------------------
        full_diff = np.zeros((H, W), dtype=np.uint8)
        full_edge = np.zeros((H, W), dtype=np.uint8)
        full_ssim = np.ones((H, W),  dtype=np.float64)

        y2 = min(oy + cph, H);  ch_ = y2 - oy
        x2 = min(ox + cpw, W);  cw_ = x2 - ox

        crop_diff = cv2.add(missing, extra)
        synth     = np.ones(cmp_ref.shape, dtype=np.float64)
        synth[missing > 0] = 0.0
        synth[extra   > 0] = 0.35

        full_diff[oy:y2, ox:x2] = crop_diff[:ch_, :cw_]
        full_edge[oy:y2, ox:x2] = missing[:ch_,   :cw_]
        full_ssim[oy:y2, ox:x2] = synth[:ch_,     :cw_]

        result.ssim_score       = ncc
        result.edge_diff_score  = 1.0 - recall
        result.pixel_diff_score = float(np.clip(1.0 - purity, 0.0, 1.0))
        result.diff_map         = full_diff
        result.edge_diff_map    = full_edge
        result.ssim_map         = full_ssim

        # ---- Defect localisation ---------------------------------------
        defect_px = crop_diff.copy()
        defect_px = cv2.morphologyEx(defect_px, cv2.MORPH_CLOSE,  _DILATE_KERNEL)
        defect_px = cv2.morphologyEx(defect_px, cv2.MORPH_DILATE, _DILATE_KERNEL)
        contours, _ = cv2.findContours(
            defect_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [
            c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA
        ]
        result.defect_bboxes = [
            (x + ox, y + oy, bw, bh)
            for c in result.defect_contours
            for x, y, bw, bh in [cv2.boundingRect(c)]
        ]
        return result

    # ------------------------------------------------------------------
    # Mode 2 — Reference-free debris detection
    # ------------------------------------------------------------------
    def inspect_debris_only(self, live: np.ndarray) -> InspectionResult:
        """
        Detects debris ON or near text without using any reference image.
        Called when no captured angle matches the current frame.

        Logic:
          - Large ink blobs  → text characters (expected, OK)
          - Small ink blobs that are NEAR a character → debris (DEFECT)
          - Small ink blobs far from any character    → background noise (ignored)
        """
        result = InspectionResult()
        H, W   = live.shape[:2]

        # Full-size clean maps as default
        result.diff_map      = np.zeros((H, W), dtype=np.uint8)
        result.edge_diff_map = np.zeros((H, W), dtype=np.uint8)
        result.ssim_map      = np.ones((H, W),  dtype=np.float64)

        live_bin = self._binarize(live, self._polarity)
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            live_bin, connectivity=8
        )
        if n_lbl <= 1:
            result.is_text_found = False  # no ink at all in ROI
            return result

        areas    = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lbl)]
        max_area = max(areas)
        char_min = max_area * 0.30   # blobs ≥ 30 % of largest = characters

        # Separate character blobs from small blobs
        char_mask  = np.zeros_like(live_bin)
        candidates = []
        for i in range(1, n_lbl):
            a = stats[i, cv2.CC_STAT_AREA]
            if a >= char_min:
                char_mask[labels == i] = 255
            elif a >= DEBRIS_MIN_COMPONENT_AREA:
                candidates.append(i)

        if not candidates:
            return result  # no small blobs near text

        # Zone around characters where debris is meaningful
        text_zone    = cv2.dilate(char_mask, _DEBRIS_ZONE_KERNEL)
        debris_mask  = np.zeros_like(live_bin)

        for i in candidates:
            comp = ((labels == i).astype(np.uint8) * 255)
            if np.count_nonzero(cv2.bitwise_and(comp, text_zone)) > 0:
                debris_mask = cv2.bitwise_or(debris_mask, comp)

        if np.count_nonzero(debris_mask) == 0:
            return result

        result.is_defect    = True
        result.defect_score = DEFECT_SCORE_THRESHOLD
        result.diff_map     = debris_mask
        result.ssim_map[debris_mask > 0] = 0.0

        contours, _ = cv2.findContours(
            debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [
            c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA
        ]
        result.defect_bboxes = [cv2.boundingRect(c) for c in result.defect_contours]
        return result

    # ------------------------------------------------------------------
    # Multi-reference entry point
    # ------------------------------------------------------------------
    def inspect_best(self, references: list, live: np.ndarray) -> tuple:
        best_result = None
        best_ref    = references[0]
        for ref in references:
            self.set_reference(ref)
            result = self.inspect(ref, live)
            if best_result is None or result.defect_score < best_result.defect_score:
                best_result = result
                best_ref    = ref
        return best_result, best_ref  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Binarisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _text_bbox(bin_mask: np.ndarray, margin: int = 0) -> tuple | None:
        pts = cv2.findNonZero(bin_mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        H, W = bin_mask.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(W, x + w + margin)
        y2 = min(H, y + h + margin)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _detect_polarity(gray: np.ndarray) -> str:
        kernel = Inspector._bh_kernel(gray)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   kernel)
        return 'dark' if float(bh.mean()) >= float(th.mean()) else 'light'

    @staticmethod
    def _binarize(gray: np.ndarray, polarity: str = 'dark') -> np.ndarray:
        kernel = Inspector._bh_kernel(gray)
        if polarity == 'dark':
            features = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            features = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        features = cv2.GaussianBlur(features, (3, 3), 0)
        _, bin_mask = cv2.threshold(
            features, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  _OPEN_KERNEL)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )
        cleaned = np.zeros_like(bin_mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= TEXT_MIN_COMPONENT_AREA:
                cleaned[labels == i] = 255
        return cleaned

    @staticmethod
    def _bh_kernel(gray: np.ndarray) -> np.ndarray:
        size = min(TEXT_BH_SIZE, min(gray.shape) // 3)
        size = max(size, 3)
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
