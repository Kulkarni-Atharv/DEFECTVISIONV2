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

_MIN_CONTOUR_AREA = 15


class Inspector:
    """
    Text-structure inspection engine — position-invariant.

    How it works
    ────────────
    The text is located independently in the reference and in the live
    frame by binarising each image and taking the bounding box of all
    ink pixels.  Both images are then cropped to their respective text
    regions (plus TEXT_CROP_MARGIN px on every side to capture nearby
    debris) and the live crop is resized to match the reference crop.

    Comparing crops instead of full ROI images means the algorithm is
    completely insensitive to WHERE the text sits in the frame.  A
    cylinder rotating past the camera will place the text at different
    horizontal positions on each pass — this produces zero false
    positives.  Only changes to the TEXT SHAPE itself (debris added,
    ink missing, smear) change the comparison result.

    Signals — computed on the text crops
    ─────────────────────────────────────
    Recall   – fraction of reference ink present in live  (missing characters)
    Purity   – inverse fraction of extra ink in live       (smear / debris)
    NCC      – normalised cross-correlation of binary masks (shape distortion)

    Debris hard override
    ────────────────────
    Any extra-ink connected component ≥ DEBRIS_MIN_COMPONENT_AREA px²
    that survived the purity tolerance is flagged as DEFECT regardless
    of the composite score.
    """

    def __init__(self) -> None:
        self._ref_bin:  np.ndarray | None = None
        self._ref_area: int = 0
        self._polarity: str = 'dark'
        self._ref_bbox: tuple | None = None   # (x, y, w, h) of text in ref
        # Cache keyed by (data_ptr, nbytes) — stable for the lifetime of
        # each captured reference array; cleared on reference recapture.
        self._cache: dict[tuple, tuple] = {}
        # Live-frame binarization cache — reused across all reference comparisons
        # for the same live frame so binarization runs once per frame, not once per ref.
        self._live_key:  tuple | None = None
        self._live_bin:  np.ndarray | None = None
        self._live_bbox: tuple | None = None

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
        self._live_key = self._live_bin = self._live_bbox = None

    def precompute_live(self, live: np.ndarray) -> None:
        """Binarize the live frame once.  inspect() reuses the result for every
        reference comparison in the same frame — avoids repeating the expensive
        morphology+Otsu pipeline N times for N references."""
        key = (live.ctypes.data, live.nbytes)
        if key == self._live_key:
            return
        polarity = self._polarity or self._detect_polarity(live)
        self._live_bin  = self._binarize(live, polarity)
        self._live_bbox = Inspector._text_bbox(self._live_bin, TEXT_CROP_MARGIN)
        self._live_key  = key

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()
        H, W   = reference.shape[:2]

        live_key = (live.ctypes.data, live.nbytes)

        if reference.shape != live.shape:
            live = cv2.resize(live, (W, H))

        if self._ref_bin is None or self._ref_bin.shape != reference.shape:
            self.set_reference(reference)

        ref_bin = self._ref_bin

        # Reuse precomputed live binarization if available (set by precompute_live).
        # Falls back to computing fresh when the live frame changed or resize occurred.
        if self._live_key == live_key and self._live_bin is not None:
            live_bin  = self._live_bin
            live_bbox = self._live_bbox
        else:
            live_bin  = self._binarize(live, self._polarity)
            live_bbox = Inspector._text_bbox(live_bin, TEXT_CROP_MARGIN)

        # ---- Text-centric crop -----------------------------------------
        # Locate text in both images independently, crop to just the text
        # region so positional shift does not affect the comparison.
        ref_bbox  = self._ref_bbox

        if ref_bbox is not None and live_bbox is not None:
            rx, ry, rw, rh = ref_bbox
            lx, ly, lw, lh = live_bbox
            cmp_ref  = ref_bin[ry:ry + rh, rx:rx + rw]
            cmp_live = live_bin[ly:ly + lh, lx:lx + lw]
            # Resize live crop to reference crop dimensions so the binary
            # masks are the same size for set-difference operations.
            if cmp_live.shape != cmp_ref.shape:
                cmp_live = cv2.resize(
                    cmp_live, (rw, rh), interpolation=cv2.INTER_NEAREST
                )
            cmp_area = max(int(np.count_nonzero(cmp_ref)), 1)
            ox, oy   = lx, ly   # live-crop origin in full-ROI coords
        else:
            # Fallback: text not detected in one or both — full-image compare
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

        # ---- Signal 2: Purity — extra ink ------------------------------
        extra  = cv2.bitwise_and(cmp_live, cv2.bitwise_not(ref_tol_purity))
        purity = 1.0 - float(np.count_nonzero(extra)) / cmp_area

        # ---- Signal 3: NCC — structural shape --------------------------
        ref_f  = cmp_ref.astype(np.float64)
        live_f = cmp_live.astype(np.float64)
        ref_m  = ref_f  - ref_f.mean()
        live_m = live_f - live_f.mean()
        denom  = np.sqrt(np.sum(ref_m ** 2) * np.sum(live_m ** 2))
        ncc    = float(np.clip(np.sum(ref_m * live_m) / (denom + 1e-8), 0.0, 1.0))

        # ---- Composite score -------------------------------------------
        defect_score = (
            RECALL_WEIGHT * (1.0 - recall) +
            PURITY_WEIGHT * float(np.clip(1.0 - purity, 0.0, 1.0)) +
            NCC_WEIGHT    * (1.0 - ncc)
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect    = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Debris hard override --------------------------------------
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

        # ---- Full-size maps (visualiser expects ROI-sized arrays) ------
        # Embed crop results at the live text position so the heatmap
        # shows defects where they actually appear in the frame.
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

        # ---- Defect localisation (in full-ROI coordinates) -------------
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
        """Bounding box of all ink pixels + margin. Returns (x,y,w,h) or None."""
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
        _, bin_mask = cv2.threshold(features, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
