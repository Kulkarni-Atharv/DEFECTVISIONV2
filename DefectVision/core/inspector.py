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
)


@dataclass
class InspectionResult:
    # ---- Scalar scores (kept compatible with visualiser) ----------------
    # ssim_score       → NCC structural similarity  (1 = identical)
    # edge_diff_score  → missing-text ratio          (0 = nothing missing)
    # pixel_diff_score → extra-ink ratio             (0 = no extra ink)
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0
    is_defect: bool = False

    # ---- Per-pixel maps (visualiser compatible) -------------------------
    ssim_map: np.ndarray | None = None       # float64 [0,1]: 1=OK 0=defect
    diff_map: np.ndarray | None = None       # uint8: missing | extra pixels
    edge_diff_map: np.ndarray | None = None  # uint8: missing-ink binary map

    # ---- Localisation ---------------------------------------------------
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# ---- Morphology kernels (built once at import) --------------------------
_CLOSE_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_OPEN_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Recall tolerance: 3 px — absorbs stroke-boundary shifts from lighting /
# threshold variation so that a character isn't falsely flagged as missing
# just because its edge is 1-2 px off.
_TOL_RECALL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (TEXT_TOLERANCE_RECALL_PX * 2 + 1, TEXT_TOLERANCE_RECALL_PX * 2 + 1),
)

# Purity tolerance: 1 px — only absorbs single-pixel aliasing noise.
# Keeping this tight means marks drawn ON or directly beside a character
# stroke are NOT absorbed and will be detected as debris.
_TOL_PURITY = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (TEXT_TOLERANCE_PURITY_PX * 2 + 1, TEXT_TOLERANCE_PURITY_PX * 2 + 1),
)

_MIN_CONTOUR_AREA = 15


class Inspector:
    """
    Text-structure inspection engine.

    Binarisation
    ────────────
    Morphological blackhat (dark text) / tophat (light text) isolates ink
    by local contrast — independent of absolute brightness or surface colour.
    Polarity is detected once from the reference and locked for all live frames.

    Comparison — three signals on binary ink masks
    ───────────────────────────────────────────────
    Recall   – fraction of reference ink present in live  (missing characters)
    Purity   – inverse fraction of extra ink in live       (smear / debris)
    NCC      – normalised cross-correlation of masks        (shape distortion)

    Separate tolerance dilations
    ────────────────────────────
    • RECALL uses a 3 px dilation: forgiving of stroke-boundary shifts caused
      by lighting / threshold variation → no false "missing ink" detections.
    • PURITY uses a 1 px dilation: tight, so marks drawn ON or immediately
      beside a character stroke are NOT absorbed and are detected as debris.

    Debris hard override
    ────────────────────
    After the purity check, connected components of the surviving extra-ink
    pixels are analysed.  Any single component ≥ DEBRIS_MIN_COMPONENT_AREA px²
    is genuine debris (dot, added stroke, smear) and immediately flags the
    frame as DEFECT — regardless of the composite score.
    """

    def __init__(self) -> None:
        self._ref_bin:  np.ndarray | None = None
        self._ref_area: int = 0
        self._polarity: str = 'dark'

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        self._polarity = self._detect_polarity(ref_gray)
        self._ref_bin  = self._binarize(ref_gray, self._polarity)
        self._ref_area = max(int(np.count_nonzero(self._ref_bin)), 1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        if self._ref_bin is None or self._ref_bin.shape != reference.shape:
            self.set_reference(reference)

        ref_bin  = self._ref_bin
        ref_area = self._ref_area

        live_bin = self._binarize(live, self._polarity)

        # ---- Dilated masks for tolerance comparison ---------------------
        # Recall uses generous 3 px dilation (boundary-shift forgiveness).
        # Purity uses tight  1 px dilation (debris sensitivity).
        live_tol_recall = cv2.dilate(live_bin, _TOL_RECALL)
        ref_tol_purity  = cv2.dilate(ref_bin,  _TOL_PURITY)

        # ---- Signal 1: Recall — missing ink -----------------------------
        missing = cv2.bitwise_and(ref_bin, cv2.bitwise_not(live_tol_recall))
        recall  = 1.0 - float(np.count_nonzero(missing)) / ref_area

        # ---- Signal 2: Purity — extra ink (tight tolerance) -------------
        extra  = cv2.bitwise_and(live_bin, cv2.bitwise_not(ref_tol_purity))
        purity = 1.0 - float(np.count_nonzero(extra)) / ref_area

        # ---- Signal 3: NCC — structural shape ---------------------------
        ref_f  = ref_bin.astype(np.float64)
        live_f = live_bin.astype(np.float64)
        ref_m  = ref_f  - ref_f.mean()
        live_m = live_f - live_f.mean()
        denom  = np.sqrt(np.sum(ref_m ** 2) * np.sum(live_m ** 2))
        ncc    = float(np.clip(np.sum(ref_m * live_m) / (denom + 1e-8), 0.0, 1.0))

        # ---- Composite score --------------------------------------------
        defect_score = (
            RECALL_WEIGHT * (1.0 - recall) +
            PURITY_WEIGHT * float(np.clip(1.0 - purity, 0.0, 1.0)) +
            NCC_WEIGHT    * (1.0 - ncc)
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect    = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Debris hard override ---------------------------------------
        # A connected extra-ink component ≥ DEBRIS_MIN_COMPONENT_AREA that
        # survived the tight purity tolerance is genuine debris — flag it
        # regardless of the composite score so small dots / lines are never
        # missed just because ref_area is large.
        if not result.is_defect and np.count_nonzero(extra) > 0:
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(extra, connectivity=8)
            max_comp = int(max(
                (stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lbl)),
                default=0,
            ))
            if max_comp >= DEBRIS_MIN_COMPONENT_AREA:
                result.is_defect    = True
                result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- Map to legacy fields (visualiser compatibility) ------------
        result.ssim_score       = ncc
        result.edge_diff_score  = 1.0 - recall
        result.pixel_diff_score = float(np.clip(1.0 - purity, 0.0, 1.0))
        result.diff_map         = cv2.add(missing, extra)
        result.edge_diff_map    = missing

        synth = np.ones(reference.shape, dtype=np.float64)
        synth[missing > 0] = 0.0
        synth[extra   > 0] = 0.35
        result.ssim_map = synth

        # ---- Defect localisation ----------------------------------------
        defect_px = cv2.add(missing, extra)
        defect_px = cv2.morphologyEx(defect_px, cv2.MORPH_CLOSE,  _DILATE_KERNEL)
        defect_px = cv2.morphologyEx(defect_px, cv2.MORPH_DILATE, _DILATE_KERNEL)

        contours, _ = cv2.findContours(
            defect_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
        result.defect_bboxes   = [cv2.boundingRect(c) for c in result.defect_contours]

        return result

    # ------------------------------------------------------------------
    # Multi-reference entry point
    # ------------------------------------------------------------------
    def inspect_best(
        self,
        references: list,
        live: np.ndarray,
    ) -> tuple:
        """
        Run inspect() against every reference frame; return
        (best_result, best_ref) where best_result has the lowest
        defect_score (the reference that structurally matches live best).

        Each reference gets its own set_reference() call so _ref_bin
        is always correct for that specific frame.
        """
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

        # Denoise before Otsu — prevents noise from splitting the threshold
        features = cv2.GaussianBlur(features, (3, 3), 0)
        _, bin_mask = cv2.threshold(features, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  _OPEN_KERNEL)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
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
