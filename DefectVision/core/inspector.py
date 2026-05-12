from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from config import (
    TEXT_BH_SIZE,
    TEXT_MIN_COMPONENT_AREA,
    TEXT_TOLERANCE_PX,
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


# Morphology kernels (module-level — created once)
_CLOSE_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_OPEN_KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_DILATE_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_TOL_KERNEL     = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (TEXT_TOLERANCE_PX * 2 + 1, TEXT_TOLERANCE_PX * 2 + 1),
)
_MIN_CONTOUR_AREA = 15


class Inspector:
    """
    Text-structure inspection engine.

    Binarisation strategy
    ─────────────────────
    Instead of an adaptive mean threshold (sensitive to absolute brightness),
    the engine uses a morphological blackhat (or tophat) transform:

        blackhat = morph_close(gray) − gray

    This extracts dark-ink features by measuring how much darker each pixel
    is than its local neighbourhood.  The result is *independent of absolute
    brightness* — the same ink on a dark-red background and on a white
    background produces the same blackhat response.

    Polarity (dark text vs light text) is detected *once* from the reference
    image and locked in for all subsequent live frames.

    Comparison strategy
    ───────────────────
    Before computing missing / extra pixels, both binary masks are dilated
    by TEXT_TOLERANCE_PX.  Sub-pixel stroke-boundary differences caused by
    slight lighting or threshold variation are absorbed, while genuine defects
    (missing characters, smears, debris) still cross the tolerance boundary
    and are detected.

    Three signals → composite defect score:
        Recall  – fraction of reference ink present in live  (missing chars)
        Purity  – inverse fraction of extra ink in live       (smear / debris)
        NCC     – normalised cross-correlation of masks        (shape distortion)
    """

    def __init__(self) -> None:
        self._ref_bin:  np.ndarray | None = None
        self._ref_area: int = 0
        self._polarity: str = 'dark'   # 'dark' | 'light' — set on first reference

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        """Detect polarity, binarise, and cache the reference ink mask."""
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

        # Binarise live with the SAME polarity as reference so that a
        # lighting-induced polarity flip cannot produce a false defect.
        live_bin = self._binarize(live, self._polarity)

        # ---- Tolerance dilation -----------------------------------------
        # Expand each mask by TEXT_TOLERANCE_PX before comparing so that
        # stroke-boundary pixels shifted by lighting / threshold variation
        # are absorbed without hiding real (larger) defects.
        ref_tol  = cv2.dilate(ref_bin,  _TOL_KERNEL)
        live_tol = cv2.dilate(live_bin, _TOL_KERNEL)

        # ---- Signal 1: Recall — missing ink -----------------------------
        missing = cv2.bitwise_and(ref_bin, cv2.bitwise_not(live_tol))
        recall  = 1.0 - float(np.count_nonzero(missing)) / ref_area

        # ---- Signal 2: Purity — extra ink --------------------------------
        extra  = cv2.bitwise_and(live_bin, cv2.bitwise_not(ref_tol))
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
        result.defect_score  = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect     = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Map to legacy fields (visualiser compatibility) ------------
        result.ssim_score       = ncc
        result.edge_diff_score  = 1.0 - recall
        result.pixel_diff_score = float(np.clip(1.0 - purity, 0.0, 1.0))
        result.diff_map         = cv2.add(missing, extra)
        result.edge_diff_map    = missing

        # ssim_map: float [0,1] — 1=OK, 0=defect — drives heatmap renderer
        synth = np.ones(reference.shape, dtype=np.float64)
        synth[missing > 0] = 0.0    # missing ink → hot (most severe)
        synth[extra   > 0] = 0.35   # extra ink   → warm
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
    # Binarisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_polarity(gray: np.ndarray) -> str:
        """
        Return 'dark' if text is dark-on-bright, 'light' if text is
        light-on-dark.  Compares the mean response of blackhat vs tophat —
        whichever is stronger indicates the correct ink polarity.
        """
        kernel = Inspector._bh_kernel(gray)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   kernel)
        return 'dark' if float(bh.mean()) >= float(th.mean()) else 'light'

    @staticmethod
    def _binarize(gray: np.ndarray, polarity: str = 'dark') -> np.ndarray:
        """
        Convert preprocessed gray to a clean binary ink mask (white = ink).

        Uses morphological blackhat (dark text) or tophat (light text) to
        measure local ink contrast independently of absolute brightness.
        Otsu thresholds the result, so the same ink on any background
        produces a consistent binary mask.
        """
        kernel = Inspector._bh_kernel(gray)

        if polarity == 'dark':
            features = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            features = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Denoise before Otsu — suppresses CLAHE/background artefacts so the
        # threshold falls cleanly between background (≈0) and ink (positive).
        features = cv2.GaussianBlur(features, (3, 3), 0)
        _, bin_mask = cv2.threshold(features, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Connect broken strokes; remove single-pixel noise
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  _OPEN_KERNEL)

        # Drop tiny components (sensor noise, CLAHE artefacts)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        cleaned = np.zeros_like(bin_mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= TEXT_MIN_COMPONENT_AREA:
                cleaned[labels == i] = 255

        return cleaned

    @staticmethod
    def _bh_kernel(gray: np.ndarray) -> np.ndarray:
        """
        Return a square structuring element sized for this ROI.
        Capped at ROI_min_dimension / 3 so it never spans the whole image.
        """
        size = min(TEXT_BH_SIZE, min(gray.shape) // 3)
        size = max(size, 3)
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
