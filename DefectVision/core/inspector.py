from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from config import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    TEXT_MIN_COMPONENT_AREA,
    RECALL_WEIGHT,
    PURITY_WEIGHT,
    NCC_WEIGHT,
    DEFECT_SCORE_THRESHOLD,
)


@dataclass
class InspectionResult:
    # ---- Scalar scores (kept compatible with visualiser) ----------------
    # ssim_score  → NCC structural similarity   (1 = identical, 0 = different)
    # edge_diff_score → missing-text ratio      (0 = nothing missing)
    # pixel_diff_score → extra-ink ratio        (0 = no extra ink)
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0
    is_defect: bool = False

    # ---- Per-pixel maps -------------------------------------------------
    ssim_map: np.ndarray | None = None       # float64 [0,1]: 1=OK 0=defect (for heatmap)
    diff_map: np.ndarray | None = None       # uint8: missing | extra pixels
    edge_diff_map: np.ndarray | None = None  # uint8: missing-ink binary map

    # ---- Localisation ---------------------------------------------------
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# Morphology kernels
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_MIN_CONTOUR_AREA = 15


class Inspector:
    """
    Text-structure inspection engine.

    Instead of comparing raw pixel intensities (SSIM / pixel-diff / edge-diff),
    this engine compares the *binary ink structure* of the text:

      1. Both reference and live are adaptively thresholded into binary
         ink masks — completely immune to illumination differences.

      2. Three targeted signals are computed on those masks:

         Recall  — fraction of reference ink pixels present in the live frame.
                   Catches missing characters, faded ink, broken strokes.

         Purity  — inverse fraction of extra ink in live relative to reference.
                   Catches smears, debris, added characters, bleed.

         NCC     — normalised cross-correlation of the two binary masks.
                   Catches structural distortion even when pixel counts match.

      3. Composite defect score:
            score = RECALL_W × (1−recall) + PURITY_W × (1−purity) + NCC_W × (1−ncc)

    Because comparison operates on ink masks, ROI background, bottle surface,
    roller movement, and illumination variation are completely invisible to
    the algorithm.
    """

    def __init__(self) -> None:
        self._ref_bin: np.ndarray | None = None   # cleaned binary reference mask
        self._ref_area: int = 0                    # number of ink pixels in reference

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        """Binarise and cache the reference ink mask."""
        self._ref_bin  = self._binarize(ref_gray)
        self._ref_area = max(int(np.count_nonzero(self._ref_bin)), 1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        # Recompute reference bin if not yet set (safety guard)
        if self._ref_bin is None or self._ref_bin.shape != reference.shape:
            self.set_reference(reference)

        ref_bin  = self._ref_bin
        ref_area = self._ref_area

        # ---- Binarise live frame ----------------------------------------
        live_bin = self._binarize(live)

        # ---- Signal 1: Recall — missing ink -----------------------------
        # Pixels that are ink in reference but not in live.
        missing = cv2.bitwise_and(ref_bin, cv2.bitwise_not(live_bin))
        recall  = 1.0 - float(np.count_nonzero(missing)) / ref_area

        # ---- Signal 2: Purity — extra ink -------------------------------
        # Pixels that are ink in live but not in reference (normalised by
        # ref_area so the score is relative to expected ink volume).
        extra   = cv2.bitwise_and(live_bin, cv2.bitwise_not(ref_bin))
        purity  = 1.0 - float(np.count_nonzero(extra)) / ref_area

        # ---- Signal 3: NCC — structural shape ---------------------------
        ref_f  = ref_bin.astype(np.float64)
        live_f = live_bin.astype(np.float64)
        ref_m  = ref_f  - ref_f.mean()
        live_m = live_f - live_f.mean()
        denom  = np.sqrt(np.sum(ref_m ** 2) * np.sum(live_m ** 2))
        ncc    = float(np.sum(ref_m * live_m) / (denom + 1e-8))
        ncc    = float(np.clip(ncc, 0.0, 1.0))

        # ---- Composite score --------------------------------------------
        defect_score = (
            RECALL_WEIGHT * (1.0 - recall) +
            PURITY_WEIGHT * float(np.clip(1.0 - purity, 0.0, 1.0)) +
            NCC_WEIGHT    * (1.0 - ncc)
        )
        result.defect_score    = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect       = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Map to legacy fields (visualiser compatibility) ------------
        result.ssim_score        = ncc              # structure similarity
        result.edge_diff_score   = 1.0 - recall     # missing-text ratio
        result.pixel_diff_score  = float(np.clip(1.0 - purity, 0.0, 1.0))  # extra-ink ratio

        # diff_map: union of missing and extra pixels → pixel-diff panel
        result.diff_map      = cv2.add(missing, extra)
        result.edge_diff_map = missing

        # ssim_map: float [0,1] used by heatmap renderer (1=OK, 0=defect)
        synth = np.ones(reference.shape, dtype=np.float64)
        synth[missing > 0] = 0.0   # missing ink → hot (most severe)
        synth[extra   > 0] = 0.35  # extra ink   → warm
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
    # Binarisation
    # ------------------------------------------------------------------
    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        """
        Convert preprocessed gray image to a clean binary ink mask.

        Steps:
          1. Adaptive mean threshold — each local tile uses its own mean,
             so uneven illumination across the ROI has no effect.
          2. Auto-detect polarity — dark text on light bg or vice versa.
          3. Morphological cleanup — connect broken strokes, remove noise.
          4. Small-component filter — discard isolated noise dots.
        """
        h, w = gray.shape

        # Block size must be odd and ≤ min image dimension
        block = ADAPTIVE_BLOCK_SIZE
        block = min(block, h, w)
        if block % 2 == 0:
            block -= 1
        block = max(block, 3)

        # THRESH_BINARY_INV: dark pixels (ink on bright bg) → white
        bin_mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block, ADAPTIVE_C,
        )

        # Polarity guard: text is the minority; if >50% is white, invert
        if np.count_nonzero(bin_mask) > gray.size * 0.5:
            bin_mask = cv2.bitwise_not(bin_mask)

        # Connect broken strokes and remove single-pixel noise
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  _OPEN_KERNEL)

        # Remove tiny components (sensor noise, CLAHE artefacts)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        cleaned = np.zeros_like(bin_mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= TEXT_MIN_COMPONENT_AREA:
                cleaned[labels == i] = 255

        return cleaned
