from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from config import (
    SSIM_THRESHOLD,
    SSIM_WIN_SIZE,
    EDGE_DIFF_THRESHOLD,
    PIXEL_DIFF_THRESHOLD,
    SSIM_WEIGHT,
    EDGE_WEIGHT,
    PIXEL_WEIGHT,
    EDGE_SCORE_SCALE,
    PIXEL_SCORE_SCALE,
    DEFECT_SCORE_THRESHOLD,
    CHANGED_PIXEL_RATIO_THRESHOLD,
)


@dataclass
class InspectionResult:
    # Scalar metrics (0 = perfect, higher = worse for diff/edge scores)
    ssim_score: float = 1.0
    edge_diff_score: float = 0.0
    pixel_diff_score: float = 0.0
    defect_score: float = 0.0       # Weighted composite: 0 = OK, 1 = severe
    is_defect: bool = False

    # Per-pixel maps (same spatial size as the ROI)
    ssim_map: np.ndarray | None = None       # float64, range [−1, 1]
    diff_map: np.ndarray | None = None       # uint8, absolute intensity diff
    edge_diff_map: np.ndarray | None = None  # uint8, binary edge disagreement

    # Localisation
    defect_contours: list = field(default_factory=list)
    defect_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# Canny thresholds tuned for small industrial text; adjust in config if needed.
_CANNY_LOW = 40
_CANNY_HIGH = 120
# Minimum contour area (px²) to report as a discrete defect region.
_MIN_CONTOUR_AREA = 15
# Morphology kernel for cleaning the binary defect mask before contouring.
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Inspector:
    """
    Core inspection engine.

    Three complementary signals are combined into a single defect score:

      1. SSIM  — captures structural / luminance / contrast changes.
                 Best at detecting blurry ink, missing characters, and
                 debris that alters the overall texture.

      2. Edge diff — XOR of Canny edge maps.
                 Catches broken or missing strokes even when overall
                 brightness is unchanged (transparent/micro debris).

      3. Pixel diff — raw absolute intensity difference after a mild
                 threshold.  Fast sanity check; down-weighted in the
                 composite to avoid noise sensitivity.

    Weights and thresholds are fully configurable via config.py.
    """

    def __init__(self) -> None:
        self._ref_edges: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        """Cache reference edges once so they are not recomputed every frame."""
        self._ref_edges = self._edges(ref_gray)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        # ---- 1. SSIM -------------------------------------------------
        # win_size must be odd and <= min image dimension
        win = min(SSIM_WIN_SIZE, reference.shape[0], reference.shape[1])
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)

        ssim_score, ssim_map = ssim_fn(
            reference, live,
            full=True,
            data_range=255,
            win_size=win,
            gaussian_weights=True,
        )
        result.ssim_score = float(ssim_score)
        result.ssim_map = ssim_map  # float64 in [−1, 1]; 1 = identical

        # ---- 2. Pixel difference ------------------------------------
        diff = cv2.absdiff(reference, live)
        result.diff_map = diff
        defective_pixels = diff > PIXEL_DIFF_THRESHOLD
        result.pixel_diff_score = float(np.count_nonzero(defective_pixels) / diff.size)

        # ---- 3. Edge difference -------------------------------------
        live_edges = self._edges(live)
        if self._ref_edges is None:
            self._ref_edges = self._edges(reference)
        edge_diff = cv2.bitwise_xor(self._ref_edges, live_edges)
        result.edge_diff_map = edge_diff
        result.edge_diff_score = float(np.count_nonzero(edge_diff) / edge_diff.size)

        # ---- 4. Composite defect score ------------------------------
        ssim_component  = float(np.clip(1.0 - ssim_score, 0.0, 1.0))
        edge_component  = float(np.clip(result.edge_diff_score  * EDGE_SCORE_SCALE,  0.0, 1.0))
        pixel_component = float(np.clip(result.pixel_diff_score * PIXEL_SCORE_SCALE, 0.0, 1.0))

        defect_score = (
            SSIM_WEIGHT   * ssim_component +
            EDGE_WEIGHT   * edge_component +
            PIXEL_WEIGHT  * pixel_component
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Hard pixel-change override -----------------------------
        # Catches thin debris, strings, and fine additions that affect only
        # a small area — their SSIM impact is too low to reach the composite
        # threshold, but the raw pixel count is unambiguous.
        if (CHANGED_PIXEL_RATIO_THRESHOLD > 0
                and result.pixel_diff_score >= CHANGED_PIXEL_RATIO_THRESHOLD):
            result.is_defect = True
            result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 5. Defect localisation ---------------------------------
        # Build a binary mask from the (inverted) SSIM map.
        # Low SSIM → high defect intensity.
        defect_heat = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        _, defect_mask = cv2.threshold(defect_heat, 50, 255, cv2.THRESH_BINARY)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)

        contours, _ = cv2.findContours(
            defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.defect_contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
        result.defect_bboxes = [cv2.boundingRect(c) for c in result.defect_contours]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _edges(img: np.ndarray) -> np.ndarray:
        return cv2.Canny(img, _CANNY_LOW, _CANNY_HIGH)
