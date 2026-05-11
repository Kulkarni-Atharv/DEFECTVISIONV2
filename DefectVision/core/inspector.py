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
# Dilation kernel applied to the text mask — absorbs small positional shifts
# and movement halos at stroke boundaries without masking real defects.
_TEXT_MASK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


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

    All three signals are evaluated *within the text mask only*, so
    background lighting variation and movement halos outside the ink
    region do not contribute to the defect score.
    """

    def __init__(self) -> None:
        self._ref_edges: np.ndarray | None = None
        self._text_mask: np.ndarray | None = None  # 255 = text/ink region

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------
    def set_reference(self, ref_gray: np.ndarray) -> None:
        """Cache reference edges and text mask so they are not recomputed every frame."""
        self._ref_edges = self._edges(ref_gray)
        self._text_mask = self._compute_text_mask(ref_gray)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def inspect(self, reference: np.ndarray, live: np.ndarray) -> InspectionResult:
        result = InspectionResult()

        if reference.shape != live.shape:
            live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

        # Effective area for ratio normalisation: text pixels, or full frame if no mask.
        mask_valid = (
            self._text_mask is not None
            and np.count_nonzero(self._text_mask) > 0
        )
        mask_area = int(np.count_nonzero(self._text_mask)) if mask_valid else reference.size

        # ---- 1. SSIM -------------------------------------------------
        # win_size must be odd and <= min image dimension
        win = min(SSIM_WIN_SIZE, reference.shape[0], reference.shape[1])
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)

        _, ssim_map = ssim_fn(
            reference, live,
            full=True,
            data_range=255,
            win_size=win,
            gaussian_weights=True,
        )
        result.ssim_map = ssim_map

        # Average SSIM only within the text region — background lighting
        # shifts no longer drag the score down.
        if mask_valid:
            result.ssim_score = float(ssim_map[self._text_mask > 0].mean())
        else:
            result.ssim_score = float(ssim_map.mean())

        # ---- 2. Pixel difference (text region only) -----------------
        diff = cv2.absdiff(reference, live)
        result.diff_map = diff
        defective_pixels = diff > PIXEL_DIFF_THRESHOLD
        if mask_valid:
            defective_pixels = defective_pixels & (self._text_mask > 0)
        result.pixel_diff_score = float(np.count_nonzero(defective_pixels) / mask_area)

        # ---- 3. Edge difference (text region only) ------------------
        live_edges = self._edges(live)
        if self._ref_edges is None:
            self._ref_edges = self._edges(reference)
        edge_diff = cv2.bitwise_xor(self._ref_edges, live_edges)
        result.edge_diff_map = edge_diff
        if mask_valid:
            edge_in_mask = cv2.bitwise_and(edge_diff, self._text_mask)
            result.edge_diff_score = float(np.count_nonzero(edge_in_mask) / mask_area)
        else:
            result.edge_diff_score = float(np.count_nonzero(edge_diff) / edge_diff.size)

        # ---- 4. Composite defect score ------------------------------
        ssim_component  = float(np.clip(1.0 - result.ssim_score, 0.0, 1.0))
        edge_component  = float(np.clip(result.edge_diff_score  * EDGE_SCORE_SCALE,  0.0, 1.0))
        pixel_component = float(np.clip(result.pixel_diff_score * PIXEL_SCORE_SCALE, 0.0, 1.0))

        defect_score = (
            SSIM_WEIGHT   * ssim_component +
            EDGE_WEIGHT   * edge_component +
            PIXEL_WEIGHT  * pixel_component
        )
        result.defect_score = float(np.clip(defect_score, 0.0, 1.0))
        result.is_defect = result.defect_score >= DEFECT_SCORE_THRESHOLD

        # ---- Hard pixel-change override (within text region) --------
        # Catches thin debris, strings, and fine additions that affect only
        # a small area — their SSIM impact is too low to reach the composite
        # threshold, but the raw pixel count is unambiguous.
        if (CHANGED_PIXEL_RATIO_THRESHOLD > 0
                and result.pixel_diff_score >= CHANGED_PIXEL_RATIO_THRESHOLD):
            result.is_defect = True
            result.defect_score = max(result.defect_score, DEFECT_SCORE_THRESHOLD)

        # ---- 5. Defect localisation (constrained to text region) ----
        defect_heat = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        _, defect_mask = cv2.threshold(defect_heat, 50, 255, cv2.THRESH_BINARY)
        if mask_valid:
            defect_mask = cv2.bitwise_and(defect_mask, self._text_mask)
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

    @staticmethod
    def _compute_text_mask(gray: np.ndarray) -> np.ndarray:
        """
        Binary mask isolating ink/text pixels from the background.

        Uses Otsu thresholding (lighting-independent) to separate text from
        background, then dilates by ~11 px to absorb small positional shifts
        and stroke-edge halos without masking real defects.

        Handles both dark-on-light and light-on-dark text automatically:
        whichever binary class occupies the minority of pixels is treated
        as the text region.
        """
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # If more than half the pixels are marked as "text", the polarity is
        # wrong (light text on dark bg) — flip it.
        if np.count_nonzero(mask) > gray.size * 0.5:
            mask = cv2.bitwise_not(mask)
        return cv2.dilate(mask, _TEXT_MASK_KERNEL, iterations=2)
