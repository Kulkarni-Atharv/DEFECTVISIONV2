from __future__ import annotations
import cv2
import numpy as np
from config import (
    HEATMAP_ALPHA,
    ROI_BORDER_THICKNESS,
    PANEL_CELL_SCALE,
    CORNER_ACCENT_LENGTH,
)

# Status colours (BGR)
_GREEN  = (30,  200,  30)
_RED    = (20,   20, 220)
_YELLOW = (0,   210, 255)
_WHITE  = (255, 255, 255)
_GRAY   = (160, 160, 160)


class Visualizer:
    """
    Renders all visual overlays.

    Two output surfaces:
      • Main window  — full camera frame with ROI border and status badge.
      • Panel window — 2×2 grid: Reference | Live | Defect-heatmap | Diff map.
    """

    # ------------------------------------------------------------------
    # Main frame overlay
    # ------------------------------------------------------------------
    def draw_main_overlay(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
        confirmed_defect: bool,
        smoothed_score: float,
        warming_up: bool = False,
        match_conf: float = 1.0,
        searching: bool = False,
    ) -> np.ndarray:
        x, y, w, h = roi
        if searching:
            color = _GRAY
        elif warming_up:
            color = _YELLOW
        elif confirmed_defect:
            color = _RED
        else:
            color = _GREEN

        # Solid border
        cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2),
                      color, ROI_BORDER_THICKNESS + 1)

        # Corner bracket accents
        cl = CORNER_ACCENT_LENGTH
        for px, py, dx, dy in [
            (x,     y,      1,  1),
            (x + w, y,     -1,  1),
            (x,     y + h,  1, -1),
            (x + w, y + h, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * cl, py), color, 3)
            cv2.line(frame, (px, py), (px, py + dy * cl), color, 3)

        # Status badge above ROI
        if searching:
            badge = "SEARCHING..."
        elif warming_up:
            badge = "WARMING UP..."
        elif confirmed_defect:
            badge = f"DEFECT  score={smoothed_score:.3f}  match={match_conf:.2f}"
        else:
            badge = f"PASS    score={smoothed_score:.3f}  match={match_conf:.2f}"

        text_y = max(y - 10, 20)
        cv2.putText(frame, badge, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # 2×2 inspection panel
    # ------------------------------------------------------------------
    def build_panel(
        self,
        roi_bgr: np.ndarray,
        ref_gray: np.ndarray,
        live_gray: np.ndarray,
        result,
        confirmed_defect: bool,
        smoothed_score: float,
        fps: float,
        warming_up: bool = False,
        match_conf: float = 1.0,
    ) -> np.ndarray:
        """
        Assemble a 2-column × 2-row grid:
          [Reference]  [Live + defect contours]
          [Heatmap  ]  [Diff map              ]
        Plus a status bar across the bottom.
        """
        h, w = roi_bgr.shape[:2]
        cw = max(int(w * PANEL_CELL_SCALE), 160)
        ch = max(int(h * PANEL_CELL_SCALE), 80)

        # ---- Cell builders ------------------------------------------
        def gray_cell(gray: np.ndarray, label: str) -> np.ndarray:
            img = cv2.cvtColor(cv2.resize(gray, (cw, ch)), cv2.COLOR_GRAY2BGR)
            self._cell_label(img, label)
            return img

        def color_cell(bgr: np.ndarray, label: str) -> np.ndarray:
            img = cv2.resize(bgr, (cw, ch))
            self._cell_label(img, label)
            return img

        # Top-left: reference
        cell_ref = gray_cell(ref_gray, "REFERENCE")

        # Top-right: live with defect contours
        live_annotated = cv2.cvtColor(cv2.resize(live_gray, (cw, ch)), cv2.COLOR_GRAY2BGR)
        if result.defect_contours and result.defect_bboxes:
            scale_x = cw / roi_bgr.shape[1]
            scale_y = ch / roi_bgr.shape[0]
            for bx, by, bw, bh in result.defect_bboxes:
                sx1, sy1 = int(bx * scale_x), int(by * scale_y)
                sx2, sy2 = int((bx + bw) * scale_x), int((by + bh) * scale_y)
                cv2.rectangle(live_annotated, (sx1, sy1), (sx2, sy2), _RED, 1)
        self._cell_label(live_annotated, "LIVE")
        cell_live = live_annotated

        # Bottom-left: SSIM defect heatmap
        if result.ssim_map is not None:
            cell_heat = color_cell(self._make_heatmap(result.ssim_map, roi_bgr), "DEFECT MAP")
        else:
            cell_heat = gray_cell(ref_gray, "DEFECT MAP")

        # Bottom-right: raw pixel diff map
        if result.diff_map is not None:
            diff_vis = cv2.normalize(result.diff_map, None, 0, 255, cv2.NORM_MINMAX)
            cell_diff = gray_cell(diff_vis.astype(np.uint8), "PIXEL DIFF")
        else:
            cell_diff = gray_cell(np.zeros((ch, cw), dtype=np.uint8), "PIXEL DIFF")

        top_row = np.hstack([cell_ref, cell_live])
        bot_row = np.hstack([cell_heat, cell_diff])
        panel   = np.vstack([top_row, bot_row])

        # ---- Status bar ---------------------------------------------
        bar_h = 44
        bar = np.zeros((bar_h, panel.shape[1], 3), dtype=np.uint8)

        if warming_up:
            bar_color = _YELLOW
            status_text = f"WARMING UP  |  FPS: {fps:.1f}  Match: {match_conf:.2f}"
        elif confirmed_defect:
            bar_color = _RED
            status_text = (
                f"DEFECT  |  Score: {smoothed_score:.3f}"
                f"  SSIM: {result.ssim_score:.3f}"
                f"  EdgeDiff: {result.edge_diff_score:.3f}"
                f"  Match: {match_conf:.2f}"
                f"  FPS: {fps:.1f}"
            )
        else:
            bar_color = _GREEN
            status_text = (
                f"PASS    |  Score: {smoothed_score:.3f}"
                f"  SSIM: {result.ssim_score:.3f}"
                f"  EdgeDiff: {result.edge_diff_score:.3f}"
                f"  Match: {match_conf:.2f}"
                f"  FPS: {fps:.1f}"
            )

        bar[:] = bar_color
        cv2.putText(bar, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)

        return np.vstack([panel, bar])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_heatmap(
        self, ssim_map: np.ndarray, roi_bgr: np.ndarray
    ) -> np.ndarray:
        """Blend a JET colormap of the defect intensity over the ROI."""
        defect_intensity = np.uint8(np.clip((1.0 - ssim_map) * 255, 0, 255))
        heatmap = cv2.applyColorMap(defect_intensity, cv2.COLORMAP_JET)
        heatmap_rs = cv2.resize(heatmap, (roi_bgr.shape[1], roi_bgr.shape[0]))
        blended = cv2.addWeighted(
            roi_bgr,    1.0 - HEATMAP_ALPHA,
            heatmap_rs, HEATMAP_ALPHA,
            0,
        )
        return blended

    @staticmethod
    def _cell_label(img: np.ndarray, text: str) -> None:
        cv2.putText(img, text, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
