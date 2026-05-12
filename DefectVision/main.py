"""
DefectVision — Real-time print defect inspection
================================================
Entry point.  Run with:
    python main.py
    python main.py --roi 100 50 400 200   # skip GUI ROI selector

Key bindings (during inspection):
    Q      — quit
    R      — recapture reference (place clean sample first)
    S      — save snapshot of current live ROI to logs/
    SPACE  — pause / resume
"""
from __future__ import annotations
import argparse
import sys
import time
import cv2
import numpy as np

# --- project imports ------------------------------------------------
from config import (
    CAMERA_BACKEND,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    PICAMERA2_WIDTH, PICAMERA2_HEIGHT, PICAMERA2_FPS, PICAMERA2_WARMUP_S,
    REFERENCE_FRAME_COUNT, REFERENCE_WARMUP_FRAMES,
    LOG_DIR,
    POSITION_LOCK_ENABLED,
    MAX_REFERENCES, REF_MIN_DISTINCTNESS,
)
from core.camera          import create_camera
from core.roi_selector    import ROISelector
from core.preprocessor    import Preprocessor
from core.aligner         import Aligner
from core.inspector       import Inspector
from core.temporal_filter import TemporalFilter
from core.visualizer      import Visualizer
from core.position_lock   import PositionLock
from utils.logger         import DefectLogger

import os
os.makedirs(LOG_DIR, exist_ok=True)


def _grab_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y: y + h, x: x + w].copy()


def _focused_template(
    ref_gray: np.ndarray,
    ref_template: np.ndarray,
    min_fraction: float = 0.03,
    margin: int = 10,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Shrink the position-lock template to the text bounding box so that the
    NCC search matches on distinctive ink rather than featureless background.

    Returns (focused_crop, (offset_x, offset_y)) where offset is the
    top-left corner of the crop within the full ROI.  Falls back to
    (ref_template, (0, 0)) when text cannot be isolated.
    """
    try:
        polarity = Inspector._detect_polarity(ref_gray)
        bin_mask = Inspector._binarize(ref_gray, polarity)
        pts = cv2.findNonZero(bin_mask)
        if pts is None:
            return ref_template, (0, 0)
        tx, ty, tw, th = cv2.boundingRect(pts)
        if tw * th < bin_mask.size * min_fraction:
            return ref_template, (0, 0)
        h_img, w_img = ref_template.shape[:2]
        x1 = max(0, tx - margin)
        y1 = max(0, ty - margin)
        x2 = min(w_img, tx + tw + margin)
        y2 = min(h_img, ty + th + margin)
        crop = ref_template[y1:y2, x1:x2]
        if crop.size == 0:
            return ref_template, (0, 0)
        return crop, (x1, y1)
    except Exception:
        return ref_template, (0, 0)


# ====================================================================
# Multi-reference helpers
# ====================================================================

def _frame_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two same-shape grayscale images."""
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    am = af - af.mean()
    bm = bf - bf.mean()
    denom = np.sqrt(np.sum(am ** 2) * np.sum(bm ** 2))
    return float(np.clip(np.sum(am * bm) / (denom + 1e-8), 0.0, 1.0))


def _select_diverse_refs(
    proc_frames: list,
    raw_frames:  list,
    max_refs: int,
    min_distinctness: float,
) -> tuple:
    """
    Walk through proc_frames and keep up to max_refs frames whose pairwise
    NCC is below min_distinctness.  Returns (ref_grays, ref_templates).
    """
    if not proc_frames:
        return [], []

    # Subsample to at most 200 candidates so selection stays fast
    step       = max(1, len(proc_frames) // 200)
    indices    = list(range(0, len(proc_frames), step))

    sel_proc = [proc_frames[indices[0]]]
    sel_raw  = [raw_frames[indices[0]]]

    for idx in indices[1:]:
        if len(sel_proc) >= max_refs:
            break
        candidate = proc_frames[idx]
        if all(_frame_ncc(candidate, r) < min_distinctness for r in sel_proc):
            sel_proc.append(proc_frames[idx])
            sel_raw.append(raw_frames[idx])

    return sel_proc, sel_raw


# ====================================================================
# Reference capture (single averaged frame — kept for internal use)
# ====================================================================

def capture_reference(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
    n: int = REFERENCE_FRAME_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average N frames to build a stable reference.

    Returns
    -------
    (ref_gray, ref_template)
      ref_gray     — CLAHE-processed uint8, used by Inspector for SSIM comparison.
      ref_template — raw grayscale uint8, used by PositionLock for template matching.
                     Raw gray keeps the template stable across lighting variation
                     and matches against the equally raw per-frame grayscale.
    """
    acc_processed: np.ndarray | None = None
    acc_raw:       np.ndarray | None = None
    collected = 0

    while collected < n:
        ret, frame = cap.read()
        if not ret:
            continue
        crop = _grab_roi(frame, roi)
        processed = preprocessor.process(crop)
        raw_gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if acc_processed is None:
            acc_processed = np.float64(processed)
            acc_raw       = np.float64(raw_gray)
        else:
            acc_processed += processed
            acc_raw       += raw_gray
        collected += 1
        time.sleep(1.0 / CAMERA_FPS)

    return np.uint8(acc_processed / n), np.uint8(acc_raw / n)


# ====================================================================
# Reference capture UI — video-based multi-angle calibration
# ====================================================================

def capture_reference_video(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
) -> tuple[list, list] | None:
    """
    Record a short video of a clean print while the user tilts/rotates
    the object through all expected orientations.  Diverse frames are
    extracted automatically as the reference set.

    Controls
    --------
    SPACE  — start recording (box turns red) / stop recording
    Q      — confirm current references and proceed (or abort if none)

    Returns
    -------
    (ref_grays, ref_templates)
      ref_grays     — list of preprocessed uint8 grays (Inspector)
      ref_templates — list of raw grayscale uint8 (PositionLock)
    Returns None if the user quits without recording anything.
    """
    WIN = "DefectVision — Video Reference Capture  [SPACE=start/stop  Q=confirm/quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    x, y, w, h = roi

    recording:   bool       = False
    proc_frames: list       = []
    raw_frames:  list       = []
    ref_grays:   list       = []
    ref_templates: list     = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if recording:
            crop = _grab_roi(frame, roi)
            proc_frames.append(preprocessor.process(crop))
            raw_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

        display = frame.copy()
        color   = (0, 0, 220) if recording else (0, 220, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        if recording:
            label = f"RECORDING  {len(proc_frames)} frames  |  SPACE = stop"
        elif ref_grays:
            label = (f"{len(ref_grays)} angle(s) captured  |  "
                     f"Q = confirm  |  SPACE = re-record")
        else:
            label = "Tilt print through all expected angles  |  SPACE = start recording"

        cv2.putText(display, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not recording:
                recording = True
                proc_frames.clear()
                raw_frames.clear()
                print("[INFO] Recording started — tilt the print through all expected orientations …")
            else:
                recording = False
                print(f"[INFO] Stopped.  Selecting references from {len(proc_frames)} frames …")
                ref_grays, ref_templates = _select_diverse_refs(
                    proc_frames, raw_frames,
                    MAX_REFERENCES, REF_MIN_DISTINCTNESS,
                )
                print(f"[INFO] {len(ref_grays)} distinct reference angle(s) selected.")

        elif key == ord('q'):
            cv2.destroyWindow(WIN)
            return (ref_grays, ref_templates) if ref_grays else None


# ====================================================================
# Main inspection loop
# ====================================================================

def run_inspection(
    cap,
    roi: tuple[int, int, int, int],
    ref_grays: list,
    ref_templates: list,
    preprocessor: Preprocessor,
    aligner: Aligner,
    inspector: Inspector,
    temporal: TemporalFilter,
    visualizer: Visualizer,
    logger: DefectLogger,
    position_lock: PositionLock | None = None,
) -> None:
    inspector.set_reference(ref_grays[0])
    temporal.reset()
    if position_lock is not None:
        position_lock.reset()

    WIN_MAIN  = "DefectVision — Live Feed"
    WIN_PANEL = "DefectVision — Inspection Panel"
    cv2.namedWindow(WIN_MAIN,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PANEL, cv2.WINDOW_NORMAL)

    frame_num   = 0
    fps         = 0.0
    fps_t0      = time.monotonic()
    fps_counter = 0

    from core.inspector import InspectionResult
    result           = InspectionResult()
    smoothed_score   = 0.0
    confirmed_defect = False
    match_conf       = 0.0
    current_roi      = roi     # updated each frame when position lock is active
    paused           = False

    print("[INFO] Inspection running.  Q=quit  R=new reference  S=snapshot  SPACE=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_num   += 1
            fps_counter += 1

            if fps_counter >= 30:
                fps         = fps_counter / max(time.monotonic() - fps_t0, 1e-6)
                fps_t0      = time.monotonic()
                fps_counter = 0

            # ---- Position lock: find print in frame ------------------
            if position_lock is not None:
                frame_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                match = position_lock.find(frame_gray_full)

                if match is None:
                    # Print not found or blurry — show searching state, skip inspection
                    main_display = frame.copy()
                    main_display = visualizer.draw_main_overlay(
                        main_display, current_roi,
                        confirmed_defect=False, smoothed_score=0.0,
                        warming_up=False, match_conf=0.0, searching=True,
                    )
                    cv2.putText(main_display,
                                f"FPS: {fps:.1f}  Frame: {frame_num}",
                                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
                    cv2.imshow(WIN_MAIN, main_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
                    continue

                current_roi, match_conf = match
            else:
                current_roi = roi
                match_conf  = 1.0

            # ---- Extract and preprocess live ROI ---------------------
            roi_bgr   = _grab_roi(frame, current_roi)
            live_gray = preprocessor.process(roi_bgr)

            # ---- Align live to each reference, keep best match -------
            # Each reference may be at a different angle; aligning to the
            # closest one minimises structural residuals before comparison.
            best_result = None
            best_ref    = ref_grays[0]
            best_live   = live_gray
            for ref in ref_grays:
                aligned = aligner.align(ref, live_gray)
                res     = inspector.inspect(ref, aligned)
                if best_result is None or res.defect_score < best_result.defect_score:
                    best_result = res
                    best_ref    = ref
                    best_live   = aligned
            result    = best_result
            live_gray = best_live

            # ---- Temporal consistency --------------------------------
            warming_up = not temporal.window_full
            smoothed_score, confirmed_defect = temporal.update(
                result.defect_score, result.is_defect
            )
            if warming_up:
                confirmed_defect = False

            # ---- Log -------------------------------------------------
            logger.log(frame_num, result, confirmed_defect, smoothed_score, roi_bgr)

            # ---- Main feed display -----------------------------------
            main_display = frame.copy()
            main_display = visualizer.draw_main_overlay(
                main_display, current_roi, confirmed_defect, smoothed_score,
                warming_up, match_conf,
            )
            cv2.putText(main_display,
                        f"FPS: {fps:.1f}  Frame: {frame_num}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.imshow(WIN_MAIN, main_display)

            # ---- Inspection panel ------------------------------------
            panel = visualizer.build_panel(
                roi_bgr, best_ref, live_gray,
                result, confirmed_defect, smoothed_score, fps, warming_up, match_conf,
            )
            cv2.imshow(WIN_PANEL, panel)

        else:
            cv2.waitKey(50)

        # ---- Key handling --------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

        elif key == ord('r'):
            print("[INFO] Recapturing references — record video of clean sample …")
            cap_result = capture_reference_video(cap, roi, preprocessor)
            if cap_result is not None:
                ref_grays, ref_templates = cap_result
                inspector.set_reference(ref_grays[0])
                temporal.reset()
                if position_lock is not None:
                    focused_tpl, tpl_offset = _focused_template(
                        ref_grays[0], ref_templates[0]
                    )
                    position_lock.update_template(
                        focused_tpl,
                        roi_offset    = tpl_offset,
                        full_roi_size = (roi[2], roi[3]),
                    )
                print(f"[INFO] Reference updated: {len(ref_grays)} angle(s).")
            else:
                print("[INFO] Reference recapture cancelled.")

        elif key == ord('s'):
            snap_path = os.path.join(
                LOG_DIR,
                f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(snap_path, roi_bgr)
            print(f"[INFO] Snapshot saved: {snap_path}")

    cv2.destroyAllWindows()


# ====================================================================
# Entry point
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DefectVision print defect inspection")
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Skip GUI ROI selector and use fixed coordinates  e.g. --roi 100 50 400 200"
    )
    args = parser.parse_args()

    # ---- Camera ----------------------------------------------------
    print(f"[INFO] Starting camera (backend={CAMERA_BACKEND}) …")
    cam = create_camera(
        CAMERA_BACKEND,
        index    = CAMERA_INDEX,
        width    = PICAMERA2_WIDTH  if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_WIDTH,
        height   = PICAMERA2_HEIGHT if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_HEIGHT,
        fps      = PICAMERA2_FPS    if CAMERA_BACKEND.upper() == "PICAMERA2" else CAMERA_FPS,
        warmup_s = PICAMERA2_WARMUP_S,
        warmup_frames = REFERENCE_WARMUP_FRAMES,
    )
    if not cam.is_opened():
        print("[ERROR] Could not open camera.  Check CAMERA_BACKEND / CAMERA_INDEX in config.py.")
        sys.exit(1)

    w, h = cam.get_resolution()
    print(f"[INFO] Camera ready: {w}×{h} @ {cam.get_fps():.0f} fps")

    # ---- ROI selection ---------------------------------------------
    if args.roi:
        roi = tuple(args.roi)
        print(f"[INFO] ROI from CLI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    else:
        print("[INFO] Select the print ROI on the live feed.")
        roi = ROISelector().select(cam)
        if roi is None:
            print("[INFO] ROI selection cancelled.  Exiting.")
            cam.release()
            sys.exit(0)

    x, y, w, h = roi
    print(f"[INFO] ROI: x={x} y={y} w={w} h={h}")

    # ---- Subsystem init --------------------------------------------
    preprocessor = Preprocessor()
    aligner      = Aligner()
    inspector    = Inspector()
    temporal     = TemporalFilter()
    visualizer   = Visualizer()
    logger       = DefectLogger()

    # ---- Reference capture (video — multi-angle) -------------------
    ref_result = capture_reference_video(cam, roi, preprocessor)
    if ref_result is None:
        print("[INFO] Reference capture cancelled.  Exiting.")
        cam.release()
        sys.exit(0)

    ref_grays, ref_templates = ref_result
    print(f"[INFO] {len(ref_grays)} reference angle(s) captured. "
          f"Shape: {ref_grays[0].shape}  dtype: {ref_grays[0].dtype}")

    # ---- Position lock ---------------------------------------------
    position_lock: PositionLock | None = None
    if POSITION_LOCK_ENABLED:
        inspector.set_reference(ref_grays[0])  # needed so _focused_template can binarise
        focused_tpl, tpl_offset = _focused_template(ref_grays[0], ref_templates[0])
        position_lock = PositionLock(
            focused_tpl,
            roi_offset    = tpl_offset,
            full_roi_size = (roi[2], roi[3]),
        )
        print(
            f"[INFO] Position lock ON — "
            f"focused template {focused_tpl.shape[1]}×{focused_tpl.shape[0]} px"
            f"  offset={tpl_offset}"
        )
    else:
        print("[INFO] Position lock OFF — fixed ROI mode")

    # ---- Inspection loop -------------------------------------------
    try:
        run_inspection(
            cam, roi, ref_grays, ref_templates,
            preprocessor, aligner, inspector, temporal, visualizer, logger,
            position_lock,
        )
    finally:
        cam.release()
        summary = logger.summary()
        print("\n[SESSION SUMMARY]")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
