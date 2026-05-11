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


# ====================================================================
# Reference capture
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
# Reference capture UI
# ====================================================================

def wait_for_reference_capture(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Show a live preview with the ROI highlighted.
    The user places a clean sample and presses SPACE to capture.
    Returns (ref_gray, ref_template) or None on abort.
    """
    WIN = "DefectVision — Reference Capture  [SPACE=capture] [Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    x, y, w, h = roi

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(display,
                    "Place CLEAN sample under camera  —  press SPACE to capture reference",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print(f"[INFO] Capturing {REFERENCE_FRAME_COUNT} reference frames …")
            ref_gray, ref_template = capture_reference(cap, roi, preprocessor)
            cv2.destroyWindow(WIN)
            return ref_gray, ref_template
        elif key == ord('q'):
            cv2.destroyWindow(WIN)
            return None


# ====================================================================
# Main inspection loop
# ====================================================================

def run_inspection(
    cap,
    roi: tuple[int, int, int, int],
    ref_gray: np.ndarray,
    preprocessor: Preprocessor,
    aligner: Aligner,
    inspector: Inspector,
    temporal: TemporalFilter,
    visualizer: Visualizer,
    logger: DefectLogger,
    position_lock: PositionLock | None = None,
) -> None:
    inspector.set_reference(ref_gray)
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

            # ---- Alignment: phase correlation corrects residual ±1-2 px
            # jitter left over after template-match integer positioning.
            live_gray = aligner.align(ref_gray, live_gray)

            # ---- Structural inspection -------------------------------
            result = inspector.inspect(ref_gray, live_gray)

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
                roi_bgr, ref_gray, live_gray,
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
            print("[INFO] Recapturing reference — place CLEAN sample, then press SPACE …")
            result = wait_for_reference_capture(cap, roi, preprocessor)
            if result is not None:
                ref_gray, ref_template = result
                inspector.set_reference(ref_gray)
                temporal.reset()
                if position_lock is not None:
                    position_lock._tpl = ref_template
                    position_lock.reset()
                print("[INFO] Reference updated.")
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

    # ---- Reference capture -----------------------------------------
    ref_result = wait_for_reference_capture(cam, roi, preprocessor)
    if ref_result is None:
        print("[INFO] Reference capture cancelled.  Exiting.")
        cam.release()
        sys.exit(0)

    ref_gray, ref_template = ref_result
    print(f"[INFO] Reference shape: {ref_gray.shape}  dtype: {ref_gray.dtype}")

    # ---- Position lock ---------------------------------------------
    position_lock: PositionLock | None = None
    if POSITION_LOCK_ENABLED:
        position_lock = PositionLock(ref_template)
        print(f"[INFO] Position lock ON — template {ref_template.shape[1]}×{ref_template.shape[0]} px")
    else:
        print("[INFO] Position lock OFF — fixed ROI mode")

    # ---- Inspection loop -------------------------------------------
    try:
        run_inspection(
            cam, roi, ref_gray,
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
