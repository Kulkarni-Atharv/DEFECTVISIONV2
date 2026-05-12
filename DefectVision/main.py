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
    INSPECT_EARLY_EXIT_SCORE,
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


def _select_diverse_bgr_frames(
    bgr_frames: list,
    max_refs: int,
    min_distinctness: float,
) -> list:
    """
    Walk through bgr_frames and keep up to max_refs frames whose pairwise
    NCC (computed at 0.25× scale for speed) is below min_distinctness.
    Returns a list of BGR frames.
    """
    if not bgr_frames:
        return []

    scale = 0.25
    step  = max(1, len(bgr_frames) // 200)

    def small_gray(i: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.resize(bgr_frames[i], None, fx=scale, fy=scale),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float64)

    indices   = list(range(0, len(bgr_frames), step))
    sel_small = [small_gray(indices[0])]
    sel_idx   = [indices[0]]

    for idx in indices[1:]:
        if len(sel_idx) >= max_refs:
            break
        s = small_gray(idx)
        if all(_frame_ncc(s, r) < min_distinctness for r in sel_small):
            sel_small.append(s)
            sel_idx.append(idx)

    return [bgr_frames[i] for i in sel_idx]


def _auto_text_crop(
    frame_bgr: np.ndarray,
    preprocessor: Preprocessor,
    target_hw: tuple[int, int] | None = None,
    margin: int = 25,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detect the text bounding box in *frame_bgr* using the Inspector's
    binarisation and return (processed_gray_crop, raw_gray_crop), both
    resized to *target_hw* (h, w) if given.

    Returns (None, None) if no text is found.
    """
    try:
        gray = preprocessor.process(frame_bgr)
        polarity = Inspector._detect_polarity(gray)
        bin_mask = Inspector._binarize(gray, polarity)
        pts = cv2.findNonZero(bin_mask)
        if pts is None:
            return None, None
        tx, ty, tw, th = cv2.boundingRect(pts)
        h_img, w_img = gray.shape
        x1 = max(0, tx - margin)
        y1 = max(0, ty - margin)
        x2 = min(w_img, tx + tw + margin)
        y2 = min(h_img, ty + th + margin)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop_gray = gray[y1:y2, x1:x2]
        raw_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
        if target_hw is not None:
            th_t, tw_t = target_hw
            crop_gray = cv2.resize(crop_gray, (tw_t, th_t))
            raw_gray  = cv2.resize(raw_gray,  (tw_t, th_t))
        return crop_gray, raw_gray
    except Exception:
        return None, None


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
      ref_gray     — CLAHE-processed uint8, used by Inspector for comparison.
      ref_template — raw grayscale uint8, used by PositionLock for template matching.
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
# Reference capture UI — full-frame video-based multi-angle calibration
# ====================================================================

def capture_reference_video(
    cap,
    preprocessor: Preprocessor,
) -> tuple[list, list] | None:
    """
    Record a short video of a clean print at different angles.
    Recording uses the FULL FRAME — no ROI constraint — so the user can
    tilt and rotate freely without the print leaving the crop area.

    After stopping, diverse frames are extracted automatically and the
    text bounding box is detected from each frame.  All crops are
    normalised to the same size so Inspector and PositionLock can use
    them interchangeably.

    Controls
    --------
    SPACE  — start recording (overlay turns red) / stop recording
    Q      — confirm current references and proceed (or abort if none)

    Returns
    -------
    (ref_grays, ref_templates)
      ref_grays     — list of preprocessed uint8 gray text crops (Inspector)
      ref_templates — list of raw grayscale uint8 text crops (PositionLock)
    Returns None if the user quits without capturing any references.
    """
    WIN = "DefectVision — Reference Video  [SPACE=start/stop  Q=confirm]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    recording:      bool  = False
    raw_bgr_frames: list  = []
    ref_grays:      list  = []
    ref_templates:  list  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if recording:
            raw_bgr_frames.append(frame.copy())

        display = frame.copy()
        color   = (0, 0, 220) if recording else (0, 220, 255)

        if recording:
            label = f"RECORDING  {len(raw_bgr_frames)} frames  |  SPACE = stop"
        elif ref_grays:
            label = (f"{len(ref_grays)} angle(s) ready  |  "
                     f"Q = confirm  |  SPACE = re-record")
        else:
            label = "Tilt print through all expected angles  |  SPACE = start"

        cv2.putText(display, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                recording = True
                raw_bgr_frames.clear()
                print("[INFO] Recording started — tilt the print through all expected orientations …")
            else:
                recording = False
                print(f"[INFO] Stopped.  Selecting references from {len(raw_bgr_frames)} frames …")

                diverse_bgr = _select_diverse_bgr_frames(
                    raw_bgr_frames, MAX_REFERENCES, REF_MIN_DISTINCTNESS,
                )
                # Release the large recording buffer immediately — 1000 frames
                # at full resolution can exceed 4 GB; keeping them in memory
                # while inspection runs causes lag from memory pressure.
                raw_bgr_frames.clear()

                ref_grays.clear()
                ref_templates.clear()
                target_hw: tuple[int, int] | None = None

                for bgr in diverse_bgr:
                    cg, cr = _auto_text_crop(bgr, preprocessor, target_hw)
                    if cg is not None:
                        if target_hw is None:
                            target_hw = (cg.shape[0], cg.shape[1])
                        ref_grays.append(cg)
                        ref_templates.append(cr)

                diverse_bgr.clear()  # done with the 7 full-frame BGR crops too

                if ref_grays:
                    print(
                        f"[INFO] {len(ref_grays)} distinct reference angle(s) selected.  "
                        f"Crop size: {ref_grays[0].shape[1]}×{ref_grays[0].shape[0]} px"
                    )
                else:
                    print(
                        "[WARN] No text detected in recorded frames.  "
                        "Ensure the print is clearly visible and try again."
                    )

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

    # Pre-compute search bounds (user's drawn ROI in frame coordinates)
    sx, sy, sw, sh = roi

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

            # ---- Position lock: find print within the drawn ROI ------
            if position_lock is not None:
                frame_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Constrain the search to the user's drawn ROI so that
                # unrelated objects elsewhere in the frame are ignored.
                search_region = frame_gray_full[sy:sy + sh, sx:sx + sw]
                match = position_lock.find(search_region)

                if match is None:
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

                # Convert ROI-relative coords to full-frame coords.
                # best_tpl_idx is the template that won — put that reference
                # first so the inspection loop checks the closest angle first.
                (tx, ty, tw, th), match_conf, best_tpl_idx = match
                current_roi = (sx + tx, sy + ty, tw, th)
            else:
                current_roi  = roi
                match_conf   = 1.0
                best_tpl_idx = 0

            # ---- Extract and preprocess live ROI ---------------------
            roi_bgr   = _grab_roi(frame, current_roi)
            live_gray = preprocessor.process(roi_bgr)

            # Resize live to reference size if needed (e.g. no position lock)
            if live_gray.shape != ref_grays[0].shape:
                live_gray = cv2.resize(
                    live_gray,
                    (ref_grays[0].shape[1], ref_grays[0].shape[0]),
                )

            # ---- Align live to each reference, keep best match -------
            # Check the angle that position lock already identified as the
            # closest match first.  If it scores clean, skip the rest —
            # this cuts 5-6 ECC calls per frame on the common (clean) path.
            if position_lock is not None:
                ordered_indices = [best_tpl_idx] + [
                    i for i in range(len(ref_grays)) if i != best_tpl_idx
                ]
            else:
                ordered_indices = list(range(len(ref_grays)))

            best_result = None
            best_ref    = ref_grays[0]
            best_live   = live_gray
            for i in ordered_indices:
                ref     = ref_grays[i]
                aligned = aligner.align(ref, live_gray)
                res     = inspector.inspect(ref, aligned)
                if best_result is None or res.defect_score < best_result.defect_score:
                    best_result = res
                    best_ref    = ref
                    best_live   = aligned
                if best_result.defect_score < INSPECT_EARLY_EXIT_SCORE:
                    break  # clearly clean — no need to check remaining angles
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
            cap_result = capture_reference_video(cap, preprocessor)
            if cap_result is not None:
                ref_grays, ref_templates = cap_result
                inspector.set_reference(ref_grays[0])
                temporal.reset()
                if position_lock is not None:
                    position_lock.update_template(ref_templates)
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
        help="Skip GUI ROI selector; use fixed search area  e.g. --roi 100 50 400 200"
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

    # ---- Subsystem init --------------------------------------------
    preprocessor = Preprocessor()
    aligner      = Aligner()
    inspector    = Inspector()
    temporal     = TemporalFilter()
    visualizer   = Visualizer()
    logger       = DefectLogger()

    # ---- Step 1: Reference video — FULL FRAME, no ROI constraint ---
    # User tilts/rotates the clean print freely; diverse angle frames
    # are captured and auto-cropped to the text region.
    print("[INFO] Step 1: Record reference video (tilt the print through all expected angles).")
    ref_result = capture_reference_video(cam, preprocessor)
    if ref_result is None:
        print("[INFO] Reference capture cancelled.  Exiting.")
        cam.release()
        sys.exit(0)

    ref_grays, ref_templates = ref_result
    print(
        f"[INFO] {len(ref_grays)} reference angle(s) captured.  "
        f"Shape: {ref_grays[0].shape}  dtype: {ref_grays[0].dtype}"
    )

    # ---- Step 2: Draw search ROI -----------------------------------
    # The ROI is the region of the frame where the print can appear.
    # Position lock will track the text anywhere within this area.
    print("[INFO] Step 2: Draw a generous ROI covering the area where the print can appear.")
    if args.roi:
        roi = tuple(args.roi)
        print(f"[INFO] ROI from CLI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    else:
        roi = ROISelector().select(cam)
        if roi is None:
            print("[INFO] ROI selection cancelled.  Exiting.")
            cam.release()
            sys.exit(0)

    x, y, rw, rh = roi
    print(f"[INFO] Search ROI: x={x} y={y} w={rw} h={rh}")

    # ---- Step 3: Position lock with all reference templates --------
    position_lock: PositionLock | None = None
    if POSITION_LOCK_ENABLED:
        position_lock = PositionLock(
            ref_templates,      # list of raw-gray text crops, one per angle
            roi_offset=(0, 0),  # templates are already tight text crops
            full_roi_size=None, # return template-size bounding boxes
        )
        print(
            f"[INFO] Position lock ON — {len(ref_templates)} template(s)  "
            f"size: {ref_templates[0].shape[1]}×{ref_templates[0].shape[0]} px"
        )
    else:
        print("[INFO] Position lock OFF — fixed ROI mode")

    # ---- Step 4: Inspection loop -----------------------------------
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
