"""
DefectVision — Real-time print defect inspection
================================================
Entry point.  Run with:
    python main.py
    python main.py --roi 100 50 400 200   # skip GUI ROI selector

Key bindings during reference capture:
    SPACE  — capture this angle (averages several frames)
    D      — delete last captured angle
    Q      — confirm and start inspection

Key bindings during inspection:
    Q      — quit
    R      — recapture references (place clean sample first)
    S      — save snapshot of current live ROI to logs/
    SPACE  — pause / resume
"""
from __future__ import annotations
import argparse
import sys
import threading
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
    INSPECT_EARLY_EXIT_SCORE,
    POSITION_LOCK_SINGLE_REF_CONF,
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


class _FrameGrabber:
    """Background thread that continuously drains the camera buffer so the
    detection loop always reads the latest frame without being blocked by it."""

    def __init__(self, cap) -> None:
        self._cap   = cap
        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)


def _quick_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap downscaled NCC to rank reference angles before running ECC.
    ~0.3 ms per pair at 1/4 scale — used as a pre-filter only, not for scoring.
    """
    scale = 4
    h, w = a.shape[:2]
    a_s = cv2.resize(a, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    a_f = a_s.astype(np.float32).ravel()
    b_f = b_s.astype(np.float32).ravel()
    a_m = a_f - a_f.mean()
    b_m = b_f - b_f.mean()
    denom = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))
    if denom < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a_m, b_m) / denom, 0.0, 1.0))


def _focused_template(
    ref_gray: np.ndarray,
    ref_template: np.ndarray,
    min_fraction: float = 0.03,
    margin: int = 10,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Shrink the position-lock template to the text bounding box so the
    NCC search matches on distinctive ink rather than featureless background.

    Returns (focused_crop, (offset_x, offset_y)) where offset is the
    top-left of the crop within the ROI.  Falls back to the full template
    when text cannot be isolated.
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


def _save_reference_images(
    ref_grays: list,
    ref_templates: list,
) -> None:
    ts = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(LOG_DIR, f"references_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    for i, (gray, tpl) in enumerate(zip(ref_grays, ref_templates)):
        cv2.imwrite(os.path.join(save_dir, f"angle_{i+1:02d}_preprocessed.png"), gray)
        cv2.imwrite(os.path.join(save_dir, f"angle_{i+1:02d}_raw.png"), tpl)
    print(f"[INFO] Reference images saved to: {save_dir}")


# ====================================================================
# Reference capture — manual multi-angle
# ====================================================================

def capture_reference_multi(
    cap,
    roi: tuple[int, int, int, int],
    preprocessor: Preprocessor,
    n: int = REFERENCE_FRAME_COUNT,
) -> tuple[list, list] | None:
    """
    Manually capture reference images at multiple angles.

    Position the clean print at each desired orientation, then press
    SPACE to capture that angle (averages n frames for stability).
    Repeat for every orientation expected during inspection, then
    press Q to confirm.

    Controls
    --------
    SPACE  — capture current angle  (shows live ROI crop while waiting)
    D      — delete the last captured angle
    Q      — confirm and proceed (returns None if nothing was captured)

    Returns
    -------
    (ref_grays, ref_templates)
      ref_grays     — list of preprocessed uint8 ROI crops  (Inspector)
      ref_templates — list of raw grayscale uint8 ROI crops (PositionLock)
    """
    WIN = "DefectVision — Reference Capture  [SPACE=capture  D=undo  Q=confirm]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    x, y, w, h = roi

    ref_grays:    list = []
    ref_templates: list = []
    capturing          = False
    buf_proc:     list = []
    buf_raw:      list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        crop     = _grab_roi(frame, roi)
        gray_raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if capturing:
            buf_proc.append(preprocessor.process(crop))
            buf_raw.append(gray_raw)
            if len(buf_proc) >= n:
                ref_grays.append(np.uint8(np.mean(buf_proc, axis=0)))
                ref_templates.append(np.uint8(np.mean(buf_raw, axis=0)))
                buf_proc.clear()
                buf_raw.clear()
                capturing = False
                print(f"[INFO] Angle {len(ref_grays)} captured.")

        display = frame.copy()
        color   = (0, 0, 220) if capturing else (0, 220, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        if capturing:
            label = f"Capturing {len(buf_proc)}/{n} ..."
        elif ref_grays:
            label = (f"{len(ref_grays)} angle(s) ready  |  "
                     f"SPACE=next angle  D=undo  Q=confirm")
        else:
            label = "Position print at angle 1, then press SPACE"

        cv2.putText(display, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and not capturing:
            capturing = True
            buf_proc.clear()
            buf_raw.clear()
            print(f"[INFO] Capturing angle {len(ref_grays) + 1} ({n} frames) ...")

        elif key == ord('d') and ref_grays and not capturing:
            ref_grays.pop()
            ref_templates.pop()
            print(f"[INFO] Last angle removed.  {len(ref_grays)} angle(s) remaining.")

        elif key == ord('q') and not capturing:
            cv2.destroyWindow(WIN)
            if ref_grays:
                _save_reference_images(ref_grays, ref_templates)
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
    current_roi      = roi
    paused           = False
    best_tpl_idx     = 0

    grabber = _FrameGrabber(cap)
    print("[INFO] Inspection running.  Q=quit  R=new reference  S=snapshot  SPACE=pause")

    while True:
        if not paused:
            ret, frame = grabber.read()
            if not ret:
                continue

            frame_num   += 1
            fps_counter += 1
            if fps_counter >= 30:
                fps         = fps_counter / max(time.monotonic() - fps_t0, 1e-6)
                fps_t0      = time.monotonic()
                fps_counter = 0

            # ---- Position lock: search the full frame ----------------
            # The focused template is small (text only); searching the full
            # frame is fast and the roi_offset math inside find() is correct
            # only when the full frame is passed.
            if position_lock is not None:
                frame_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                match = position_lock.find(frame_gray_full)

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

                # find() already applied roi_offset → current_roi is in frame coords
                current_roi, match_conf, best_tpl_idx = match
            else:
                current_roi  = roi
                match_conf   = 0.0   # no position lock — trigger multi-ref path below
                best_tpl_idx = 0

            # ---- Extract and preprocess live ROI ---------------------
            roi_bgr   = _grab_roi(frame, current_roi)
            live_gray = preprocessor.process(roi_bgr)

            if live_gray.shape != ref_grays[0].shape:
                live_gray = cv2.resize(
                    live_gray,
                    (ref_grays[0].shape[1], ref_grays[0].shape[0]),
                )

            # ---- Align + inspect against references ------------------
            # Fast path: position lock identified the angle confidently —
            # one ECC call, done.
            # Optimised path: cheap downscaled NCC ranks all refs in <1 ms,
            # then ECC + inspect runs only on the best candidate first.
            # Early-exit at INSPECT_EARLY_EXIT_SCORE means clean prints
            # almost always cost exactly one ECC call regardless of N angles.
            if match_conf >= POSITION_LOCK_SINGLE_REF_CONF:
                check_indices = [min(best_tpl_idx, len(ref_grays) - 1)]
            elif len(ref_grays) == 1:
                check_indices = [0]
            else:
                ncc_scores    = [_quick_ncc(live_gray, r) for r in ref_grays]
                check_indices = sorted(range(len(ref_grays)), key=lambda i: -ncc_scores[i])

            best_result = None
            best_ref    = ref_grays[0]
            best_live   = live_gray
            for i in check_indices:
                ref     = ref_grays[i]
                inspector.set_reference(ref)   # must update per-ref; shape check alone is not enough
                aligned = aligner.align(ref, live_gray)
                res     = inspector.inspect(ref, aligned)
                if best_result is None or res.defect_score < best_result.defect_score:
                    best_result = res
                    best_ref    = ref
                    best_live   = aligned
                if best_result.defect_score < INSPECT_EARLY_EXIT_SCORE:
                    break
            result    = best_result
            live_gray = best_live

            # ---- Temporal consistency --------------------------------
            warming_up = not temporal.window_full
            smoothed_score, confirmed_defect = temporal.update(
                result.defect_score, result.is_defect
            )
            if warming_up:
                confirmed_defect = False

            logger.log(frame_num, result, confirmed_defect, smoothed_score, roi_bgr)

            # ---- Main feed -------------------------------------------
            main_display = frame.copy()
            main_display = visualizer.draw_main_overlay(
                main_display, current_roi, confirmed_defect, smoothed_score,
                warming_up, match_conf,
            )
            cv2.putText(main_display,
                        f"FPS: {fps:.1f}  Frame: {frame_num}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.imshow(WIN_MAIN, main_display)

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
            print("[INFO] Recapturing references — position clean sample in ROI ...")
            grabber.stop()
            cap_result = capture_reference_multi(cap, roi, preprocessor)
            grabber = _FrameGrabber(cap)
            if cap_result is not None:
                ref_grays, ref_templates = cap_result
                inspector.set_reference(ref_grays[0])
                temporal.reset()
                if position_lock is not None:
                    focused_tpls = []
                    tpl_offsets  = []
                    for rg, rt in zip(ref_grays, ref_templates):
                        tpl, off = _focused_template(rg, rt)
                        focused_tpls.append(tpl)
                        tpl_offsets.append(off)
                    position_lock.update_template(
                        focused_tpls,
                        roi_offsets   = tpl_offsets,
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

    grabber.stop()
    cv2.destroyAllWindows()


# ====================================================================
# Entry point
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DefectVision print defect inspection")
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Skip GUI ROI selector  e.g. --roi 100 50 400 200"
    )
    args = parser.parse_args()

    # ---- Camera --------------------------------------------------------
    print(f"[INFO] Starting camera (backend={CAMERA_BACKEND}) ...")
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

    cw, ch = cam.get_resolution()
    print(f"[INFO] Camera ready: {cw}x{ch} @ {cam.get_fps():.0f} fps")

    preprocessor = Preprocessor()
    aligner      = Aligner()
    inspector    = Inspector()
    temporal     = TemporalFilter()
    visualizer   = Visualizer()
    logger       = DefectLogger()

    # ---- Step 1: Draw the inspection ROI --------------------------------
    if args.roi:
        roi = tuple(args.roi)
        print(f"[INFO] ROI from CLI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    else:
        print("[INFO] Step 1: Draw the inspection ROI on the live feed.")
        roi = ROISelector().select(cam)
        if roi is None:
            print("[INFO] ROI selection cancelled.  Exiting.")
            cam.release()
            sys.exit(0)

    x, y, rw, rh = roi
    print(f"[INFO] ROI: x={x} y={y} w={rw} h={rh}")

    # ---- Step 2: Capture references at each expected angle --------------
    print("[INFO] Step 2: Capture reference at each expected angle.")
    print("[INFO]   Position clean print in ROI → SPACE to capture → repeat → Q to confirm")
    ref_result = capture_reference_multi(cam, roi, preprocessor)
    if ref_result is None:
        print("[INFO] Reference capture cancelled.  Exiting.")
        cam.release()
        sys.exit(0)

    ref_grays, ref_templates = ref_result
    print(
        f"[INFO] {len(ref_grays)} reference angle(s) captured.  "
        f"Shape: {ref_grays[0].shape}"
    )

    # ---- Step 3: Position lock -----------------------------------------
    position_lock: PositionLock | None = None
    if POSITION_LOCK_ENABLED:
        focused_tpls = []
        tpl_offsets  = []
        for rg, rt in zip(ref_grays, ref_templates):
            tpl, off = _focused_template(rg, rt)
            focused_tpls.append(tpl)
            tpl_offsets.append(off)
        position_lock = PositionLock(
            focused_tpls,
            roi_offsets   = tpl_offsets,
            full_roi_size = (roi[2], roi[3]),
        )
        print(
            f"[INFO] Position lock ON — {len(focused_tpls)} template(s), "
            f"sizes {[f'{t.shape[1]}x{t.shape[0]}' for t in focused_tpls]}"
        )
    else:
        print("[INFO] Position lock OFF — fixed ROI mode")

    # ---- Step 4: Inspection loop ---------------------------------------
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
