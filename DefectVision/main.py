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


def _batch_ncc(live: np.ndarray, refs: list) -> np.ndarray:
    """Vectorised NCC: score live against all refs in one matrix multiply.
    ~1 ms for 30 refs at 1/4 scale vs ~9 ms for a Python loop.
    Returns float32 array of NCC scores, one per reference.
    """
    scale = 4
    h, w  = live.shape[:2]
    H, W  = max(1, h // scale), max(1, w // scale)

    live_v = cv2.resize(live, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    live_v -= live_v.mean()
    live_norm = np.sqrt(np.dot(live_v, live_v))

    ref_mat = np.stack([
        cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        for r in refs
    ])                                   # shape: (N, H*W)
    ref_mat -= ref_mat.mean(axis=1, keepdims=True)
    ref_norms = np.sqrt((ref_mat * ref_mat).sum(axis=1))

    scores = (ref_mat @ live_v) / (ref_norms * live_norm + 1e-8)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


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
# Detection worker — runs in background thread
# ====================================================================

def _run_detection(
    frame: np.ndarray,
    current_roi: tuple,
    ref_grays: list,
    preprocessor: Preprocessor,
    aligner: Aligner,
    inspector: Inspector,
    match_conf: float,
    best_tpl_idx: int,
) -> tuple:
    """Extract ROI → preprocess → NCC rank → ECC align → inspect.
    All CPU-bound work is here so the main thread is never blocked.
    Returns (result, roi_bgr, best_ref, best_live_aligned).
    """
    roi_bgr   = _grab_roi(frame, current_roi)
    live_gray = preprocessor.process(roi_bgr)

    if live_gray.shape != ref_grays[0].shape:
        live_gray = cv2.resize(
            live_gray, (ref_grays[0].shape[1], ref_grays[0].shape[0])
        )

    if match_conf >= POSITION_LOCK_SINGLE_REF_CONF:
        check_indices = [min(best_tpl_idx, len(ref_grays) - 1)]
    elif len(ref_grays) == 1:
        check_indices = [0]
    else:
        # Vectorised batch NCC ranks all refs in ~1ms regardless of N.
        # Cap to top-2: with good NCC ranking the right reference is always
        # in the top-2, so we never need to run ECC on angles 3-30.
        ncc_scores    = _batch_ncc(live_gray, ref_grays)
        top2          = np.argsort(ncc_scores)[::-1][:2].tolist()
        check_indices = top2

    best_result = None
    best_ref    = ref_grays[0]
    best_live   = live_gray
    for i in check_indices:
        ref     = ref_grays[i]
        inspector.set_reference(ref)   # instant after first call — result is cached
        aligned = aligner.align(ref, live_gray)
        res     = inspector.inspect(ref, aligned)
        if best_result is None or res.defect_score < best_result.defect_score:
            best_result = res
            best_ref    = ref
            best_live   = aligned
        if best_result.defect_score < INSPECT_EARLY_EXIT_SCORE:
            break

    return best_result, roi_bgr, best_ref, best_live


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
    from concurrent.futures import ThreadPoolExecutor

    # Pre-warm binarization cache for all references so the first
    # inspection frame pays zero set_reference() cost.
    inspector.clear_cache()
    for _r in ref_grays:
        inspector.set_reference(_r)
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

    # Last-known detection state — overlaid on the live feed every frame
    smoothed_score   = 0.0
    confirmed_defect = False
    warming_up       = True
    match_conf       = 0.0
    current_roi      = roi
    best_tpl_idx     = 0
    paused           = False
    last_panel       = None
    last_roi_bgr     = np.zeros((4, 4, 3), dtype=np.uint8)
    det_future       = None

    grabber = _FrameGrabber(cap)
    print("[INFO] Inspection running.  Q=quit  R=new reference  S=snapshot  SPACE=pause")

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            if not paused:
                ret, frame = grabber.read()
                if not ret:
                    cv2.waitKey(1)
                    continue

                frame_num   += 1
                fps_counter += 1
                if fps_counter >= 30:
                    fps         = fps_counter / max(time.monotonic() - fps_t0, 1e-6)
                    fps_t0      = time.monotonic()
                    fps_counter = 0

                # ---- Position lock -----------------------------------
                if position_lock is not None:
                    frame_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    match = position_lock.find(frame_gray_full)
                    if match is None:
                        main_display = visualizer.draw_main_overlay(
                            frame.copy(), current_roi,
                            confirmed_defect=False, smoothed_score=0.0,
                            warming_up=False, match_conf=0.0, searching=True,
                        )
                        cv2.putText(main_display,
                                    f"FPS: {fps:.1f}  Frame: {frame_num}",
                                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (200, 200, 200), 1)
                        cv2.imshow(WIN_MAIN, main_display)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            paused = not paused
                        continue
                    current_roi, match_conf, best_tpl_idx = match
                else:
                    current_roi  = roi
                    match_conf   = 0.0
                    best_tpl_idx = 0

                # ---- Harvest completed detection ---------------------
                if det_future is not None and det_future.done():
                    try:
                        det_result, roi_bgr, best_ref, best_live = det_future.result()
                        _wu = not temporal.window_full
                        smoothed_score, confirmed_defect = temporal.update(
                            det_result.defect_score, det_result.is_defect
                        )
                        if _wu:
                            confirmed_defect = False
                        warming_up   = _wu
                        last_roi_bgr = roi_bgr
                        logger.log(frame_num, det_result, confirmed_defect,
                                   smoothed_score, roi_bgr)
                        last_panel = visualizer.build_panel(
                            roi_bgr, best_ref, best_live,
                            det_result, confirmed_defect, smoothed_score,
                            fps, warming_up, match_conf,
                        )
                    except Exception as e:
                        print(f"[WARN] Detection error: {e}")
                    det_future = None

                # ---- Submit new detection if worker is free ----------
                if det_future is None:
                    det_future = executor.submit(
                        _run_detection,
                        frame.copy(), current_roi, ref_grays[:],
                        preprocessor, aligner, inspector,
                        match_conf, best_tpl_idx,
                    )

                # ---- Live feed: always updated, never blocked --------
                main_display = visualizer.draw_main_overlay(
                    frame.copy(), current_roi, confirmed_defect, smoothed_score,
                    warming_up, match_conf,
                )
                cv2.putText(main_display,
                            f"FPS: {fps:.1f}  Frame: {frame_num}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (200, 200, 200), 1)
                cv2.imshow(WIN_MAIN, main_display)
                if last_panel is not None:
                    cv2.imshow(WIN_PANEL, last_panel)

            else:
                cv2.waitKey(50)

            # ---- Key handling ----------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")

            elif key == ord('r'):
                print("[INFO] Recapturing references — position clean sample in ROI ...")
                if det_future is not None:
                    try:
                        det_future.result(timeout=2.0)
                    except Exception:
                        pass
                    det_future = None
                grabber.stop()
                cap_result = capture_reference_multi(cap, roi, preprocessor)
                grabber = _FrameGrabber(cap)
                if cap_result is not None:
                    ref_grays, ref_templates = cap_result
                    inspector.clear_cache()
                    for _r in ref_grays:
                        inspector.set_reference(_r)
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
                cv2.imwrite(snap_path, last_roi_bgr)
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
