"""
Live capture + CRAFT text-region detection for Raspberry Pi CM5 with Picamera2.

CRAFT detects WHERE text is (bounding boxes / polygons).  It does NOT read the
text — use a recognition model (e.g. TrOCR, EasyOCR) downstream if you need
the actual characters.

Controls:
  SPACE  -> capture a clean frame, run CRAFT, draw boxes, save result, then exit
  Q      -> quit without capturing

Install on CM5:
  pip install -r requirements_cm5.txt
  sudo apt install -y python3-picamera2   # if not already present

Usage (run from repo root):
  python capture_cm5.py --model weights/craft_mlt_25k.pth
  python capture_cm5.py --model weights/craft_mlt_25k.pth --roi 100 50 800 600
  python capture_cm5.py --model weights/craft_mlt_25k.pth --refine --refiner_model weights/craft_refiner_CTW1500.pth

CRAFT source files (craft.py, craft_utils.py, imgproc.py, refinenet.py) must be
present in the repo root. Clone from: https://github.com/clovaai/CRAFT-pytorch
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.autograd import Variable

import craft_utils
import imgproc
from craft import CRAFT

# ── colour palette ─────────────────────────────────────────────────────────────
YELLOW = (0, 200, 255)
GREEN  = (0, 220, 60)
BLACK  = (0, 0, 0)


# ── model loading ──────────────────────────────────────────────────────────────

def _clean_state_dict(state_dict: dict) -> OrderedDict:
    """Strip DataParallel 'module.' prefix if present."""
    start = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return OrderedDict(
        {".".join(k.split(".")[start:]): v for k, v in state_dict.items()}
    )


def load_craft(model_path: str) -> CRAFT:
    net = CRAFT()
    net.load_state_dict(
        _clean_state_dict(torch.load(model_path, map_location="cpu"))
    )
    net.eval()
    return net


def load_refiner(refiner_path: str):
    from refinenet import RefineNet
    refiner = RefineNet()
    refiner.load_state_dict(
        _clean_state_dict(torch.load(refiner_path, map_location="cpu"))
    )
    refiner.eval()
    return refiner


# ── CRAFT inference ────────────────────────────────────────────────────────────

def detect_regions(
    net: CRAFT,
    image_rgb: np.ndarray,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
    poly: bool = False,
    refine_net=None,
) -> tuple[list, list, np.ndarray]:
    """
    Run CRAFT on a single RGB image (H x W x 3, uint8).

    Returns
    -------
    boxes   : list of np.ndarray[4, 2]  - quad bounding boxes in original coords
    polys   : list of np.ndarray[N, 2]  - tight polygons (same as boxes if poly=False)
    heatmap : BGR uint8 visualisation of text + link score maps
    """
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_rgb, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1.0 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)   # HWC -> CHW
    x = Variable(x.unsqueeze(0))                # add batch dim

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_ref = refine_net(y, feature)
        score_link = y_ref[0, :, :, 0].cpu().numpy()
        poly = True

    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )
    boxes  = craft_utils.adjustResultCoordinates(boxes,  ratio_w, ratio_h)
    polys  = craft_utils.adjustResultCoordinates(polys,  ratio_w, ratio_h)
    for i, p in enumerate(polys):
        if p is None:
            polys[i] = boxes[i]

    render = np.hstack((score_text, score_link))
    heatmap = imgproc.cvt2HeatmapImg(render)

    return boxes, polys, heatmap


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, polys: list, color=GREEN, thickness: int = 2) -> None:
    """Draw detected text regions onto frame in-place."""
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_hud(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        "LIVE PREVIEW  |  SPACE = Detect & Exit    Q = Quit",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA,
    )


# ── terminal output ────────────────────────────────────────────────────────────

def print_results(polys: list, elapsed: float, save_path: str | None) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CRAFT Text Regions  ({elapsed * 1000:.1f} ms)")
    print(sep)

    if not polys:
        print("  No text regions detected.")
    else:
        for i, poly in enumerate(polys, 1):
            pts = np.array(poly, dtype=np.int32)
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            print(f"  [{i:02d}]  bbox=({x0},{y0})->({x1},{y1})  "
                  f"w={x1-x0}  h={y1-y0}")

    if save_path:
        print(f"\n  Result saved -> {save_path}")
    print(f"{sep}\n")


# ── camera ─────────────────────────────────────────────────────────────────────

def start_camera():
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "picamera2 not found.\n"
            "  sudo apt install -y python3-picamera2\n"
            "  # or: pip install picamera2"
        ) from e

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (1456, 1088), "format": "RGB888"},
        display=None,   # OpenCV handles display
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(2.0)   # IMX296 global shutter needs ~2 s for AEC/AWB to settle
    return picam2


def capture_frame(picam2) -> np.ndarray:
    frame = picam2.capture_array("main")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# ── main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── load models ──
    print(f"[CM5] Loading CRAFT from {args.model} ...")
    net = load_craft(args.model)

    refine_net = None
    if args.refine:
        print(f"[CM5] Loading RefineNet from {args.refiner_model} ...")
        refine_net = load_refiner(args.refiner_model)

    print("[CM5] Starting camera ...")
    picam2 = start_camera()
    print("[CM5] Camera ready — SPACE to capture, Q to quit")

    roi = args.roi   # (x, y, w, h) or None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            frame = capture_frame(picam2)

            if roi:
                x, y, w, h = roi
                display = frame[y: y + h, x: x + w].copy()
            else:
                display = frame.copy()

            draw_hud(display)
            cv2.imshow("CRAFT CM5 - Live Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord(" "):
                # ── clean capture (no HUD) ──
                raw = capture_frame(picam2)
                crop = (raw[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
                        if roi else raw)

                # BGR (OpenCV) -> RGB (CRAFT expects RGB)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                print("\n[CM5] Detecting text regions ...")
                t0 = time.time()
                boxes, polys, heatmap = detect_regions(
                    net, rgb,
                    text_threshold=args.text_threshold,
                    link_threshold=args.link_threshold,
                    low_text=args.low_text,
                    canvas_size=args.canvas_size,
                    mag_ratio=args.mag_ratio,
                    poly=args.poly,
                    refine_net=refine_net,
                )
                elapsed = time.time() - t0

                # ── draw & save ──
                result_img = crop.copy()
                draw_boxes(result_img, polys)

                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path  = str(save_dir / f"craft_{ts}.jpg")
                heat_path = str(save_dir / f"craft_{ts}_heatmap.jpg")
                cv2.imwrite(out_path,  result_img)
                cv2.imwrite(heat_path, heatmap)

                # ── display result ──
                cv2.imshow("CRAFT Result", result_img)
                cv2.imshow("CRAFT Heatmap", heatmap)
                cv2.waitKey(0)

                print_results(polys, elapsed, out_path)
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("[CM5] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAFT live text detection on CM5")

    # model
    parser.add_argument("--model", default="weights/craft_mlt_25k.pth",
                        help="Path to CRAFT .pth weights")
    parser.add_argument("--refine", action="store_true",
                        help="Enable link refiner (improves curved/connected text)")
    parser.add_argument("--refiner_model", default="weights/craft_refiner_CTW1500.pth",
                        help="Path to RefineNet .pth weights")

    # detection thresholds
    parser.add_argument("--text_threshold", type=float, default=0.7,
                        help="Text confidence threshold (default: 0.7)")
    parser.add_argument("--link_threshold", type=float, default=0.4,
                        help="Link confidence threshold (default: 0.4)")
    parser.add_argument("--low_text",       type=float, default=0.4,
                        help="Low-bound text score (default: 0.4)")
    parser.add_argument("--canvas_size",    type=int,   default=1280,
                        help="Max image dimension fed to CRAFT (default: 1280)")
    parser.add_argument("--mag_ratio",      type=float, default=1.5,
                        help="Magnification ratio before resizing (default: 1.5)")
    parser.add_argument("--poly",           action="store_true",
                        help="Output tight polygons instead of quad boxes")

    # camera / I/O
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Camera crop region in pixels: X Y W H")
    parser.add_argument("--save_dir", default="results_cm5",
                        help="Directory to save annotated images (default: results_cm5/)")

    main(parser.parse_args())
