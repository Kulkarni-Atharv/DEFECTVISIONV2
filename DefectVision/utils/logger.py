from __future__ import annotations
import csv
import os
import time
from datetime import datetime
from config import LOG_DIR, LOG_ENABLED, SNAPSHOT_ON_DEFECT


class DefectLogger:
    """
    Writes per-frame inspection results to a timestamped CSV file.
    On SNAPSHOT_ON_DEFECT=True, saves the raw ROI image alongside the log.
    """

    _HEADER = [
        "timestamp", "frame",
        "defect_score", "ssim_score", "edge_diff_score", "pixel_diff_score",
        "confirmed_defect",
    ]

    def __init__(self) -> None:
        self._enabled = LOG_ENABLED
        self._defect_count = 0
        self._total_frames = 0
        self._session_start = time.monotonic()
        self._csv_path: str | None = None

        if not self._enabled:
            return

        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(LOG_DIR, f"inspection_{ts}.csv")
        with open(self._csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(self._HEADER)

    # ------------------------------------------------------------------

    def log(
        self,
        frame_num: int,
        result,
        confirmed_defect: bool,
        smoothed_score: float,
        roi_bgr=None,
    ) -> None:
        self._total_frames += 1
        if confirmed_defect:
            self._defect_count += 1

        if not self._enabled or self._csv_path is None:
            return

        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            frame_num,
            f"{smoothed_score:.4f}",
            f"{result.ssim_score:.4f}",
            f"{result.edge_diff_score:.4f}",
            f"{result.pixel_diff_score:.4f}",
            int(confirmed_defect),
        ]
        with open(self._csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

        if SNAPSHOT_ON_DEFECT and confirmed_defect and roi_bgr is not None:
            import cv2
            snap_path = os.path.join(
                LOG_DIR,
                f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            )
            cv2.imwrite(snap_path, roi_bgr)

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        elapsed = max(time.monotonic() - self._session_start, 1e-6)
        return {
            "total_frames": self._total_frames,
            "defect_count": self._defect_count,
            "defect_rate_pct": round(100 * self._defect_count / max(self._total_frames, 1), 2),
            "duration_s": round(elapsed, 1),
            "avg_fps": round(self._total_frames / elapsed, 1),
            "log_path": self._csv_path,
        }
