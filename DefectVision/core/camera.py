"""
Camera abstraction — unified read() / release() / is_opened() interface.

Backends:
  OpenCVCamera    → cv2.VideoCapture  (Windows dev / generic Linux)
  PiCamera2Camera → picamera2          (Raspberry Pi CM5)

Both return BGR frames so the rest of the pipeline is backend-agnostic.
Select via CAMERA_BACKEND in config.py.
"""
from __future__ import annotations
import time
import cv2
import numpy as np


class OpenCVCamera:
    _BACKEND_FLAGS = {
        "DSHOW":  cv2.CAP_DSHOW,
        "V4L2":   cv2.CAP_V4L2,
        "AUTO":   cv2.CAP_ANY,
        "OPENCV": cv2.CAP_ANY,
    }

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str = "AUTO",
        warmup_frames: int = 10,
    ) -> None:
        flag = self._BACKEND_FLAGS.get(backend.upper(), cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(index, flag)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        for _ in range(warmup_frames):
            self._cap.read()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    def get_resolution(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS))


class PiCamera2Camera:
    """
    Picamera2 backend for CM5 / IMX296 global-shutter sensor.

    Uses the default picamera2 preview format (XBGR8888, 4-channel).
    capture_array() may return 3-channel (RGB) or 4-channel (RGBA/XBGR)
    depending on the platform; both are handled and converted to BGR.
    """

    def __init__(
        self,
        width: int = 1456,
        height: int = 1088,
        fps: int = 30,
        warmup_s: float = 2.0,
    ) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "[ERROR] picamera2 not found.\n"
                "  sudo apt install -y python3-picamera2"
            ) from exc

        self._picam2 = Picamera2()
        # No explicit format — let picamera2 pick its native default (XBGR8888).
        # Forcing RGB888/BGR888 causes colour inversions on some firmware versions.
        cfg = self._picam2.create_preview_configuration(
            main={"size": (width, height)},
            display=None,   # OpenCV owns the display
        )
        self._picam2.configure(cfg)
        self._picam2.start()
        time.sleep(warmup_s)   # IMX296 AEC/AWB settle
        self._width  = width
        self._height = height

    def is_opened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        frame = self._picam2.capture_array()
        # picamera2 default format is XBGR8888 (4-ch) on most firmware,
        # but may be RGB (3-ch) on others — handle both.
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self) -> None:
        self._picam2.stop()

    def get_resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def get_fps(self) -> float:
        return 30.0


def create_camera(backend: str, **kwargs):
    """
    Factory — returns an initialised camera object.

    backend="PICAMERA2"            → PiCamera2Camera
    backend="DSHOW"/"V4L2"/"AUTO"  → OpenCVCamera
    """
    if backend.strip().upper() == "PICAMERA2":
        return PiCamera2Camera(
            width    = kwargs.get("width",    1456),
            height   = kwargs.get("height",   1088),
            fps      = kwargs.get("fps",      30),
            warmup_s = kwargs.get("warmup_s", 2.0),
        )
    return OpenCVCamera(
        index         = kwargs.get("index",         0),
        width         = kwargs.get("width",         1280),
        height        = kwargs.get("height",        720),
        fps           = kwargs.get("fps",           30),
        backend       = backend,
        warmup_frames = kwargs.get("warmup_frames", 10),
    )
