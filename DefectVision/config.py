# ============================================================
# DefectVision — Print Defect Inspection System
# All tunable parameters live here.
# ============================================================

# ---- Camera -------------------------------------------------
# Backend options:
#   "PICAMERA2"  → picamera2 (Raspberry Pi CM5 / Pi 5)
#   "DSHOW"      → cv2.VideoCapture with DirectShow (Windows)
#   "V4L2"       → cv2.VideoCapture with V4L2 (Linux, non-Pi)
#   "AUTO"       → cv2.VideoCapture with auto backend
CAMERA_BACKEND = "PICAMERA2"

# OpenCV backend settings (used when CAMERA_BACKEND != "PICAMERA2")
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Picamera2 backend settings (used when CAMERA_BACKEND = "PICAMERA2")
PICAMERA2_WIDTH    = 1456   # IMX296 native width
PICAMERA2_HEIGHT   = 1088   # IMX296 native height
PICAMERA2_FPS      = 30
PICAMERA2_WARMUP_S = 2.0    # AEC/AWB settle time (seconds)

# ---- Reference capture --------------------------------------
REFERENCE_FRAME_COUNT = 10   # Frames averaged to build the clean reference
REFERENCE_WARMUP_FRAMES = 10 # Discard this many frames before capturing (sensor warm-up)

# ---- Preprocessing ------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DENOISE_KERNEL_SIZE = 3      # Gaussian blur kernel size (must be odd); 1 = disabled

# ---- Alignment (phase correlation) --------------------------
ALIGN_ENABLED = True
# Max fraction of ROI size allowed as shift — rejects unreliable correlations.
# With position lock active, template-match variance can be 10-15 px, so this
# must be large enough for the aligner to cover that residual offset.
ALIGN_MAX_SHIFT_RATIO = 0.25  # 25 % of ROI width/height (was 0.08)

# ---- Inspection thresholds ----------------------------------
# SSIM: 0 = completely different, 1 = identical
SSIM_THRESHOLD = 0.80
SSIM_WIN_SIZE = 5            # Smaller window = catches finer / more localised defects

# Edge difference: fraction of pixels whose edges differ
EDGE_DIFF_THRESHOLD = 0.06

# Raw pixel difference: absolute intensity difference per pixel (0–255)
PIXEL_DIFF_THRESHOLD = 15    # Lowered from 30 — catches subtle debris and fine strings

# ---- Hard pixel-change override -----------------------------
# If this fraction of ROI pixels exceeds PIXEL_DIFF_THRESHOLD, the frame is
# flagged as defective immediately — regardless of the composite SSIM score.
# This catches thin strings, fine debris, and small text additions that
# affect only a small area but are clearly real changes.
# Set to 0.0 to disable and rely on composite score only.
CHANGED_PIXEL_RATIO_THRESHOLD = 0.07    # 7 % — raised from 0.8 % because 1-px
# letter-edge halos on a moving object routinely affect 5–7 % of ROI pixels.

# ---- Defect scoring (weighted combination) ------------------
SSIM_WEIGHT   = 0.50
EDGE_WEIGHT   = 0.25
PIXEL_WEIGHT  = 0.25

# Scaling factors that normalise edge/pixel fraction scores into [0, 1].
# Lower values = more tolerant of alignment-induced edge halos on moving objects.
# Raise them back toward 6.0 / 12.0 for a static-camera setup.
EDGE_SCORE_SCALE  = 4.0   # applied to edge_diff_score  (was hardcoded 6.0)
PIXEL_SCORE_SCALE = 8.0   # applied to pixel_diff_score (was hardcoded 12.0)

# Combined defect score: 0.0 = perfect, 1.0 = severe defect.
# Raised from 0.18 — moving-object alignment noise raises the baseline score.
DEFECT_SCORE_THRESHOLD = 0.32

# ---- Temporal consistency filter ----------------------------
# Prevents single noisy frames from triggering false alarms.
TEMPORAL_WINDOW = 6          # Slightly shorter for faster response
TEMPORAL_DEFECT_RATIO = 0.50 # 50 % of window frames must flag (was 60 %)

# ---- Visualization ------------------------------------------
HEATMAP_ALPHA = 0.45         # 0 = no heatmap, 1 = full heatmap overlay
ROI_BORDER_THICKNESS = 3
PANEL_CELL_SCALE = 2.5       # Display scale multiplier for each panel cell
CORNER_ACCENT_LENGTH = 18    # Length of corner bracket lines on main feed

# ---- Position Lock (moving object tracking) -------------------------
# Replaces the fixed-ROI crop with template matching so the print region
# is found dynamically each frame, regardless of conveyor position.
POSITION_LOCK_ENABLED        = True
POSITION_LOCK_THRESHOLD      = 0.72   # min normalised match confidence (0–1)
POSITION_LOCK_SEARCH_MARGIN  = 80     # px around last position for fast search
POSITION_LOCK_BLUR_THRESHOLD = 30.0   # Laplacian variance below this = skip frame; 0 = disabled

# ---- Logging ------------------------------------------------
LOG_ENABLED = True
LOG_DIR = "logs"
SNAPSHOT_ON_DEFECT = False   # Auto-save ROI image on every confirmed defect
