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
REFERENCE_FRAME_COUNT   = 10   # Frames averaged to build the clean reference
REFERENCE_WARMUP_FRAMES = 10   # Discard this many frames before capturing

# ---- Preprocessing ------------------------------------------
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DENOISE_KERNEL_SIZE  = 3       # Gaussian blur kernel (must be odd); 1 = disabled

# Background normalisation: estimate the slow-varying illumination gradient
# by blurring with a large Gaussian, then divide the image by it.
# This removes bottle-surface texture and uneven lighting before CLAHE.
# Set to 0 to disable.
TOPHAT_BG_SIGMA = 25           # Gaussian sigma for background estimation

# ---- Alignment ----------------------------------------------
ALIGN_ENABLED         = True
ALIGN_MAX_SHIFT_RATIO = 0.25   # Max translation as fraction of ROI dimension
ALIGN_MAX_ROTATION_DEG = 5.0   # Reject ECC result if estimated rotation > this

# ECC (Enhanced Correlation Coefficient) parameters.
# ECC handles translation + rotation; far more robust than phase correlation.
ALIGN_ECC_MAX_ITER = 50        # Max iterations per frame
ALIGN_ECC_EPSILON  = 0.001     # Convergence threshold

# ---- Text binarisation (inspector) --------------------------
# Blackhat morphology isolates dark ink by local contrast, independent of
# absolute brightness.  Kernel should be larger than one stroke width but
# smaller than the gap between characters.
TEXT_BH_SIZE            = 21   # Blackhat/tophat kernel size (px); increase for large text
TEXT_MIN_COMPONENT_AREA = 20   # Min connected-component px² counted as real text

# Tolerance dilation: before comparing missing/extra, both masks are dilated
# by this many pixels so that sub-pixel stroke-boundary differences (caused
# by slight lighting or threshold variation) are absorbed without hiding real
# defects like missing characters or large smears.
TEXT_TOLERANCE_PX       = 3

# Legacy adaptive-threshold params (no longer used; kept for reference)
ADAPTIVE_BLOCK_SIZE     = 31
ADAPTIVE_C              = 8

# ---- Text-structure comparison weights ----------------------
# Defect score = RECALL_W*(1−recall) + PURITY_W*(1−purity) + NCC_W*(1−ncc)
RECALL_WEIGHT = 0.50           # Missing ink / broken / faded characters
PURITY_WEIGHT = 0.25           # Extra ink / smear / debris
NCC_WEIGHT    = 0.25           # Overall structural shape mismatch

# ---- Defect decision ----------------------------------------
DEFECT_SCORE_THRESHOLD = 0.20  # 0 = no defect, 1 = worst; tuned for binary-mask scoring

# ---- Temporal consistency filter ----------------------------
TEMPORAL_WINDOW       = 6      # Rolling window length (frames)
TEMPORAL_DEFECT_RATIO = 0.50   # Fraction of window frames that must flag defect

# ---- Visualization ------------------------------------------
HEATMAP_ALPHA        = 0.45    # 0 = no heatmap overlay, 1 = full
ROI_BORDER_THICKNESS = 3
PANEL_CELL_SCALE     = 2.5     # Display scale multiplier per panel cell
CORNER_ACCENT_LENGTH = 18      # Length of corner bracket lines on main feed

# ---- Position Lock (moving object tracking) -----------------
POSITION_LOCK_ENABLED        = True
POSITION_LOCK_THRESHOLD      = 0.72
POSITION_LOCK_SEARCH_MARGIN  = 80
POSITION_LOCK_BLUR_THRESHOLD = 30.0

# ---- Logging ------------------------------------------------
LOG_ENABLED        = True
LOG_DIR            = "logs"
SNAPSHOT_ON_DEFECT = False

# ---- Legacy parameters (kept for reference; not used by current inspector) --
SSIM_THRESHOLD              = 0.80
SSIM_WIN_SIZE               = 5
EDGE_DIFF_THRESHOLD         = 0.06
PIXEL_DIFF_THRESHOLD        = 15
CHANGED_PIXEL_RATIO_THRESHOLD = 0.0
SSIM_WEIGHT                 = 0.50
EDGE_WEIGHT                 = 0.25
PIXEL_WEIGHT                = 0.25
EDGE_SCORE_SCALE            = 4.0
PIXEL_SCORE_SCALE           = 8.0
