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
ALIGN_MAX_ROTATION_DEG = 10.0  # Reject ECC result if estimated rotation > this

# ECC (Enhanced Correlation Coefficient) parameters.
# ECC handles translation + rotation; far more robust than phase correlation.
ALIGN_ECC_MAX_ITER = 25        # 25 sufficient — NCC pre-filter picks the right ref so ECC starts close
ALIGN_ECC_EPSILON  = 0.001     # Convergence threshold

# ---- Text binarisation (inspector) --------------------------
# Blackhat morphology isolates dark ink by local contrast, independent of
# absolute brightness.  Kernel should be larger than one stroke width but
# smaller than the gap between characters.
TEXT_BH_SIZE            = 21   # Blackhat/tophat kernel size (px); increase for large text
TEXT_MIN_COMPONENT_AREA = 20   # Min connected-component px² counted as real text

# Tolerance dilation applied before missing / extra comparison.
# Two separate values because the two signals have different sensitivity needs:
#
#   RECALL tolerance (missing ink): generous — stroke-boundary pixels can shift
#     2-3 px with lighting/threshold changes; absorbing them prevents false
#     "missing character" detections.
#
#   PURITY tolerance (extra ink): 2 px absorbs typical stroke-boundary shifts
#     from frame-to-frame lighting/alignment variation without masking real marks.
#     1 px was too tight — it let boundary artifacts through as extra-ink clusters.
TEXT_TOLERANCE_RECALL_PX = 4   # stroke-boundary tolerance inside the text crop
TEXT_TOLERANCE_PURITY_PX = 2   # tight: catches real smears, rejects aliasing noise

# Margin (px) added around the text bounding box before comparison.
# Large enough to capture nearby debris; small enough to exclude
# unrelated background marks on the other side of the ROI.
TEXT_CROP_MARGIN = 14

# Purity zone: extra-ink (debris) is only flagged within this many px of
# a reference text stroke.  Background artifacts at the crop margin are
# ignored.  Increase if debris far from strokes is being missed.
DEBRIS_TEXT_ZONE_PX = 18

# NCC match threshold: if the best reference NCC score is below this,
# the angle was not captured.  Switch to reference-free debris-only mode
# so the system uses its own judgement rather than a mismatched reference.
NCC_MATCH_THRESHOLD = 0.45

# Debris hard override: after extra-ink detection, find connected components.
# Any single component ≥ this many px² that survived the purity tolerance is
# genuine debris (dot, smear, added stroke) — flag as DEFECT immediately
# regardless of composite score.
# 30 px² ≈ a 6×5 compact mark — well below any user-visible dot (~5 px radius
# = 78 px²) but above stroke-boundary noise strips (typically thin and < 20 px²).
DEBRIS_MIN_COMPONENT_AREA = 50   # raised from 30 — edge-shift artifacts are thin strips < 50px²

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

# Multi-reference early exit: when checking multiple references per frame,
# stop as soon as one scores below this value — the print is clearly clean.
# Saves 5-6 ECC alignment calls on the common (clean-print) path.
INSPECT_EARLY_EXIT_SCORE = 0.05

# Single-reference fast path: when position lock confidence exceeds this,
# only inspect against the matching reference (skip all other angles).
# The NCC template score is already a strong angle-identity signal; at 0.60+
# it reliably identifies which captured angle is in the frame, so running
# ECC + inspect against all other angles just wastes CPU.
POSITION_LOCK_SINGLE_REF_CONF = 0.60

# ---- Multi-reference video calibration ----------------------
# Record a short video of a clean print at different angles; the system
# automatically picks up to MAX_REFERENCES frames that are structurally
# distinct (pairwise NCC below REF_MIN_DISTINCTNESS).
MAX_REFERENCES       = 7     # Hard cap on stored reference frames
REF_MIN_DISTINCTNESS = 0.80  # NCC threshold: frame added only when NCC vs
                              # every already-selected ref is below this value

# ---- Temporal consistency filter ----------------------------
TEMPORAL_WINDOW       = 6      # Rolling window length (frames)
TEMPORAL_DEFECT_RATIO = 0.50   # Fraction of window frames that must flag defect

# ---- Visualization ------------------------------------------
HEATMAP_ALPHA        = 0.45    # 0 = no heatmap overlay, 1 = full
ROI_BORDER_THICKNESS = 3
PANEL_CELL_SCALE     = 2.5     # Display scale multiplier per panel cell
CORNER_ACCENT_LENGTH = 18      # Length of corner bracket lines on main feed

# ---- Position Lock (moving object tracking) -----------------
POSITION_LOCK_ENABLED        = False
POSITION_LOCK_THRESHOLD      = 0.45   # tight text-crop templates match at ~0.45+ even with angle/lighting variation
POSITION_LOCK_SEARCH_MARGIN  = 80
# Blur gate: Laplacian variance of matched crop must exceed this.
# Large ROIs containing mostly smooth surface (bottle, label backing) score
# 10-20 even when perfectly in focus — the old value of 30 falsely rejected them.
POSITION_LOCK_BLUR_THRESHOLD = 8.0

# ---- Logging ------------------------------------------------
LOG_ENABLED        = False
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
