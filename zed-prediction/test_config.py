# ── Change this before each run ──────────────────────────────────────────────
# Paper Phase A: horizon is locked to 1.0 s in zed2.py
# Run order: CTRV → hip_steering → (later) static baseline

# TEST_NAME = "ctrv_Dwalk_1s"
# TEST_NAME = "hip_Dwalk_1s"
# TEST_NAME = "ctrv_straight_1s"
TEST_NAME = "hip_straight_1s"

# ── Motion model ──────────────────────────────────────────────────────────────
# "ctrv"         — Constant Turn Rate & Velocity (baseline)
# "hip_steering" — Prof idea 1: heading steered toward hip orientation
# "profidea2"    — position correction (not for main paper table)
# "profidea3"    — speed-coupled hip (not for main paper table)

# MOTION_MODEL = "ctrv"
MOTION_MODEL = "hip_steering"
# MOTION_MODEL = "profidea2"
# MOTION_MODEL = "profidea3"

STEERING_GAIN_B = 3.0

PROFIDEA2_ALPHA = 0.05
PROFIDEA2_BETA  = 1.5
PROFIDEA3_K = 0.35

# ── SVO replay ───────────────────────────────────────────────────────────────
SVO_PATH = "recordings/straight-walk.svo2"
# SVO_PATH = "recordings/D-walk.svo2"
# SVO_PATH = ""   # live camera
