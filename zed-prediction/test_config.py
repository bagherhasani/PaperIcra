# ── Live danger-zone demo (walk in front of ZED + RViz) ──────────────────────
TEST_NAME = "live_hip_rviz"
MOTION_MODEL = "hip_steering"

STEERING_GAIN_B = 3.0
PROFIDEA2_ALPHA = 0.05
PROFIDEA2_BETA  = 1.5
PROFIDEA3_K = 0.35

# Live camera (empty = real ZED, not SVO)
SVO_PATH = ""

# ── Paper 1s SVO runs (comment live above, uncomment one below) ───────────────
# TEST_NAME = "ctrv_Dwalk_1s"
# TEST_NAME = "hip_Dwalk_1s"
# TEST_NAME = "ctrv_straight_1s"
# TEST_NAME = "hip_straight_1s"
# MOTION_MODEL = "ctrv"
# SVO_PATH = "recordings/straight-walk.svo2"
# SVO_PATH = "recordings/D-walk.svo2"
