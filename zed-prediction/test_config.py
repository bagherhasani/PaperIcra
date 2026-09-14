TEST_NAME = "block2_now_sanity"
MOTION_MODEL = "hip_steering"
STEERING_GAIN_B = 3.0
PRED_MIN_SPEED = 0.30
EVAL_SKIP_S = 2.0

# Block 2 A/B switch — start with "now", then flip to "pred"
NAV_CLOUD_MODE = "pred"

GONZALEZ_K = 0.5
LATERAL_LP_ALPHA = 0.15
SPEED_EMA_ALPHA = 0.25
PROFIDEA2_ALPHA = 0.05
PROFIDEA2_BETA = 1.5
PROFIDEA3_K = 0.35

# Live ZED for Block 2 (comment out SVO). For SVO replay keep the path.
# SVO_PATH = "recordings/D-walk.svo2"
SVO_PATH = ""
