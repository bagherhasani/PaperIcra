TEST_NAME = "block2_B_pred"
MOTION_MODEL = "hip_steering"
STEERING_GAIN_B = 3.0
PRED_MIN_SPEED = 0.30
PRED_SPEED_OFF = 0.20
EVAL_SKIP_S = 2.0

# Block 2 A/B: "now" | "pred"
NAV_CLOUD_MODE = "pred"
# Nav/RViz costmap horizon (seconds). Paper ADE eval stays 1.0 s in zed2.
NAV_PRED_HORIZON = 2.0
RVIZ_PRED_HORIZON = 2.0

# Publish / tracking gates
BODY_MIN_CONF = 40
TRACK_LOST_S = 1.0
CLOUD_EMA_ALPHA = 0.35
# Camera optical center → base_link on the ground plane (m). Tune if ZED is
# not above base_link origin. +X forward, +Y left (ROS).
CAM_TO_BASE_X = 0.0
CAM_TO_BASE_Y = 0.0

GONZALEZ_K = 0.5
LATERAL_LP_ALPHA = 0.15
SPEED_EMA_ALPHA = 0.25
PROFIDEA2_ALPHA = 0.05
PROFIDEA2_BETA = 1.5
PROFIDEA3_K = 0.35

# Live ZED for Block 2 (empty = live camera)
# SVO_PATH = "recordings/D-walk.svo2"
SVO_PATH = ""
