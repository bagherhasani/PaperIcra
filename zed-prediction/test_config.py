# ── Change this before each run ──────────────────────────────────────────────

# TEST_NAME = "straight_replay_ctrv"
# TEST_NAME = "straight_replay_hip_steering"
# TEST_NAME = "straight_replay_profidea2"
# TEST_NAME = "straight_replay_profidea3"
# TEST_NAME = "ctrv-Dwalk"
# TEST_NAME = "profidea1-Dwalk"
# TEST_NAME = "profidea2-Dwalk"
TEST_NAME = "profidea1-Dwalk-4.8ahead"

# ── Motion model ──────────────────────────────────────────────────────────────
# "ctrv"        — standard Constant Turn Rate & Velocity (kinematic baseline)
# "hip_steering" — Prof idea 1: heading steered toward hip orientation
# "profidea2"   — Prof idea 2: position correction via hip/velocity misalignment
# "profidea3"   — Prof idea 3: hip-steering + turn-coupled speed reduction

# MOTION_MODEL = "ctrv"
MOTION_MODEL = "hip_steering"
# MOTION_MODEL = "profidea2"
# MOTION_MODEL = "profidea3"

# hip_steering gain — all hip-based models use this
STEERING_GAIN_B = 3.0

# profidea2 params
PROFIDEA2_ALPHA = 0.05
PROFIDEA2_BETA  = 1.5

# profidea3 params:
#   PROFIDEA3_K — deceleration gain (0=same as profidea1, 0.35=35% slower at 90° turn)
PROFIDEA3_K = 0.35

# ── SVO replay ───────────────────────────────────────────────────────────────
# SVO_PATH = "recordings/straight-walk.svo2"
SVO_PATH = "recordings/D-walk.svo2"
# SVO_PATH = ""   # live camera
