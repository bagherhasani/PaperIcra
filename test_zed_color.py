"""
pytest test suite for ZedFollower pure-logic methods.

Covers (matching the actual current implementation):
  - stop_robot()
  - control_robot(distance, angle_rad)
  - search_for_target()
  - _shirt_color_stats(mask_color, poly_pts, valid_mask=None)
  - _bbox_from_4pts(bb2d)
  - _color_ratio_polygon_mask(mask_color, poly_pts)

Run with:
  cd "/home/user/ros2_ws/src/zed-detection " && python -m pytest test_zed_color.py -v
"""

import math
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch sys.modules BEFORE importing the module under test.
# The hyphen in the filename prevents normal import; we use importlib.util.
# ---------------------------------------------------------------------------

# --- pyzed.sl mock ---
sl_mock = MagicMock()
sl_mock.ERROR_CODE.SUCCESS = "SUCCESS"
sl_mock.Camera.return_value.open.return_value = "SUCCESS"
sys.modules["pyzed"]    = MagicMock()
sys.modules["pyzed.sl"] = sl_mock

# --- rclpy mocks ---
sys.modules["rclpy"]            = MagicMock()
sys.modules["rclpy.node"]       = MagicMock()
sys.modules["rclpy.node"].Node  = object   # plain object so __new__ works cleanly

# --- geometry_msgs mock (Twist must carry real attributes) ---
class MockVector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

class MockTwist:
    def __init__(self):
        self.linear  = MockVector3()
        self.angular = MockVector3()

geometry_msgs_mock = MagicMock()
geometry_msgs_mock.msg.Twist = MockTwist
sys.modules["geometry_msgs"]     = geometry_msgs_mock
sys.modules["geometry_msgs.msg"] = geometry_msgs_mock.msg

# --- tensorrt + pycuda mocks ---
sys.modules["tensorrt"]        = MagicMock()
sys.modules["pycuda"]          = MagicMock()
sys.modules["pycuda.driver"]   = MagicMock()
sys.modules["pycuda.autoinit"] = MagicMock()

# --- cv2 mock ---
import cv2 as _real_cv2   # use real cv2 so polygon/mask operations work in tests
sys.modules["cv2"] = _real_cv2

# --- Now import the module via importlib ---
import importlib.util

_MODULE_PATH = "/home/user/ros2_ws/src/zed-detection /zed-color.py"
spec   = importlib.util.spec_from_file_location("zed_color", _MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ZedFollower = module.ZedFollower


# ---------------------------------------------------------------------------
# Fixture: create a ZedFollower bypassing __init__
# ---------------------------------------------------------------------------

def make_node(**overrides):
    """Create a ZedFollower without opening any hardware, ready for unit tests."""
    node = ZedFollower.__new__(ZedFollower)

    # Core publisher / actuation
    node.cmd_pub          = MagicMock()
    node.actuation_enabled = True

    # Motion parameters (mirrors __init__ defaults)
    node.target_distance  = 1.0
    node.min_distance     = 0.6
    node.max_lin_speed    = 0.4
    node.max_ang_speed    = 0.8
    node.k_lin            = 0.7
    node.k_ang            = 1.0

    # State
    node.last_lin         = 0.0
    node.last_ang         = 0.0
    node.last_status      = "Idle"
    node.last_angle_error = 0.3

    # Search behaviour
    node.search_turn_speed = 0.5
    node.search_full_turn  = True
    node.search_angle_rad  = 0.0
    node._search_prev_t    = None

    # Color detection
    node.target_hsv_ranges       = [(np.array([108, 100, 40]), np.array([122, 255, 255]))]
    node.color_ratio_threshold   = 0.06

    # Re-ID (Chunk 1)
    node.reid                       = None
    node.target_embedding           = None
    node.reid_threshold             = 0.72
    node.reid_discrimination_margin = 0.12
    node._lock_confirm_streak       = 0
    node._lock_confirm_required     = 3
    node._confirm_candidate_id      = None
    node._pending_embeddings        = []

    # Logging
    node.get_logger = MagicMock(return_value=MagicMock())

    for key, val in overrides.items():
        setattr(node, key, val)

    return node


@pytest.fixture
def node():
    return make_node()


# ===========================================================================
# stop_robot()
# ===========================================================================

class TestStopRobot:
    def test_publishes_zero_twist(self, node):
        node.stop_robot()
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.linear.x  == 0.0
        assert twist.angular.z == 0.0

    def test_resets_last_lin_to_zero(self, node):
        node.last_lin = 0.35
        node.stop_robot()
        assert node.last_lin == 0.0

    def test_resets_last_ang_to_zero(self, node):
        node.last_ang = -0.4
        node.stop_robot()
        assert node.last_ang == 0.0

    def test_sets_status_stopped(self, node):
        node.stop_robot()
        assert node.last_status == "STOPPED"

    def test_always_publishes_even_when_actuation_disabled(self, node):
        """stop_robot must publish regardless of actuation_enabled — safety command."""
        node.actuation_enabled = False
        node.stop_robot()
        node.cmd_pub.publish.assert_called_once()


# ===========================================================================
# control_robot(distance, angle_rad)
# ===========================================================================

class TestControlRobotActuationDisabled:
    def test_does_not_publish_when_disabled(self, node):
        """Disabled mode: returns without publishing (no movement command sent)."""
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        node.cmd_pub.publish.assert_not_called()

    def test_sets_zero_last_values_when_disabled(self, node):
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        assert node.last_lin == 0.0
        assert node.last_ang == 0.0

    def test_sets_disabled_status(self, node):
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        assert node.last_status == "LOCKED (no actuation)"


class TestControlRobotTooClose:
    def test_zero_linear_when_below_min_distance(self, node):
        node.control_robot(0.3, 0.0)
        assert node.last_lin == 0.0

    def test_status_too_close_when_below_min(self, node):
        node.control_robot(0.3, 0.0)
        assert node.last_status == "TOO CLOSE - HOLDING"

    def test_zero_linear_when_distance_is_none(self, node):
        node.control_robot(None, 0.0)
        assert node.last_lin == 0.0

    def test_status_too_close_when_distance_is_none(self, node):
        node.control_robot(None, 0.0)
        assert node.last_status == "TOO CLOSE - HOLDING"

    def test_publishes_when_too_close(self, node):
        node.control_robot(0.3, 0.0)
        node.cmd_pub.publish.assert_called_once()


class TestControlRobotLinear:
    def test_forward_when_far(self, node):
        """distance > target_distance → positive linear."""
        node.control_robot(2.0, 0.0)
        assert node.last_lin > 0.0

    def test_linear_clamped_to_max(self, node):
        node.control_robot(100.0, 0.0)
        assert node.last_lin <= node.max_lin_speed

    def test_reverse_when_just_above_min_but_below_target(self, node):
        """distance between min_distance and target_distance → can reverse (negative)."""
        # e.g. min=0.6, target=1.0, distance=0.8 → error_d=-0.2 → raw=-0.14
        node.control_robot(0.8, 0.0)
        assert node.last_lin < 0.0

    def test_aligned_status_at_target(self, node):
        """At exactly target distance with zero angle → both commands ≈0 → ALIGNED."""
        node.control_robot(node.target_distance, 0.0)
        assert node.last_status == "ALIGNED - HOLDING"

    def test_following_status_when_far(self, node):
        node.control_robot(3.0, 0.0)
        assert node.last_status == "FOLLOWING TARGET"


class TestControlRobotAngular:
    def test_positive_angle_produces_positive_angular(self, node):
        node.control_robot(2.0, 0.5)
        assert node.last_ang > 0.0

    def test_negative_angle_produces_negative_angular(self, node):
        node.control_robot(2.0, -0.5)
        assert node.last_ang < 0.0

    def test_zero_angle_produces_zero_angular(self, node):
        node.control_robot(2.0, 0.0)
        assert node.last_ang == 0.0

    def test_angular_clamped_to_max(self, node):
        node.control_robot(2.0, math.pi)
        assert abs(node.last_ang) <= node.max_ang_speed

    def test_angular_clamped_negative_direction(self, node):
        node.control_robot(2.0, -math.pi)
        assert abs(node.last_ang) <= node.max_ang_speed

    def test_angular_proportional_to_angle_error(self, node):
        """Angular output = k_ang * angle_rad (before clamping)."""
        angle = 0.3   # small enough not to hit the clamp
        node.control_robot(2.0, angle)
        expected = node.k_ang * angle
        assert abs(node.last_ang - expected) < 1e-9

    def test_publishes_twist(self, node):
        node.control_robot(2.0, 0.3)
        node.cmd_pub.publish.assert_called_once()


# ===========================================================================
# search_for_target()
# ===========================================================================

class TestSearchForTargetActuationDisabled:
    def test_does_not_publish_rotation_when_disabled(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        # When disabled, method returns before publishing — no publish call
        node.cmd_pub.publish.assert_not_called()

    def test_sets_zero_last_values_when_disabled(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        assert node.last_lin == 0.0
        assert node.last_ang == 0.0

    def test_sets_disabled_status(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        assert node.last_status == "SEARCHING (disabled)"


class TestSearchForTargetLinearAlwaysZero:
    def test_linear_zero_during_normal_search(self, node):
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_lin == 0.0

    def test_linear_zero_when_search_complete(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        assert node.last_lin == 0.0


class TestSearchForTargetTurnDirection:
    def test_positive_angle_error_gives_positive_angular(self, node):
        node.last_angle_error = 0.3
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang > 0.0

    def test_negative_angle_error_gives_negative_angular(self, node):
        node.last_angle_error = -0.3
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang < 0.0

    def test_zero_angle_error_treated_as_positive(self, node):
        node.last_angle_error = 0.0
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang >= 0.0


class TestSearchForTargetFullTurn:
    def test_stops_after_360_degrees(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        assert node.last_ang == 0.0

    def test_status_after_360_degrees(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        assert "360" in node.last_status or "DONE" in node.last_status

    def test_publishes_zero_twist_after_360(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.angular.z == 0.0

    def test_still_rotating_just_before_360(self, node):
        node.search_angle_rad = 2.0 * math.pi - 0.01
        node.search_for_target()
        assert node.last_ang != 0.0

    def test_angle_accumulates_over_time(self, node):
        node.search_angle_rad = 0.0
        node._search_prev_t   = None
        node.search_for_target()   # initialises timer
        initial = node.search_angle_rad
        node._search_prev_t = time.time() - 1.0   # simulate 1s elapsed
        node.search_for_target()
        assert node.search_angle_rad > initial

    def test_searching_status_during_rotation(self, node):
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert "SEARCHING" in node.last_status


# ===========================================================================
# _bbox_from_4pts(bb2d)
# ===========================================================================

class TestBboxFrom4Pts:
    def test_basic_rectangle(self, node):
        bb2d = [[100, 50], [300, 50], [300, 400], [100, 400]]
        x1, y1, x2, y2 = node._bbox_from_4pts(bb2d)
        assert x1 == 100
        assert y1 == 50
        assert x2 == 300
        assert y2 == 400

    def test_unordered_points(self, node):
        """Points in any order — should still give correct min/max."""
        bb2d = [[300, 400], [100, 50], [300, 50], [100, 400]]
        x1, y1, x2, y2 = node._bbox_from_4pts(bb2d)
        assert x1 == 100
        assert y1 == 50
        assert x2 == 300
        assert y2 == 400

    def test_returns_integers(self, node):
        bb2d = [[100.7, 50.3], [300.1, 400.9]]
        x1, y1, x2, y2 = node._bbox_from_4pts(bb2d)
        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)

    def test_negative_coords_clipped_to_zero(self, node):
        """Negative coordinates are clipped to 0."""
        bb2d = [[-50, -20], [200, 300]]
        x1, y1, x2, y2 = node._bbox_from_4pts(bb2d)
        assert x1 == 0
        assert y1 == 0


# ===========================================================================
# _shirt_color_stats(mask_color, poly_pts, valid_mask=None)
# Returns: (ratio, total_col, left_col, right_col, span_frac, denom_area)
# ===========================================================================

IMG_H = 720
IMG_W = 1280


def _make_rect_poly(x1, y1, x2, y2):
    """Helper: rectangular polygon as (N,2) int32 array (clockwise)."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


class TestShirtColorStatsBasic:
    def test_all_white_mask_returns_ratio_one(self, node):
        mask  = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly  = _make_rect_poly(100, 100, 300, 500)
        ratio, total, left, right, span, area = node._shirt_color_stats(mask, poly)
        assert abs(ratio - 1.0) < 1e-6
        assert total > 0
        assert area  > 0

    def test_all_black_mask_returns_ratio_zero(self, node):
        mask  = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        poly  = _make_rect_poly(100, 100, 300, 500)
        ratio, total, left, right, span, area = node._shirt_color_stats(mask, poly)
        assert ratio == 0.0
        assert total == 0

    def test_ratio_in_range_zero_to_one(self, node):
        mask = np.random.randint(0, 256, (IMG_H, IMG_W), dtype=np.uint8)
        poly = _make_rect_poly(50, 50, 400, 600)
        ratio, *_ = node._shirt_color_stats(mask, poly)
        assert 0.0 <= ratio <= 1.0

    def test_none_poly_returns_zeros(self, node):
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        result = node._shirt_color_stats(mask, None)
        assert result == (0.0, 0, 0, 0, 0.0, 0)

    def test_none_mask_returns_zeros(self, node):
        poly = _make_rect_poly(100, 100, 300, 500)
        result = node._shirt_color_stats(None, poly)
        assert result == (0.0, 0, 0, 0, 0.0, 0)


class TestShirtColorStatsLeftRight:
    def test_left_half_white_detected_in_left_col(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        # Poly from x=100 to x=300 (width 200); left half = 100–200
        poly = _make_rect_poly(100, 100, 300, 400)
        mask[100:400, 100:200] = 255   # only left half of polygon is white
        _, total, left, right, _, _ = node._shirt_color_stats(mask, poly)
        assert left > 0
        assert right == 0

    def test_right_half_white_detected_in_right_col(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        mask[100:400, 200:300] = 255   # only right half of polygon is white
        _, total, left, right, _, _ = node._shirt_color_stats(mask, poly)
        assert right > 0
        assert left == 0

    def test_both_halves_detected_when_full_mask(self, node):
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        _, _, left, right, _, _ = node._shirt_color_stats(mask, poly)
        assert left > 0
        assert right > 0


class TestShirtColorStatsVerticalSpan:
    def test_full_height_coverage_gives_span_near_one(self, node):
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        _, _, _, _, span_frac, _ = node._shirt_color_stats(mask, poly)
        assert span_frac > 0.9

    def test_single_row_coverage_gives_small_span(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        mask[200, 100:300] = 255   # single row
        _, _, _, _, span_frac, _ = node._shirt_color_stats(mask, poly)
        assert span_frac < 0.1   # 1 row out of 300 rows = ~0.003

    def test_no_colored_pixels_gives_zero_span(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        _, _, _, _, span_frac, _ = node._shirt_color_stats(mask, poly)
        assert span_frac == 0.0


class TestShirtColorStatsValidMask:
    def test_valid_mask_restricts_denominator(self, node):
        """With valid_mask covering only half the polygon, denom_area should be ~half."""
        mask_color = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly       = _make_rect_poly(100, 100, 300, 400)

        # valid_mask: only left half of image is "person"
        valid = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        valid[:, :200] = 255   # left half: covers polygon x=100–200

        _, _, _, _, _, denom_full = node._shirt_color_stats(mask_color, poly)
        _, _, _, _, _, denom_half = node._shirt_color_stats(mask_color, poly, valid_mask=valid)

        assert denom_half < denom_full

    def test_valid_mask_none_same_as_no_mask(self, node):
        mask  = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly  = _make_rect_poly(100, 100, 300, 400)
        r1, *_ = node._shirt_color_stats(mask, poly, valid_mask=None)
        r2, *_ = node._shirt_color_stats(mask, poly)
        assert r1 == r2

    def test_valid_mask_entirely_outside_poly_returns_zeros(self, node):
        """valid_mask covers a region that doesn't overlap the polygon → denom=0."""
        mask_color = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly       = _make_rect_poly(100, 100, 300, 400)
        valid      = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        valid[500:700, 500:700] = 255   # no overlap with poly
        result = node._shirt_color_stats(mask_color, poly, valid_mask=valid)
        assert result == (0.0, 0, 0, 0, 0.0, 0)


# ===========================================================================
# _color_ratio_polygon_mask(mask_color, poly_pts)
# ===========================================================================

class TestColorRatioPolygonMask:
    def test_all_white_returns_one(self, node):
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        poly = _make_rect_poly(100, 100, 300, 400)
        ratio = node._color_ratio_polygon_mask(mask, poly)
        assert abs(ratio - 1.0) < 1e-6

    def test_all_black_returns_zero(self, node):
        mask  = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        poly  = _make_rect_poly(100, 100, 300, 400)
        ratio = node._color_ratio_polygon_mask(mask, poly)
        assert ratio == 0.0

    def test_none_poly_returns_zero(self, node):
        mask  = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        ratio = node._color_ratio_polygon_mask(mask, None)
        assert ratio == 0.0

    def test_ratio_in_range_zero_to_one(self, node):
        mask  = np.random.randint(0, 256, (IMG_H, IMG_W), dtype=np.uint8)
        poly  = _make_rect_poly(50, 50, 400, 600)
        ratio = node._color_ratio_polygon_mask(mask, poly)
        assert 0.0 <= ratio <= 1.0
