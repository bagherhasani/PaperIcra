#!/usr/bin/env python3
"""
RViz markers + Nav2 local-costmap person cloud (Block 2).

Frame contract:
  Inputs are EKF camera-horizontal coords: +px forward (cam Z), +py cam-right (cam X).
  Published ROS base_link: x=px, y=-py  (+y = left).

NAV_CLOUD_MODE:
  "now"  → disc at person now (Mode A; lidar also sees body — expect overlap)
  "pred" → disc at +1 s only (Mode B). No disc on feet. If speed < gate → now disc only.
"""

from __future__ import annotations

import math
import struct

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    from test_config import NAV_CLOUD_MODE, PRED_MIN_SPEED
except ImportError:
    NAV_CLOUD_MODE = "pred"
    PRED_MIN_SPEED = 0.30

try:
    from test_config import CLOUD_EMA_ALPHA, PRED_SPEED_OFF
except ImportError:
    CLOUD_EMA_ALPHA = 0.35
    PRED_SPEED_OFF = 0.20

FRAME = "base_link"
RADIUS = 0.22


def ekf_to_base(px, py):
    """EKF (+py = camera right) → ROS base_link (+y = left)."""
    return float(px), float(-py)


class DangerZonePublisher:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("danger_zone_live")
        self.pub_markers = self.node.create_publisher(
            MarkerArray, "/person/danger_zone", 10
        )
        self.pub_nav = self.node.create_publisher(PointCloud2, "/person/nav_cloud", 10)
        self.pub_pred = self.node.create_publisher(
            PointCloud2, "/person/predicted_cloud", 10
        )
        self.mode = str(NAV_CLOUD_MODE).lower().strip()
        if self.mode not in ("pred", "now"):
            self.mode = "pred"
        self.pred_on = float(PRED_MIN_SPEED)
        self.pred_off = float(PRED_SPEED_OFF)
        self.ema_a = float(CLOUD_EMA_ALPHA)
        self._moving_pred = False  # hysteresis for corridor vs now
        self._ema_now = None
        self._ema_pred = None
        self.node.get_logger().info(
            f"RViz=/person/danger_zone | mode={self.mode} | "
            f"pred_gate=[{self.pred_off},{self.pred_on}] | frame={FRAME} (y flipped)"
        )

    def reset_smooth(self):
        self._ema_now = None
        self._ema_pred = None
        self._ema_arrow = None
        self._moving_pred = False

    def _header(self):
        h = Header()
        h.stamp = self.node.get_clock().now().to_msg()
        h.frame_id = FRAME
        return h

    def _delete(self, ns, mid):
        m = Marker()
        m.header = self._header()
        m.ns = ns
        m.id = mid
        m.action = Marker.DELETE
        return m

    def _dot(self, x, y):
        m = Marker()
        m.header = self._header()
        m.ns = "now_dot"
        m.id = 10
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.30
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.22
        m.color.r, m.color.g, m.color.b, m.color.a = 0.15, 0.45, 1.0, 1.0
        m.lifetime.sec = 1
        return m

    def _arrow(self, x0, y0, x1, y1):
        m = Marker()
        m.header = self._header()
        m.ns = "pred_2s"
        m.id = 11
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.12
        m.scale.y = 0.28
        m.scale.z = 0.36
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.05, 0.05, 1.0
        m.points = [
            Point(x=float(x0), y=float(y0), z=0.30),
            Point(x=float(x1), y=float(y1), z=0.30),
        ]
        m.lifetime.sec = 1
        return m

    def _disc_points(self, cx, cy, radius=RADIUS, n_ring=10, n_rad=3):
        pts = [(cx, cy, 0.15)]
        for i in range(1, n_rad + 1):
            r = radius * i / n_rad
            for k in range(n_ring):
                a = 2.0 * math.pi * k / n_ring
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a), 0.15))
        return pts

    def _corridor_tail(self, x0, y0, x1, y1, steps=4):
        """Only the far half of now→pred (Mode B), avoids painting feet."""
        pts = []
        for i in range(1, steps + 1):
            t = 0.5 + 0.5 * (i / float(steps))
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.15))
        return pts

    def _cloud(self, points):
        msg = PointCloud2()
        msg.header = self._header()
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        buf = bytearray()
        for x, y, z in points:
            buf.extend(struct.pack("fff", float(x), float(y), float(z)))
        msg.data = bytes(buf)
        return msg

    def _ema(self, state, xy):
        if state is None:
            return [xy[0], xy[1]]
        a = self.ema_a
        state[0] = a * xy[0] + (1.0 - a) * state[0]
        state[1] = a * xy[1] + (1.0 - a) * state[1]
        return state

    def publish(self, px, py, future_px, future_py, pred_1s_px=None, pred_1s_py=None,
                speed=0.0):
        """
        px,py = now (EKF); future_* = +2 s RViz arrow; pred_1s_* = +1 s Nav cloud.
        speed = EKF speed for standstill gate.
        """
        if pred_1s_px is None:
            pred_1s_px = future_px
        if pred_1s_py is None:
            pred_1s_py = future_py

        # Hysteresis: avoid corridor flicker at the speed threshold
        if speed >= self.pred_on:
            self._moving_pred = True
        elif speed <= self.pred_off:
            self._moving_pred = False

        # ROS base_link
        nx, ny = ekf_to_base(px, py)
        ax, ay = ekf_to_base(future_px, future_py)
        p1x, p1y = ekf_to_base(pred_1s_px, pred_1s_py)

        self._ema_now = self._ema(self._ema_now, (nx, ny))
        self._ema_pred = self._ema(self._ema_pred, (p1x, p1y))
        nx, ny = self._ema_now
        p1x, p1y = self._ema_pred
        # RViz +2s arrow: EMA tip relative to smoothed now
        self._ema_arrow = self._ema(getattr(self, "_ema_arrow", None), (ax, ay))
        ax, ay = self._ema_arrow

        arr = MarkerArray()
        arr.markers.append(self._delete("now", 0))
        arr.markers.append(self._delete("predicted_1s", 1))
        arr.markers.append(self._delete("intent", 2))
        arr.markers.append(self._dot(nx, ny))
        arr.markers.append(self._arrow(nx, ny, ax, ay))
        self.pub_markers.publish(arr)

        use_pred_cloud = (self.mode == "pred" and self._moving_pred)
        if use_pred_cloud:
            # Mode B walking: future disc only (+ short tail), NOT feet
            pts = self._corridor_tail(nx, ny, p1x, p1y)
            pts.extend(self._disc_points(p1x, p1y))
        else:
            # Mode A, or Mode B while standing: now disc only
            pts = self._disc_points(nx, ny)

        cloud = self._cloud(pts)
        self.pub_nav.publish(cloud)
        self.pub_pred.publish(cloud)

        rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
