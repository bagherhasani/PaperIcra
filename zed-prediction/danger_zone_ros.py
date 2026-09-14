#!/usr/bin/env python3
"""
Live RViz + Nav2 local costmap hook.

- MarkerArray /person/danger_zone  : blue dot = now, red arrow = +2 s
- PointCloud2 /person/predicted_cloud : disc at +2 s (Nav2 obstacle)

Frame is base_link so Nav2 TF (map/odom → base_link) places it on the real map.
ZED EKF: px = forward, py = lateral  →  ROS: x = px, y = py
"""

import math
import struct

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from test_config import ROBOT_MOVE


FRAME = "base_link"  # Nav2 robot frame (shows on map via TF)
RADIUS = 0.40        # person footprint radius (m)


class DangerZonePublisher:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("danger_zone_live")
        self.pub_markers = self.node.create_publisher(
            MarkerArray, "/person/danger_zone", 10
        )
        self.pub_cloud = self.node.create_publisher(
            PointCloud2, "/person/predicted_cloud", 10
        )
        self.pub_cmd = self.node.create_publisher(Twist, "/cmd_vel", 10)
        if not ROBOT_MOVE:
            self._hold_timer = self.node.create_timer(0.05, self._hold_still)
            self.node.get_logger().warn(
                "ROBOT_MOVE=False — publishing zero /cmd_vel (robot will not drive)"
            )
        self.node.get_logger().info(
            f"RViz: /person/danger_zone  |  Nav2 cloud: /person/predicted_cloud  |  "
            f"ROBOT_MOVE={ROBOT_MOVE}"
        )

    def _hold_still(self):
        self.pub_cmd.publish(Twist())

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
        m.scale.x = m.scale.y = m.scale.z = 0.28
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
        m.scale.x = 0.14
        m.scale.y = 0.32
        m.scale.z = 0.40
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.05, 0.05, 1.0
        m.points = [
            Point(x=float(x0), y=float(y0), z=0.30),
            Point(x=float(x1), y=float(y1), z=0.30),
        ]
        m.lifetime.sec = 1
        return m

    def _disc_points(self, cx, cy, radius=RADIUS, n_ring=12, n_rad=4):
        pts = [(cx, cy, 0.15)]
        for i in range(1, n_rad + 1):
            r = radius * i / n_rad
            for k in range(n_ring):
                a = 2.0 * math.pi * k / n_ring
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a), 0.15))
        return pts

    def _corridor_points(self, x0, y0, x1, y1, steps=8):
        pts = []
        for i in range(1, steps + 1):
            t = i / float(steps)
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

    def publish(self, px, py, future_px, future_py):
        arr = MarkerArray()
        arr.markers.append(self._delete("now", 0))
        arr.markers.append(self._delete("predicted_1s", 1))
        arr.markers.append(self._delete("intent", 2))
        arr.markers.append(self._dot(px, py))
        arr.markers.append(self._arrow(px, py, future_px, future_py))
        self.pub_markers.publish(arr)

        pts = self._corridor_points(px, py, future_px, future_py)
        pts.extend(self._disc_points(future_px, future_py))
        self.pub_cloud.publish(self._cloud(pts))

        rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
