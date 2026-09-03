#!/usr/bin/env python3
"""
Live RViz: blue dot = person now, red arrow = EKF predicted +2 seconds.

EKF plane: px = forward, py = lateral  →  RViz map: x = px, y = py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import StaticTransformBroadcaster


FRAME = "map"


class DangerZonePublisher:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("danger_zone_live")
        self.pub = self.node.create_publisher(MarkerArray, "/person/danger_zone", 10)

        self.tf_static = StaticTransformBroadcaster(self.node)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "danger_zone_origin"
        t.transform.rotation.w = 1.0
        self.tf_static.sendTransform(t)

        self.node.get_logger().info(
            "Live arrows → /person/danger_zone  |  Fixed Frame=map  |  red arrow = +2 s"
        )

    def _delete(self, ns, mid):
        m = Marker()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = FRAME
        m.ns = ns
        m.id = mid
        m.action = Marker.DELETE
        return m

    def _dot(self, x, y):
        m = Marker()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = FRAME
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
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = FRAME
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

    def publish(self, px, py, future_px, future_py):
        arr = MarkerArray()
        arr.markers.append(self._delete("now", 0))
        arr.markers.append(self._delete("predicted_1s", 1))
        arr.markers.append(self._delete("intent", 2))
        arr.markers.append(self._dot(px, py))
        arr.markers.append(self._arrow(px, py, future_px, future_py))
        self.pub.publish(arr)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
