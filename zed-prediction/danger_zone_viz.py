#!/usr/bin/env python3
"""
Minimal proof: extend the "danger zone" in RViz.

Publishes visualization_msgs/MarkerArray on /person/danger_zone:
  - BLUE  = person NOW
  - RED   = predicted +1 s (danger zone ahead)
  - YELLOW arrow = now → predicted

Also publishes a static TF for frame "map" so empty RViz can display markers.

Run:
  source /opt/ros/humble/setup.bash
  python3 danger_zone_viz.py

RViz2:
  Fixed Frame = map
  Add → MarkerArray → Topic = /person/danger_zone
  Click "Reset" or set viewpoint near (0,0,0) — markers stay near origin.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster


FRAME = "map"
HZ = 10.0
SECONDS_AHEAD = 1.0
RADIUS = 0.40
HEIGHT = 1.6
DEMO = True


class DangerZoneViz(Node):
    def __init__(self):
        super().__init__("danger_zone_viz")
        self.pub = self.create_publisher(MarkerArray, "/person/danger_zone", 10)

        # So RViz Fixed Frame "map" always exists (no Nav2 required)
        self.tf_static = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "danger_zone_origin"
        t.transform.rotation.w = 1.0
        self.tf_static.sendTransform(t)

        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / HZ, self.tick)
        self._tick_n = 0
        self.get_logger().info(
            "Publishing /person/danger_zone  |  Fixed Frame = map  |  "
            "markers stay near origin (demo walk loops)"
        )

    def demo_pose(self):
        """Walk back and forth near origin so RViz always sees you."""
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        # Loop every ~12 s between x=0 and x=4
        period = 12.0
        phase = (t % period) / period
        if phase < 0.5:
            u = phase * 2.0
            px = 0.5 + 3.5 * u
            speed = 3.5 / (period * 0.5)
            heading = 0.0
        else:
            u = (phase - 0.5) * 2.0
            px = 4.0 - 3.5 * u
            speed = 3.5 / (period * 0.5)
            heading = math.pi
        py = 0.4 * math.sin(2.0 * math.pi * t / period)
        return px, py, speed, heading

    def predict_1s(self, px, py, speed, heading):
        dt = SECONDS_AHEAD
        return (
            px + speed * math.cos(heading) * dt,
            py + speed * math.sin(heading) * dt,
        )

    def cylinder(self, x, y, r, g, b, ns, mid_id):
        m = Marker()
        m.header = Header()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = FRAME
        m.ns = ns
        m.id = mid_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = HEIGHT / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = 2.0 * RADIUS
        m.scale.y = 2.0 * RADIUS
        m.scale.z = HEIGHT
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 0.55
        # Keep visible; refreshed every tick
        m.lifetime.sec = 1
        return m

    def arrow(self, x0, y0, x1, y1):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = FRAME
        m.ns = "intent"
        m.id = 2
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.10
        m.scale.y = 0.18
        m.scale.z = 0.25
        m.color.r = 1.0
        m.color.g = 0.85
        m.color.b = 0.0
        m.color.a = 1.0
        m.points = [
            Point(x=float(x0), y=float(y0), z=0.25),
            Point(x=float(x1), y=float(y1), z=0.25),
        ]
        m.lifetime.sec = 1
        return m

    def tick(self):
        if DEMO:
            px, py, speed, heading = self.demo_pose()
        else:
            px, py, speed, heading = 0.0, 0.0, 0.0, 0.0

        fx, fy = self.predict_1s(px, py, speed, heading)

        arr = MarkerArray()
        arr.markers.append(self.cylinder(px, py, 0.15, 0.35, 1.0, "now", 0))
        arr.markers.append(self.cylinder(fx, fy, 1.0, 0.05, 0.05, "predicted_1s", 1))
        arr.markers.append(self.arrow(px, py, fx, fy))
        self.pub.publish(arr)

        self._tick_n += 1
        if self._tick_n % 20 == 1:
            self.get_logger().info(
                f"NOW=({px:.2f},{py:.2f})  PRED_1s=({fx:.2f},{fy:.2f})  "
                f"— look near origin in RViz"
            )


def main():
    rclpy.init()
    node = DangerZoneViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
