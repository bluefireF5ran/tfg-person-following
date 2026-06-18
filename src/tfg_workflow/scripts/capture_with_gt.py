#!/usr/bin/env python3
"""Capture images from TurtleBot4 camera + auto-generate GT bounding boxes.

Ground truth is computed by:
1. Parsing actor trajectories from the SDF world file
2. At each frame, interpolating actor positions from sim time
3. Reading camera intrinsics from /camera_info topic (runtime)
4. Looking up base_link → camera optical transform from TF2
5. Projecting actor 3D positions to 2D image plane
6. Generating PASCAL VOC XML annotations

The robot can be teleported to specified waypoints via Gazebo transport.

Usage (with simulation running):
  python3 capture_with_gt.py \\
    --sdf ../worlds/warehouse_actors.sdf \\
    --output_dir ../data/sim_dataset \\
    --rate 2.0 --max_images 200

  # With explicit robot waypoints:
  python3 capture_with_gt.py \\
    --sdf ../worlds/warehouse_actors.sdf \\
    --output_dir ../data/sim_dataset \\
    --waypoints "-2,2,0  2,3,90  0,5,180" \\
    --images_per_wp 50
"""
import argparse
import math
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from gz.msgs10 import boolean_pb2, pose_pb2
from gz.transport13 import Node as GzNode

# ── Encoding → (dtype, channels) for cv_bridge bypass ───────────
_ENC = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
}


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    dt, ch = _ENC[msg.encoding]
    arr = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


# ═══════════════════════════════════════════════════════════════════
# SDF trajectory parser
# ═══════════════════════════════════════════════════════════════════

class ActorTrajectory:
    """Holds one actor's scripted trajectory and interpolates poses."""

    def __init__(self, name: str, delay: float, waypoints: list):
        """waypoints: list of (time, x, y, z, yaw) tuples."""
        self.name = name
        self.delay = delay
        self.wps = waypoints  # sorted by time
        self.duration = waypoints[-1][0] if waypoints else 0.0

    def pose_at(self, sim_time: float):
        """Interpolate (x, y, z) at the given simulation time."""
        t = sim_time - self.delay
        if t < 0 or self.duration <= 0:
            wp = self.wps[0]
            return wp[1], wp[2], wp[3]

        t_cycle = t % self.duration
        # Find the two bracketing waypoints
        for i in range(len(self.wps) - 1):
            t0, x0, y0, z0, _ = self.wps[i]
            t1, x1, y1, z1, _ = self.wps[i + 1]
            if t0 <= t_cycle <= t1:
                frac = (t_cycle - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                return (x0 + frac * (x1 - x0),
                        y0 + frac * (y1 - y0),
                        z0 + frac * (z1 - z0))

        wp = self.wps[-1]
        return wp[1], wp[2], wp[3]


def parse_actors_from_sdf(sdf_path: str) -> list[ActorTrajectory]:
    """Parse <actor> elements from an SDF world file."""
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    actors = []
    for actor_el in root.iter("actor"):
        name = actor_el.get("name", "unknown")
        script = actor_el.find("script")
        if script is None:
            continue
        delay_el = script.find("delay_start")
        delay = float(delay_el.text) if delay_el is not None else 0.0

        wps = []
        for wp_el in script.iter("waypoint"):
            t = float(wp_el.find("time").text)
            pose_text = wp_el.find("pose").text.split()
            x, y, z = float(pose_text[0]), float(pose_text[1]), float(pose_text[2])
            yaw = float(pose_text[5]) if len(pose_text) > 5 else 0.0
            wps.append((t, x, y, z, yaw))
        wps.sort(key=lambda w: w[0])
        if wps:
            actors.append(ActorTrajectory(name, delay, wps))
    return actors


# ═══════════════════════════════════════════════════════════════════
# 3D → 2D projection
# ═══════════════════════════════════════════════════════════════════

# Actor 3D bounding box in local frame (relative to trajectory z):
#   The Mingfei walk.dae mesh origin is near the hips (~1.0m above feet).
#   In the SDF, trajectory z=1.0 so feet ≈ z=0.0 and head ≈ z=1.77.
#   We define the bounding box extent from the trajectory origin.
ACTOR_BOTTOM_OFFSET = 1.0    # distance from trajectory z to feet
ACTOR_TOP_OFFSET    = 0.77   # distance from trajectory z to head
ACTOR_HALF_WIDTH    = 0.25   # half shoulder width


def compute_bbox(actor_xyz, robot_xyz, robot_yaw,
                 cam_x, cam_y, cam_z, fx, fy, cx, cy, img_w, img_h,
                 min_bbox_pixels=10, max_distance=12.0):
    """Compute 2D bbox from actor world position + robot pose + camera calibration.

    Returns (xmin, ymin, xmax, ymax) in pixels or None if not visible.
    """
    ax, ay, az = actor_xyz
    rx, ry, rz = robot_xyz

    # Distance check
    dist = math.hypot(ax - rx, ay - ry)
    if dist > max_distance or dist < 0.3:
        return None

    # Person 3D bounding box corners in world frame
    feet_z = az - ACTOR_BOTTOM_OFFSET
    head_z = az + ACTOR_TOP_OFFSET
    half_w = ACTOR_HALF_WIDTH

    corners = [
        (ax - half_w, ay - half_w, feet_z),
        (ax + half_w, ay - half_w, feet_z),
        (ax - half_w, ay + half_w, feet_z),
        (ax + half_w, ay + half_w, feet_z),
        (ax - half_w, ay - half_w, head_z),
        (ax + half_w, ay - half_w, head_z),
        (ax - half_w, ay + half_w, head_z),
        (ax + half_w, ay + half_w, head_z),
    ]

    cos_neg_yaw = math.cos(-robot_yaw)
    sin_neg_yaw = math.sin(-robot_yaw)

    us, vs = [], []
    for (wx, wy, wz) in corners:
        # World → robot base_link
        dx = wx - rx
        dy = wy - ry
        dz = wz - rz
        x_r = cos_neg_yaw * dx - sin_neg_yaw * dy
        y_r = sin_neg_yaw * dx + cos_neg_yaw * dy
        z_r = dz

        # Robot base → camera base (apply offset)
        x_c = x_r - cam_x
        y_c = y_r - cam_y
        z_c = z_r - cam_z

        # Camera base (X-fwd, Y-left, Z-up) → optical (X-right, Y-down, Z-fwd)
        Xo = -y_c
        Yo = -z_c
        Zo = x_c

        if Zo <= 0.1:
            continue

        u = fx * Xo / Zo + cx
        v = fy * Yo / Zo + cy
        us.append(u)
        vs.append(v)

    if len(us) < 2:
        return None

    x1, y1 = min(us), min(vs)
    x2, y2 = max(us), max(vs)

    # Clip to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if (x2 - x1) < min_bbox_pixels or (y2 - y1) < min_bbox_pixels:
        return None

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


# ═══════════════════════════════════════════════════════════════════
# PASCAL VOC XML writer
# ═══════════════════════════════════════════════════════════════════

def write_voc_xml(xml_path: Path, img_filename: str, width: int, height: int,
                  bboxes: list[tuple]):
    """Write PASCAL VOC annotation XML matching CITIC5 format."""
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "images"
    ET.SubElement(ann, "filename").text = img_filename

    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "SimDataset"

    sz = ET.SubElement(ann, "size")
    ET.SubElement(sz, "width").text = str(width)
    ET.SubElement(sz, "height").text = str(height)
    ET.SubElement(sz, "depth").text = "3"

    ET.SubElement(ann, "segmented").text = "0"

    for (x1, y1, x2, y2) in bboxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = "person"
        ET.SubElement(obj, "pose").text = "Unspecified"
        truncated = "1" if (x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height) else "0"
        ET.SubElement(obj, "truncated").text = truncated
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)

    tree = ET.ElementTree(ann)
    ET.indent(tree, space="\t")
    tree.write(str(xml_path), encoding="unicode", xml_declaration=False)


# ═══════════════════════════════════════════════════════════════════
# Gazebo robot teleportation
# ═══════════════════════════════════════════════════════════════════

def teleport_robot(gz_node: GzNode, world: str, model_name: str,
                   x: float, y: float, yaw: float, z: float = 0.0):
    """Teleport the robot to (x, y, z, yaw) via Gazebo set_pose service."""
    service = f"/world/{world}/set_pose"
    pose = pose_pb2.Pose()
    pose.name = model_name
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(yaw / 2)
    pose.orientation.w = math.cos(yaw / 2)

    ok, _ = gz_node.request(
        service, pose, pose_pb2.Pose, boolean_pb2.Boolean, timeout=3000
    )
    return ok


# ═══════════════════════════════════════════════════════════════════
# Main ROS 2 capture node
# ═══════════════════════════════════════════════════════════════════

class CaptureGTNode(Node):
    def __init__(self, args):
        super().__init__("capture_with_gt")
        self.args = args
        self.output_images = Path(args.output_dir) / "images"
        self.output_labels = Path(args.output_dir) / "labels"
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

        # Parse actor trajectories from SDF
        self.actors = parse_actors_from_sdf(args.sdf)
        self.get_logger().info(f"Parsed {len(self.actors)} actors from {args.sdf}")
        for a in self.actors:
            self.get_logger().info(f"  {a.name}: {len(a.wps)} waypoints, delay={a.delay}s")

        # Camera calibration (from /camera_info, filled at runtime)
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = 0
        self._cam_lock = threading.Lock()

        # Camera extrinsics (rough defaults from TB4 Lite URDF)
        # Will be can be overridden by --cam_x, --cam_z arguments
        self.cam_x = args.cam_x    # forward offset from base_link
        self.cam_y = 0.0           # lateral (centered)
        self.cam_z = args.cam_z    # height above base_link

        # Capture state
        self.counter = args.start_index
        self.rate_interval = 1.0 / max(args.rate, 0.01)
        self.last_capture_time = 0.0
        self._done = False

        # Current robot pose (set by us via teleportation)
        self.robot_x = args.start_x
        self.robot_y = args.start_y
        self.robot_yaw = math.radians(args.start_yaw)

        # Waypoint management
        self.waypoints = self._parse_waypoints(args.waypoints)
        self.current_wp_idx = 0
        self.images_this_wp = 0
        self.images_per_wp = args.images_per_wp
        self.settle_until = 0.0  # don't capture until this wall time

        # Subscribe to camera
        self.create_subscription(
            CameraInfo,
            f"{args.camera_ns}/camera_info",
            self._on_camera_info,
            10,
        )
        self.create_subscription(
            Image,
            f"{args.camera_ns}/image_raw",
            self._on_image,
            qos_profile_sensor_data,
        )

        # Gazebo transport node (for teleportation)
        self.gz_node = GzNode()

        # If waypoints provided, teleport to first one
        if self.waypoints:
            self._teleport_to_waypoint(0)

        self.get_logger().info("Capture node ready. Waiting for camera…")

    def _parse_waypoints(self, wp_str):
        """Parse waypoints from 'x,y,yaw_deg  x,y,yaw_deg ...' string."""
        if not wp_str:
            return []
        result = []
        for token in wp_str.strip().split():
            parts = token.split(",")
            x, y, yaw_deg = float(parts[0]), float(parts[1]), float(parts[2])
            result.append((x, y, math.radians(yaw_deg)))
        return result

    def _teleport_to_waypoint(self, idx):
        """Teleport robot to waypoint[idx] and update internal state."""
        if idx >= len(self.waypoints):
            return
        x, y, yaw = self.waypoints[idx]
        self.get_logger().info(
            f"Teleporting to waypoint {idx}: ({x:.1f}, {y:.1f}, {math.degrees(yaw):.0f}°)"
        )
        ok = teleport_robot(
            self.gz_node, self.args.world, self.args.robot_model,
            x, y, yaw
        )
        if not ok:
            self.get_logger().warn(f"Teleport failed for waypoint {idx}")

        self.robot_x, self.robot_y, self.robot_yaw = x, y, yaw
        self.current_wp_idx = idx
        self.images_this_wp = 0
        self.settle_until = time.monotonic() + self.args.settle_time

    def _on_camera_info(self, msg: CameraInfo):
        with self._cam_lock:
            if self.fx is not None:
                return  # already calibrated
            self.img_w = msg.width
            self.img_h = msg.height
            K = msg.k
            self.fx = K[0]
            self.fy = K[4]
            self.cx = K[2]
            self.cy = K[5]
            self.get_logger().info(
                f"Camera intrinsics: {self.img_w}×{self.img_h}, "
                f"fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}"
            )

    def _on_image(self, msg: Image):
        if self._done:
            return

        with self._cam_lock:
            if self.fx is None:
                return  # no calibration yet
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
            img_w, img_h = self.img_w, self.img_h

        # Rate limiting
        now = time.monotonic()
        if now - self.last_capture_time < self.rate_interval:
            return
        # Wait for settle after teleport
        if now < self.settle_until:
            return
        self.last_capture_time = now

        # Get simulation time from message header
        sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Convert image
        frame = imgmsg_to_cv2(msg)

        # Compute GT bounding boxes for each actor
        bboxes = []
        robot_xyz = (self.robot_x, self.robot_y, 0.0)
        for actor in self.actors:
            ax, ay, az = actor.pose_at(sim_time)
            bbox = compute_bbox(
                (ax, ay, az), robot_xyz, self.robot_yaw,
                self.cam_x, self.cam_y, self.cam_z,
                fx, fy, cx, cy, img_w, img_h,
            )
            if bbox is not None:
                bboxes.append(bbox)

        # Save image + annotation
        img_name = f"sim_{self.counter:06d}.jpg"
        xml_name = f"sim_{self.counter:06d}.xml"
        cv2.imwrite(str(self.output_images / img_name), frame)
        write_voc_xml(
            self.output_labels / xml_name,
            img_name, img_w, img_h, bboxes,
        )

        self.counter += 1
        self.images_this_wp += 1

        if self.counter % 20 == 0:
            self.get_logger().info(
                f"Saved {self.counter} images "
                f"(wp {self.current_wp_idx}, {len(bboxes)} persons in frame)"
            )

        # Advance to next waypoint if needed
        if self.waypoints and self.images_this_wp >= self.images_per_wp:
            next_idx = self.current_wp_idx + 1
            if next_idx < len(self.waypoints):
                self._teleport_to_waypoint(next_idx)
            else:
                self.get_logger().info("All waypoints completed.")
                self._done = True
                raise SystemExit(0)

        # Max images check
        max_imgs = self.args.max_images
        if max_imgs > 0 and self.counter >= (self.args.start_index + max_imgs):
            self.get_logger().info(f"Target reached ({max_imgs} images). Done.")
            self._done = True
            raise SystemExit(0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture simulation images with automatic GT bounding boxes"
    )
    # Paths
    p.add_argument("--sdf", required=True, help="Path to world SDF with <actor> elements")
    p.add_argument("--output_dir", required=True, help="Output directory (images/ + labels/)")

    # Capture settings
    p.add_argument("--rate", type=float, default=2.0, help="Capture rate Hz (default 2)")
    p.add_argument("--max_images", type=int, default=0, help="Max images total (0=unlimited)")
    p.add_argument("--start_index", type=int, default=0, help="Starting image counter")

    # Robot waypoints
    p.add_argument("--waypoints", type=str, default="",
                   help='Robot waypoints: "x,y,yaw_deg  x,y,yaw_deg ..."')
    p.add_argument("--images_per_wp", type=int, default=50,
                   help="Images to capture at each waypoint (default 50)")
    p.add_argument("--settle_time", type=float, default=2.0,
                   help="Seconds to wait after teleport before capturing")

    # Initial robot position (if no waypoints)
    p.add_argument("--start_x", type=float, default=-2.0)
    p.add_argument("--start_y", type=float, default=2.0)
    p.add_argument("--start_yaw", type=float, default=0.0, help="Degrees")

    # Gazebo
    p.add_argument("--world", default="warehouse_actors", help="Gazebo world name")
    p.add_argument("--robot_model", default="turtlebot4", help="Robot model name in Gazebo")

    # Camera settings
    p.add_argument("--camera_ns", default="/oakd/rgb/preview",
                   help="Camera topic namespace")
    p.add_argument("--cam_x", type=float, default=-0.037,
                   help="Camera X offset from base_link (m, forward)")
    p.add_argument("--cam_z", type=float, default=0.19,
                   help="Camera Z offset from base_link (m, upward)")

    return p.parse_args(rclpy.utilities.remove_ros_args()[1:])


def main():
    rclpy.init()
    args = parse_args()
    node = CaptureGTNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n = node.counter - args.start_index
        print(f"\nCapture complete: {n} images saved to {args.output_dir}")
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
