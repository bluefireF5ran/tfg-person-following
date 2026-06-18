#!/usr/bin/env python3
"""Move person models randomly in the Gazebo warehouse_people world.

Uses gz.transport to call /world/<world>/set_pose, interpolating
smoothly between waypoints so sensors (camera, lidar) see movement.

Usage:
  python3 scripts/move_people.py                        # defaults
  python3 scripts/move_people.py --world warehouse_people --speed 0.8
"""
import argparse
import math
import random
import time

from gz.msgs10 import boolean_pb2, pose_pb2, pose_v_pb2
from gz.transport13 import Node

# ── Walkable area (warehouse interior, conservative bounds) ──────
X_MIN, X_MAX = -10.0, 12.0
Y_MIN, Y_MAX = -12.0, 20.0
Z_GROUND = 0.0  # person models sit on the ground plane

# People model names as they appear in warehouse_people.sdf
PERSON_MODELS = [
    "Person 0 - Sitting",
    "Person 1 - Standing",
    "person_2_standing",
    "person_3_phone",
    "person_4_sitting",
    "person_5_standing",
    "person_6_phone",
    "person_7_standing",
]

# ── Helpers ──────────────────────────────────────────────────────

def random_target() -> tuple[float, float, float]:
    """Pick a random walkable position."""
    return (
        random.uniform(X_MIN, X_MAX),
        random.uniform(Y_MIN, Y_MAX),
        Z_GROUND,
    )


def yaw_towards(dx: float, dy: float) -> tuple[float, float, float, float]:
    """Return a quaternion (x, y, z, w) facing the direction (dx, dy)."""
    yaw = math.atan2(dy, dx)
    # quaternion from yaw only (roll=pitch=0)
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class PersonMover:
    """Interpolates a single model between random waypoints."""

    def __init__(self, name: str, speed: float):
        self.name = name
        self.speed = speed
        # Start at a random position
        self.cx, self.cy, self.cz = random_target()
        self.tx, self.ty, self.tz = random_target()
        self._recompute()

    def _recompute(self):
        dx = self.tx - self.cx
        dy = self.ty - self.cy
        self.dist = math.hypot(dx, dy)
        self.progress = 0.0
        if self.dist > 0.01:
            self.qx, self.qy, self.qz, self.qw = yaw_towards(dx, dy)
        else:
            self.qx, self.qy, self.qz, self.qw = 0.0, 0.0, 0.0, 1.0

    def step(self, dt: float) -> pose_pb2.Pose:
        """Advance by dt seconds, return the new Pose message."""
        if self.dist > 0.01:
            self.progress += (self.speed * dt) / self.dist
        else:
            self.progress = 1.0

        if self.progress >= 1.0:
            # Arrived → pick new target
            self.cx, self.cy, self.cz = self.tx, self.ty, self.tz
            self.tx, self.ty, self.tz = random_target()
            self._recompute()

        x = lerp(self.cx, self.tx, self.progress)
        y = lerp(self.cy, self.ty, self.progress)

        pose = pose_pb2.Pose()
        pose.name = self.name
        pose.position.x = x
        pose.position.y = y
        pose.position.z = Z_GROUND
        pose.orientation.x = self.qx
        pose.orientation.y = self.qy
        pose.orientation.z = self.qz
        pose.orientation.w = self.qw
        return pose


def _kill_other_instances():
    """Kill any other running move_people.py processes to avoid conflicts."""
    import os
    import signal
    my_pid = os.getpid()
    try:
        import subprocess
        out = subprocess.check_output(
            ["pgrep", "-f", "move_people\\.py"], text=True
        )
        for line in out.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"[INFO] Killed stale move_people.py (PID {pid})")
                except ProcessLookupError:
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def main():
    parser = argparse.ArgumentParser(description="Move person models randomly in Gazebo")
    parser.add_argument("--world", default="warehouse_people", help="Gazebo world name")
    parser.add_argument("--speed", type=float, default=0.4, help="Walk speed m/s (default 0.4)")
    parser.add_argument("--hz", type=float, default=10.0, help="Update rate Hz (default 10)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to move (default: all 8 people)")
    args = parser.parse_args()

    _kill_other_instances()

    models = args.models if args.models else PERSON_MODELS
    service = f"/world/{args.world}/set_pose"
    dt = 1.0 / args.hz

    node = Node()
    movers = [PersonMover(name, args.speed) for name in models]

    print(f"Moving {len(movers)} people at {args.speed} m/s, {args.hz} Hz")
    print(f"Service: {service}")
    print("Press Ctrl+C to stop\n")

    # First request uses longer timeout to let transport discover the service
    first = True
    last_time = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            real_dt = now - last_time
            last_time = now

            timeout_ms = 3000 if first else 500
            for mover in movers:
                pose = mover.step(real_dt)
                result, _ = node.request(
                    service, pose,
                    pose_pb2.Pose, boolean_pb2.Boolean,
                    timeout=timeout_ms,
                )
                if not result:
                    print(f"[WARN] set_pose failed for {mover.name}")
            first = False

            elapsed = time.monotonic() - now
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
