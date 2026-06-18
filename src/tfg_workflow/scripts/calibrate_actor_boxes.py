#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# calibrate_actor_boxes.py: Calibra, POR MALLA, las dimensiones del box 3D de cada
# actor para que la proyección 2D del GT coincida con las cajas reales que detecta
# YOLO. Genera un JSON (actor -> {half_width, person_height}) que consume
# capture_mot_dataset.py (--box_config).
#
# Motivación (Plan B): un box 3D fijo no ajusta a mallas distintas (tip, female,
# nurse, male). Aquí se mide el factor de escala por actor frente a YOLO y se
# corrige, en vez de adivinar. El BoundingBoxCamera de Gazebo está bloqueado en
# esta máquina (conflicto ogre/ogre2 + EGL), de ahí este enfoque geométrico.
#
# Requiere la simulación corriendo (robot ESTÁTICO en su pose de spawn):
#   bash scripts/run_tb4_sim.sh warehouse_unified
#
# Uso:
#   python3 calibrate_actor_boxes.py --sdf ../worlds/warehouse_unified.sdf \
#       --world warehouse_unified --frames 400 \
#       --out ../data/mot_dataset/actor_box_config.json

import argparse
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Punto de partida del box (se escalará por malla).
BASE_HALF_WIDTH = 0.22
BASE_PERSON_HEIGHT = 1.70
GATE_PX = 45.0          # distancia máx. (px) entre centro proyectado y centro YOLO
MIN_SAMPLES = 8         # muestras mínimas por actor para fiarse de la calibración


def find_model(name):
    for c in [name, os.path.join(SCRIPT_DIR, "..", name),
              os.path.join(os.path.expanduser("~/tfg_ws"), name)]:
        if os.path.isfile(c):
            return c
    return name


def main():
    p = argparse.ArgumentParser(description="Calibración por malla del box de actores vs YOLO")
    p.add_argument("--sdf", required=True)
    p.add_argument("--world", default="warehouse_unified")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--frames_per_vp", type=int, default=100,
                   help="Frames a recoger en cada viewpoint")
    p.add_argument("--viewpoints", default=" -2,0,90  2,0,110  0,-2,80  -3,1,60  3,1,120  -1,4,185  0,7,-90 ",
                   help="Viewpoints 'x,y,yaw  ...' (teleport) para ver a TODOS los actores")
    p.add_argument("--settle_time", type=float, default=2.0)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--camera_ns", default="/oakd/rgb/preview")
    p.add_argument("--cam_x", type=float, default=-0.037)
    p.add_argument("--cam_z", type=float, default=0.19)
    p.add_argument("--robot_model", default="turtlebot4")
    p.add_argument("--ground_z", type=float, default=0.0)
    p.add_argument("--robot_x", type=float, default=-2.0)
    p.add_argument("--robot_y", type=float, default=2.0)
    p.add_argument("--robot_yaw_deg", type=float, default=0.0)
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/data/actor_box_config.json"),
                   help="JSON de salida. FUERA del dataset para que el borrado del orquestador no lo elimine.")
    args = p.parse_args(__import__("rclpy").utilities.remove_ros_args()[1:])

    import threading
    import time
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from ultralytics import YOLO

    import capture_with_gt as c
    from capture_with_gt import parse_actors_from_sdf, compute_bbox, imgmsg_to_cv2, teleport_robot
    from gz.transport13 import Node as GzNode

    # Viewpoints (teleport) para que TODOS los actores entren en cuadro alguna vez.
    viewpoints = []
    for tok in args.viewpoints.strip().split():
        x, y, yw = tok.split(",")
        viewpoints.append((float(x), float(y), math.radians(float(yw))))
    print(f"[INFO] {len(viewpoints)} viewpoints, {args.frames_per_vp} frames cada uno.")

    actors = parse_actors_from_sdf(args.sdf)
    names = sorted(a.name for a in actors)
    model = YOLO(find_model(args.model))

    acc = {n: {"wr": [], "hr": []} for n in names}

    class CalibNode(Node):
        def __init__(self):
            super().__init__("calibrate_actor_boxes")
            self.fx = None; self.lock = threading.Lock()
            self.n = 0
            self.vp_idx = -1
            self.frames_in_vp = 0
            self.robot_xyz = (0.0, 0.0, 0.0); self.robot_yaw = 0.0
            self.settle_until = 0.0
            self.gz = GzNode()
            self.create_subscription(CameraInfo, f"{args.camera_ns}/camera_info", self._info, 10)
            self.create_subscription(Image, f"{args.camera_ns}/image_raw", self._img, qos_profile_sensor_data)
            self._next_vp()

        def _next_vp(self):
            self.vp_idx += 1
            if self.vp_idx >= len(viewpoints):
                raise SystemExit(0)
            x, y, yw = viewpoints[self.vp_idx]
            ok = teleport_robot(self.gz, args.world, args.robot_model, x, y, yw)
            self.robot_xyz, self.robot_yaw = (x, y, 0.0), yw
            self.frames_in_vp = 0
            self.settle_until = time.monotonic() + args.settle_time
            self.get_logger().info(f"[vp{self.vp_idx+1}] ({x:.1f},{y:.1f},{math.degrees(yw):.0f}°) teleport={'OK' if ok else 'FALLO'}")

        def _info(self, m):
            with self.lock:
                if self.fx is None:
                    self.fx, self.fy, self.cx, self.cy = m.k[0], m.k[4], m.k[2], m.k[5]
                    self.w, self.h = m.width, m.height

        def _img(self, m):
            with self.lock:
                if self.fx is None:
                    return
                fx, fy, cx, cy, W, H = self.fx, self.fy, self.cx, self.cy, self.w, self.h
            if time.monotonic() < self.settle_until:
                return
            robot_xyz, robot_yaw = self.robot_xyz, self.robot_yaw
            frame = imgmsg_to_cv2(m)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            res = model.predict(frame, classes=[0], conf=args.conf, device="cuda:0", verbose=False)
            dets = []
            if len(res) and res[0].boxes is not None:
                for b in res[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b
                    dets.append((x1, y1, x2 - x1, y2 - y1))  # left,top,w,h
            # Proyección de cada actor con box base (pies al suelo).
            for a in actors:
                ax, ay, az = a.pose_at(t)
                c.ACTOR_BOTTOM_OFFSET = az - args.ground_z
                c.ACTOR_TOP_OFFSET = (args.ground_z + BASE_PERSON_HEIGHT) - az
                c.ACTOR_HALF_WIDTH = BASE_HALF_WIDTH
                bb = compute_bbox((ax, ay, az), robot_xyz, robot_yaw,
                                  args.cam_x, 0.0, args.cam_z, fx, fy, cx, cy, W, H)
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                pw, ph = x2 - x1, y2 - y1
                # descarta cajas recortadas en el borde (sesgan la escala)
                if x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1:
                    continue
                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                # empareja con la detección YOLO de centro más cercano dentro del gate
                best = None; bestd = GATE_PX
                for (dl, dt, dw, dh) in dets:
                    dcx, dcy = dl + dw / 2, dt + dh / 2
                    d = math.hypot(pcx - dcx, pcy - dcy)
                    if d < bestd:
                        bestd = d; best = (dw, dh)
                if best and pw > 3 and ph > 3:
                    acc[a.name]["wr"].append(best[0] / pw)
                    acc[a.name]["hr"].append(best[1] / ph)
            self.n += 1
            self.frames_in_vp += 1
            if self.frames_in_vp >= args.frames_per_vp:
                self._next_vp()

    rclpy.init()
    node = CalibNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()

    # Calibración: escala el box base por la mediana de ratios YOLO/proyección.
    appr = {"actor_0": "tip_walk", "actor_1": "tip_run", "actor_2": "female",
            "actor_3": "nurse", "actor_4": "male"}
    config = {}
    print("\n" + "=" * 60)
    print("  CALIBRACIÓN POR MALLA (ratio YOLO/proyección)")
    print("=" * 60)
    for n in names:
        wr, hr = acc[n]["wr"], acc[n]["hr"]
        if len(wr) >= MIN_SAMPLES:
            hw = BASE_HALF_WIDTH * float(np.median(wr))
            phh = BASE_PERSON_HEIGHT * float(np.median(hr))
        else:
            hw, phh = BASE_HALF_WIDTH, BASE_PERSON_HEIGHT
        config[n] = {"half_width": round(hw, 3), "person_height": round(phh, 3),
                     "n_samples": len(wr), "appearance": appr.get(n, "?")}
        print(f"  {n} ({appr.get(n,'?'):9s}): n={len(wr):3d}  "
              f"half_width={hw:.3f}  person_height={phh:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[SUCCESS] Config de cajas por actor: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
