#!/usr/bin/env python3
"""ROS 2 node: person detection + multi-object TRACKING (perception only).

Runs YOLO with a multi-object tracker (BoT-SORT by default) via Ultralytics
``model.track(persist=True)``: each person gets a STABLE track id across frames.
Publishes ALL tracked persons in a Detection2DArray; the follower decides which one
is the target (by proximity + dwell). For each person it also publishes the DISTANCE
measured from the OAK-D depth image at the person's torso (relación distancia↔imagen,
método del PDF de Eugenio) in ``results[0].pose.pose.position.z``.

NOTE: in this Gazebo sim the 2D LiDAR does NOT see the animated actors (no collision),
so the per-person range comes from the OAK-D depth camera; the LiDAR is reserved for
obstacle avoidance in the follower.

Also publishes an annotated image on /person_detections/image (the target id, taken
from /follow_target_id published by the follower, is drawn in green). Bypasses
cv_bridge (broken with NumPy 2.x on Jazzy).
"""
import argparse
import sys
from pathlib import Path

import rclpy
import rclpy.parameter
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from ultralytics import YOLO


def resolve_model(name: str) -> str:
    """Localiza el fichero de pesos en directorios conocidos (el TFG lo lanza desde
    ~/tfg_ws pero los pesos están en src/tfg_workflow/). Si no, devuelve el nombre tal
    cual (Ultralytics lo descargará si es un modelo oficial)."""
    here = Path(__file__).resolve()
    for c in [Path(name), Path.cwd() / name, here.parent / name, here.parent.parent / name]:
        if c.is_file():
            return str(c)
    return name


# ── Image conversion (no cv_bridge) ──────────────────────────
_ENC_INFO = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4), "mono8": (np.uint8, 1),
}


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → BGR numpy array."""
    enc = msg.encoding
    if enc not in _ENC_INFO:
        raise ValueError(f"Unsupported encoding: {enc}")
    dtype, ch = _ENC_INFO[enc]
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, ch)
    if enc == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def depthmsg_to_np(msg: Image) -> np.ndarray:
    """sensor_msgs/Image 32FC1 → float32 HxW (metros)."""
    dt = np.dtype(np.float32).newbyteorder('>' if msg.is_bigendian else '<')
    arr = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, msg.width)
    return np.ascontiguousarray(arr, dtype=np.float32)


def cv2_to_imgmsg(bgr: np.ndarray, header) -> Image:
    """BGR numpy array → sensor_msgs/Image (bgr8), sin cv_bridge."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


# ── ROS 2 Node ───────────────────────────────────────────────
class PersonDetectorNode(Node):
    """YOLO + BoT-SORT; publica TODAS las personas (id + bbox + distancia por profundidad)."""

    def __init__(self, model_file: str, image_topic: str, conf_thresh: float,
                 tracker_cfg: str, depth_topic: str):
        super().__init__("person_detector")

        self.declare_parameter("process_hz", 12.0)
        self._last_proc = None

        resolved = resolve_model(model_file)
        self.get_logger().info(f"Cargando modelo YOLO: {resolved}")
        self.model = YOLO(resolved)
        self.conf_thresh = conf_thresh
        self.tracker_cfg = tracker_cfg

        self.latest_depth = None          # último frame de profundidad (m), cacheado
        self.follow_target_id = -1        # id que el follower está siguiendo (para pintarlo)

        self.get_logger().info(
            f"Tracker: {self.tracker_cfg} | distancia por profundidad ({depth_topic})"
        )

        # Pub / Sub (QoS de sensor: best-effort, frame más reciente)
        self.det_pub = self.create_publisher(Detection2DArray, "/person_detections", 10)
        self.img_pub = self.create_publisher(Image, "/person_detections/image", qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Int32, "/follow_target_id", self._on_target_id, 10)

        self.get_logger().info(
            f"Escuchando {image_topic} (+depth) → publicando todas las personas en /person_detections"
        )

    def _on_depth(self, msg: Image):
        try:
            self.latest_depth = depthmsg_to_np(msg)
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")

    def _on_target_id(self, msg: Int32):
        self.follow_target_id = int(msg.data)

    def _track_depth(self, x1, y1, x2, y2) -> float:
        """Distancia (m) a la persona = mediana de la profundidad en su TORSO (franja
        superior-central de la caja), robusta a fondo/piernas. 0.0 si no hay dato."""
        d = self.latest_depth
        if d is None:
            return 0.0
        h, w = d.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        ty1, ty2 = y1 + int(0.20 * bh), y1 + int(0.55 * bh)
        tx1, tx2 = x1 + int(0.25 * bw), x1 + int(0.75 * bw)
        roi = d[max(0, ty1):min(h, ty2), max(0, tx1):min(w, tx2)]
        valid = roi[np.isfinite(roi) & (roi > 0.05)]
        if valid.size < 10:
            return 0.0
        return float(np.median(valid))

    # ── Image callback ───────────────────────────────────────
    def _on_image(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        hz = float(self.get_parameter("process_hz").value)
        if hz > 0 and self._last_proc is not None and (now - self._last_proc) < (1.0 / hz):
            return
        self._last_proc = now

        try:
            cv_img = imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
        h, w, _ = cv_img.shape

        # YOLO + tracker (estado persistente → ids estables)
        results = self.model.track(
            cv_img, classes=[0], conf=self.conf_thresh, persist=True,
            tracker=self.tracker_cfg, verbose=False, device="cuda:0",
        )

        det_array = Detection2DArray()
        det_array.header = msg.header

        tracks = []
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, conf in zip(xyxy, ids, confs):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                tracks.append({'id': int(tid), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                               'conf': float(conf), 'depth': self._track_depth(x1, y1, x2, y2)})

        # ── Publicar TODAS las personas (id, bbox, distancia por profundidad) ──
        for t in tracks:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (t['x1'] + t['x2']) / 2.0
            det.bbox.center.position.y = (t['y1'] + t['y2']) / 2.0
            det.bbox.size_x = float(t['x2'] - t['x1'])
            det.bbox.size_y = float(t['y2'] - t['y1'])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = f"person:{t['id']}"
            hyp.hypothesis.score = t['conf']
            hyp.pose.pose.position.z = t['depth']     # distancia (m) por profundidad de cámara
            det.results.append(hyp)
            det_array.detections.append(det)
        self.det_pub.publish(det_array)

        # ── Dibujo: objetivo (id que sigue el follower) en verde, resto en rojo ──
        for t in tracks:
            is_target = (t['id'] == self.follow_target_id)
            color = (0, 255, 0) if is_target else (0, 0, 255)
            tag = f"TARGET id{t['id']} {t['depth']:.1f}m" if is_target else f"id{t['id']} {t['depth']:.1f}m"
            cv2.rectangle(cv_img, (t['x1'], t['y1']), (t['x2'], t['y2']), color, 2)
            cv2.putText(cv_img, tag, (t['x1'], max(0, t['y1'] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        try:
            self.img_pub.publish(cv2_to_imgmsg(cv_img, msg.header))
        except Exception:
            pass
        try:
            cv2.imshow("YOLO + BoT-SORT (TFG) - objetivo en verde", cv_img)
            cv2.waitKey(1)
        except cv2.error:
            pass  # headless

        # Log de diagnóstico (cada ~2 s): personas detectadas, su profundidad, y si llega depth.
        if not hasattr(self, "_last_log") or (now - self._last_log) >= 2.0:
            self._last_log = now
            depth_ok = "OK" if self.latest_depth is not None else "NO"
            info = ", ".join(f"id{t['id']}@{t['depth']:.1f}m" for t in tracks[:4]) or "ninguna"
            self.get_logger().info(f"{len(tracks)} personas [{info}] | depth_topic={depth_ok}")


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO + BoT-SORT person detector (perception) — ROS 2 node"
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--topic", default="/oakd/rgb/preview/image_raw")
    parser.add_argument("--depth-topic", default="/oakd/rgb/preview/depth")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--tracker", default="botsort.yaml",
                        help="Ultralytics tracker (botsort.yaml | bytetrack.yaml)")

    rclpy.init(args=sys.argv)
    cleaned = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(cleaned)

    node = PersonDetectorNode(
        model_file=args.model, image_topic=args.topic, conf_thresh=args.conf,
        tracker_cfg=args.tracker, depth_topic=args.depth_topic,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        # Carrera típica al cerrar (take_message mientras se destruye el contexto tras el
        # SIGINT del launch): no es fatal, el cleanup del finally se ejecuta igual.
        try:
            node.get_logger().warn(f"spin terminado por excepción (probable cierre): {e}")
        except Exception:
            pass
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
