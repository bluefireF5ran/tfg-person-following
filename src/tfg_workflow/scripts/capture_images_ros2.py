#!/usr/bin/env python3
"""Capture images from a ROS 2 Image topic to disk.

Bypasses cv_bridge (broken with NumPy 2.x on Jazzy) by converting
sensor_msgs/Image → numpy array directly.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Encoding → (numpy dtype, channels)
_ENC_INFO = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
}


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to a BGR numpy array (no cv_bridge)."""
    enc = msg.encoding
    if enc not in _ENC_INFO:
        raise ValueError(f"Encoding '{enc}' not supported. Expected one of {list(_ENC_INFO)}")
    dtype, channels = _ENC_INFO[enc]
    arr = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, channels))
    if enc == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


class ImageCaptureNode(Node):
    def __init__(self, topic: str, output_dir: Path, rate: float, max_images: int):
        super().__init__("image_capture_node")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_interval = 1.0 / max(rate, 0.1)
        self.last_time = 0.0
        self.counter = 0
        self.max_images = max_images

        self.subscription = self.create_subscription(Image, topic, self.callback, 10)
        self.get_logger().info(f"Suscrito a topic: {topic}")
        self.get_logger().info(f"Guardando imágenes en: {self.output_dir}")

    def callback(self, msg: Image):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_time < self.min_interval:
            return

        frame = imgmsg_to_cv2(msg)
        filename = self.output_dir / f"img_{self.counter:06d}.jpg"
        cv2.imwrite(str(filename), frame)

        self.counter += 1
        self.last_time = now_sec

        if self.counter % 10 == 0:
            self.get_logger().info(f"Imágenes guardadas: {self.counter}")

        if self.max_images > 0 and self.counter >= self.max_images:
            self.get_logger().info(f"Objetivo alcanzado ({self.max_images} imágenes). Cerrando nodo.")
            raise SystemExit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Captura imágenes desde un topic ROS 2 de tipo sensor_msgs/Image")
    parser.add_argument("--topic", type=str, required=True, help="Topic de imagen ROS 2")
    parser.add_argument("--output_dir", type=str, required=True, help="Carpeta de salida")
    parser.add_argument("--rate", type=float, default=2.0, help="Frecuencia máxima de guardado (Hz)")
    parser.add_argument("--max_images", type=int, default=0, help="Número máximo de imágenes (0 = sin límite)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    rclpy.init()
    node = ImageCaptureNode(args.topic, output_dir, args.rate, args.max_images)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
