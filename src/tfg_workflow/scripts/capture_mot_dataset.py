#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# capture_mot_dataset.py: Genera un dataset de SEGUIMIENTO multi-objeto etiquetado
# (formato MOTChallenge) a partir del escenario de Gazebo, asignando un ID ENTERO
# ESTABLE a cada actor. Es el último encargo del correo del tutor: "etiquetar un
# dataset con imágenes de personas simuladas poniendo una etiqueta a cada persona
# para diferenciarla de las demás".
#
# Reutiliza la maquinaria ya validada de capture_with_gt.py:
#   - parse_actors_from_sdf / ActorTrajectory : trayectorias por actor
#   - compute_bbox                            : proyección 3D -> 2D
#   - teleport_robot / imgmsg_to_cv2          : control y conversión
#
# Para que el tracking tenga sentido (correo del tutor), las personas deben ser
# VISUALMENTE DISTINTAS. El escenario warehouse_unified.sdf tiene 5 actores con
# 4 apariencias (Mingfei tip x2 -caso difícil-, female_visitor, nurse, male_visitor).
#
# Estrategia de captura:
#   - El robot permanece ESTÁTICO en cada "viewpoint" (se teletransporta una vez),
#     de modo que cada viewpoint produce una SECUENCIA temporal continua (seqNN)
#     con IDs trazables mientras los actores se mueven por sus trayectorias.
#   - Varios viewpoints => dataset variado (ángulos/distancias distintos).
#
# Salida (estándar MOTChallenge):
#   OUT/seq01/img1/000001.jpg ...
#   OUT/seq01/gt/gt.txt          -> frame,id,bb_left,bb_top,w,h,1,1,1
#   OUT/seq01/seqinfo.ini
#   OUT/id_map.json              -> mapa actor->id (+ apariencia)
#
# Uso (con la simulación de warehouse_unified corriendo):
#   python3 capture_mot_dataset.py \
#     --sdf ../worlds/warehouse_unified.sdf --world warehouse_unified \
#     --output_dir ../data/mot_dataset \
#     --viewpoints "-2,0,90  0,-1,75  3,0,110" \
#     --frames_per_seq 150 --rate 10
#
# Validación offline de la lógica (sin Gazebo):
#   python3 capture_mot_dataset.py --selftest

import argparse
import configparser
import json
import math
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Apariencia de cada actor en warehouse_unified.sdf (para documentar el id_map).
ACTOR_APPEARANCE = {
    "actor_0": "Mingfei_tip (walk)",
    "actor_1": "Mingfei_tip (run)",      # misma malla que actor_0 -> caso difícil
    "actor_2": "female_visitor",
    "actor_3": "nurse",
    "actor_4": "male_visitor",
}


# Colores BGR por ID para el modo de verificación visual (--draw).
_ID_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def build_id_map(actors) -> dict:
    """Asigna un ID entero estable y determinista a cada actor (orden por nombre)."""
    names = sorted(a.name for a in actors)
    return {name: i + 1 for i, name in enumerate(names)}


def apply_actor_box(half_width=None, bottom_offset=None, top_offset=None):
    """Ajusta (sin editar el fichero) las dimensiones del box 3D del actor que usa
    compute_bbox de capture_with_gt. Permite cajas más ceñidas para el dataset MOT
    sin afectar al pipeline de detección. Valores None dejan el por defecto."""
    import capture_with_gt as c
    if half_width is not None:
        c.ACTOR_HALF_WIDTH = half_width
    if bottom_offset is not None:
        c.ACTOR_BOTTOM_OFFSET = bottom_offset
    if top_offset is not None:
        c.ACTOR_TOP_OFFSET = top_offset


def _actor_forward_depth(actor_xyz, robot_xyz, robot_yaw, cam):
    """Profundidad (Zo, distancia hacia delante en el eje óptico) de un punto 3D."""
    ax, ay, _ = actor_xyz
    rx, ry, _ = robot_xyz
    dx, dy = ax - rx, ay - ry
    cos_n, sin_n = math.cos(-robot_yaw), math.sin(-robot_yaw)
    x_r = cos_n * dx - sin_n * dy
    return x_r - cam["cam_x"]


def _is_occluded(depth, x1, y1, x2, y2, actor_depth, margin):
    """True si lo medido por el sensor en el centro de la caja está claramente más
    cerca que el actor (margin m), es decir, hay algo delante tapándolo."""
    import numpy as np
    h, w = depth.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
    x0, xe = max(0, cx - bw), min(w, cx + bw + 1)
    y0, ye = max(0, cy - bh), min(h, cy + bh + 1)
    patch = depth[y0:ye, x0:xe]
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if valid.size == 0:
        return False  # sin dato de profundidad: no se descarta
    return float(np.median(valid)) < (actor_depth - margin)


def _refine_box_depth(depth, x1, y1, x2, y2, actor_depth, near_margin, far_margin,
                      min_frac=0.10, fx=None):
    """Ciñe la caja a la SILUETA real del actor usando profundidad. Busca en un ROI
    ensanchado lateralmente alrededor de la caja calibrada para recuperar al actor
    aunque la proyección geométrica esté desplazada (offset lateral de la malla al
    caminar/correr), y devuelve el bounding box de la silueta. Como la banda de
    profundidad es asimétrica (estricta por detrás), no coge el fondo, así que el
    ancho resultante es el de la persona (feedback de Eugenio).
    Se ancla en la profundidad ESPERADA del actor (Zo geométrico).

    Banda de profundidad ASIMÉTRICA respecto a Zo: generosa hacia la cámara
    (Zo - near_margin) para captar brazos/piernas adelantados al correr, pero
    ESTRICTA por detrás (Zo + far_margin) para EXCLUIR las estanterías/cajas que
    están justo detrás del actor (causaban cajas más anchas que la persona).

    Devuelve None si hay pocos píxeles del actor dentro de la caja -> descarta cajas
    sobre personajes fuera de escena o muy desplazados (entrada/salida de cuadro).
    Usa profundidad (geometría real del simulador), no el detector RGB.
    """
    import numpy as np
    h, w = depth.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    bw, bh = x2 - x1, y2 - y1
    # ROI ensanchado LATERALMENTE (y un poco hacia arriba, nunca hacia el suelo) para
    # recuperar al actor aunque la proyección geométrica esté desplazada (offset lateral
    # de la malla al caminar/correr). La banda de profundidad asimétrica evita coger el
    # fondo, así que ensanchar el ROI es seguro. Tope absoluto para no fusionar actores
    # vecinos cercanos.
    px = min(int(0.7 * bw), 35)
    py = int(0.10 * bh)
    rx1, ry1 = max(0, x1 - px), max(0, y1 - py)
    rx2, ry2 = min(w, x2 + px), y2     # ry2 = y2: no bajar al suelo
    roi = depth[ry1:ry2, rx1:rx2]
    # Ancla de la banda = SUPERFICIE FRONTAL real del actor (percentil bajo de las
    # profundidades válidas del ROI), no el Zo geométrico: a distancia la proyección
    # geométrica se desvía de lo renderizado y, con far estricto, el cuerpo (~0.35 m de
    # fondo) cae justo detrás del límite y el actor se pierde. Solo se confía en el
    # percentil si está cerca de Zo (si está mucho más cerca, es clutter de primer plano
    # -mesa/objeto- y se vuelve a Zo). Banda hacia atrás amplia para cubrir el cuerpo.
    fin = roi[np.isfinite(roi) & (roi > 0.05)]
    anchor = actor_depth
    if fin.size:
        front = float(np.percentile(fin, 10))
        if abs(front - actor_depth) <= 0.6:
            anchor = front
    mask = (np.isfinite(roi) & (roi > 0.05) &
            (roi > anchor - near_margin) & (roi < anchor + far_margin))
    # Umbral de "hay actor" relativo al ÁREA DE LA CAJA ORIGINAL (bw*bh), NO al ROI
    # ensanchado: ensanchar el ROI infla su área y, con un umbral sobre el ROI, se
    # caerían actores VISIBLES pero lejanos/finos (silueta pequeña en un ROI grande).
    min_px = max(80, int(min_frac * bw * bh))
    if int(mask.sum()) < min_px:
        return None
    import cv2
    # Cierre morfológico: une fragmentos de la silueta de un actor lejano (depth con
    # ruido) en un solo blob, sin fusionar objetos separados por un hueco (mesa/estante).
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Quedarse con la COMPONENTE CONEXA MÁS GRANDE (= la silueta del actor), para
    # descartar blobs sueltos a la misma profundidad: mesas/objetos de primer plano,
    # parches de estantería al lado, etc. (evita cajas que se van hacia esos objetos).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # mayor componente (sin fondo)
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return None
    cc = (lab == k)
    ys_all, xs_all = np.where(cc)
    cl, cr = int(xs_all.min()), int(xs_all.max())          # extensión X del actor en el ROI
    cw = cr - cl + 1
    # CAP de ancho físicamente fundamentado: una persona a profundidad `anchor` ocupa
    # como mucho ~0.8 m (brazos abiertos al correr) => w_max = fx*0.8/anchor px. Si la
    # componente sale más ancha, está contaminada por estantería/suelo coplanar al
    # entrar/salir de cuadro: se recorta a la franja vertical de ancho w_max con MÁS
    # píxeles del actor (su cuerpo), en vez de dar una caja más ancha que la persona.
    if fx and anchor > 0:
        max_w = int(fx * 0.8 / anchor)
        if 4 <= max_w < cw:
            col = cc.sum(axis=0).astype(np.int64)
            csum = np.cumsum(col)
            best_l, best_v = cl, -1
            for l in range(cl, cr - max_w + 2):
                v = int(csum[l + max_w - 1] - (csum[l - 1] if l > 0 else 0))
                if v > best_v:
                    best_v, best_l = v, l
            cl, cr = best_l, best_l + max_w - 1
            ys_all, _ = np.where(cc[:, cl:cr + 1])
    top, bot = int(ys_all.min()), int(ys_all.max())
    return (rx1 + cl, ry1 + top, rx1 + cr + 1, ry1 + bot + 1)


def compute_frame_gt(actors, id_map, robot_xyz, robot_yaw, cam, sim_time, img_wh,
                     depth=None, occ_margin=0.5, person_height=1.75, ground_z=0.0,
                     box_config=None, half_width=None, depth_refine=False, refine_margin=0.15, back_margin=0.35):
    """Devuelve la lista [(id, left, top, w, h)] de actores visibles en un frame.

    `cam` es un dict con fx,fy,cx,cy,cam_x,cam_y,cam_z. `img_wh` es (w,h).
    Si `depth` (array HxW en metros) se pasa, se descartan los actores ocluidos.

    IMPORTANTE: los actores pisan el suelo independientemente de la convención de
    origen de su malla (unos tienen la trayectoria en z=1.0 -pelvis- y otros en
    z=0.0 -pies-). Por eso la caja se fija en z mundo [ground_z, ground_z+altura]
    para TODOS, ajustando por-actor los offsets que usa compute_bbox.

    `box_config` (Plan B): dict opcional actor->{half_width, person_height} que
    calibra el box POR MALLA (calibrate_actor_boxes.py). Si no, usa los globales.

    Función pura (sin ROS/Gazebo) para poder validarla offline.
    """
    import capture_with_gt as c
    from capture_with_gt import compute_bbox
    box_config = box_config or {}
    img_w, img_h = img_wh
    rows = []
    for actor in actors:
        ax, ay, az = actor.pose_at(sim_time)
        # Parámetros del box: por malla si hay config, si no los globales.
        cfg = box_config.get(actor.name, {})
        ph = cfg.get("person_height", person_height)
        hw = cfg.get("half_width", half_width)
        # Fija la caja al suelo real para esta malla (corrige la convención de origen).
        c.ACTOR_BOTTOM_OFFSET = az - ground_z
        c.ACTOR_TOP_OFFSET = (ground_z + ph) - az
        if hw is not None:
            c.ACTOR_HALF_WIDTH = hw
        bbox = compute_bbox(
            (ax, ay, az), robot_xyz, robot_yaw,
            cam["cam_x"], cam["cam_y"], cam["cam_z"],
            cam["fx"], cam["fy"], cam["cx"], cam["cy"], img_w, img_h,
        )
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if depth is not None:
            actor_depth = _actor_forward_depth((ax, ay, az), robot_xyz, robot_yaw, cam)
            if _is_occluded(depth, x1, y1, x2, y2, actor_depth, occ_margin):
                continue
            if depth_refine:
                ref = _refine_box_depth(depth, x1, y1, x2, y2, actor_depth, refine_margin, back_margin,
                                        fx=cam["fx"])
                if ref is None:
                    continue   # sin píxeles del actor a esa profundidad -> no caja
                x1, y1, x2, y2 = ref
        rows.append((id_map[actor.name], x1, y1, x2 - x1, y2 - y1))
    return rows


def write_seqinfo(seq_dir: Path, name: str, length: int, fps: float, w: int, h: int):
    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name": name, "imDir": "img1", "frameRate": str(int(round(fps))),
        "seqLength": str(length), "imWidth": str(w), "imHeight": str(h),
        "imExt": ".jpg",
    }
    with open(seq_dir / "seqinfo.ini", "w") as f:
        ini.write(f)


def selftest():
    """Valida offline parseo + id_map + proyección + escritura MOTChallenge."""
    from capture_with_gt import parse_actors_from_sdf
    sdf = os.path.join(SCRIPT_DIR, "..", "worlds", "warehouse_unified.sdf")
    actors = parse_actors_from_sdf(sdf)
    id_map = build_id_map(actors)
    print("[selftest] actores:", [(a.name, id_map[a.name], ACTOR_APPEARANCE.get(a.name, "?"))
                                   for a in actors])

    # Cámara sintética tipo OAK-D preview (250x250 aprox) + robot estático mirando +Y.
    cam = {"fx": 205.0, "fy": 205.0, "cx": 125.0, "cy": 125.0,
           "cam_x": -0.037, "cam_y": 0.0, "cam_z": 0.19}
    img_wh = (250, 250)
    robot_xyz = (-1.0, 0.0, 0.0)
    robot_yaw = math.radians(90)   # mirando hacia +Y (donde caminan los actores)

    out = Path("/tmp/mot_selftest/seq01")
    (out / "img1").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    lines, n_frames, visible_total = [], 60, 0
    for fr in range(1, n_frames + 1):
        sim_t = fr / 10.0
        rows = compute_frame_gt(actors, id_map, robot_xyz, robot_yaw, cam, sim_t, img_wh)
        visible_total += len(rows)
        for (tid, l, t, w, h) in rows:
            lines.append(f"{fr},{tid},{l},{t},{w},{h},1,1,1")
    with open(out / "gt" / "gt.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    write_seqinfo(out, "seq01", n_frames, 10.0, *img_wh)

    ids_seen = sorted({int(l.split(",")[1]) for l in lines})
    print(f"[selftest] {n_frames} frames, {len(lines)} filas GT, "
          f"{visible_total} detecciones, IDs vistos={ids_seen}")
    print(f"[selftest] gt.txt -> {out/'gt'/'gt.txt'}")
    print("[selftest] muestra:")
    for l in lines[:6]:
        print("   ", l)
    if not lines:
        print("[selftest][WARN] No hubo detecciones; ajusta robot_xyz/yaw/cámara.")
        return 1
    print("[selftest] OK")
    return 0


# ═══════════════════════════════════════════════════════════════════
# Nodo ROS 2 de captura (requiere simulación de Gazebo en marcha)
# ═══════════════════════════════════════════════════════════════════

def run_capture(args):
    import threading
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    from capture_with_gt import (parse_actors_from_sdf, imgmsg_to_cv2,
                                 teleport_robot)
    from gz.transport13 import Node as GzNode

    actors = parse_actors_from_sdf(args.sdf)
    id_map = build_id_map(actors)

    # Parseo de viewpoints "x,y,yaw_deg  x,y,yaw_deg ..."
    if args.no_teleport:
        # El robot no se mueve: una sola secuencia con su pose real declarada.
        viewpoints = [(args.robot_x, args.robot_y, math.radians(args.robot_yaw_deg))]
        print(f"[INFO] Modo --no_teleport: pose real del robot = "
              f"({args.robot_x}, {args.robot_y}, {args.robot_yaw_deg}°), 1 secuencia.")
    else:
        viewpoints = []
        for tok in args.viewpoints.strip().split():
            x, y, yaw = tok.split(",")
            viewpoints.append((float(x), float(y), math.radians(float(yaw))))
        if not viewpoints:
            print("[ERROR] Indica al menos un --viewpoints 'x,y,yaw_deg'")
            return 1

    # Plan B: config de cajas por malla (calibrate_actor_boxes.py).
    box_config = {}
    if args.box_config and os.path.isfile(args.box_config):
        with open(args.box_config) as f:
            box_config = json.load(f)
        print(f"[INFO] Box por malla cargado de {args.box_config}: "
              + ", ".join(f"{k}(hw={v.get('half_width')},h={v.get('person_height')})"
                          for k, v in box_config.items()))
    elif args.box_config:
        print(f"[WARN] No existe el box_config {args.box_config}; uso box global.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "id_map.json", "w") as f:
        json.dump({"id_map": id_map,
                   "appearance": {n: ACTOR_APPEARANCE.get(n, "?") for n in id_map}},
                  f, indent=2)

    class MOTCaptureNode(Node):
        def __init__(self):
            super().__init__("capture_mot_dataset")
            self.actors = actors
            self.id_map = id_map
            self.fx = self.fy = self.cx = self.cy = None
            self.img_w = self.img_h = 0
            self._lock = threading.Lock()
            self.cam_x, self.cam_y, self.cam_z = args.cam_x, 0.0, args.cam_z

            self.vp_idx = -1
            self.frame_in_seq = 0
            self.lines = []
            self.seq_dir = None
            self.last_cap = 0.0
            self.settle_until = 0.0
            self.rate_interval = 1.0 / max(args.rate, 0.01)
            self.robot_xyz = (0.0, 0.0, 0.0)
            self.robot_yaw = 0.0
            self.gz = GzNode()

            self.latest_depth = None
            self.create_subscription(CameraInfo, f"{args.camera_ns}/camera_info",
                                     self._on_info, 10)
            self.create_subscription(Image, f"{args.camera_ns}/image_raw",
                                     self._on_image, qos_profile_sensor_data)
            if args.use_depth or args.depth_refine or args.save_depth:
                self.create_subscription(Image, args.depth_topic,
                                         self._on_depth, qos_profile_sensor_data)
                self.get_logger().info(f"Filtro de oclusión por profundidad ON ({args.depth_topic}, margen={args.occ_margin} m)")
            self._next_viewpoint()

        def _on_depth(self, msg: Image):
            import numpy as np
            dt = np.dtype(np.float32).newbyteorder('>' if msg.is_bigendian else '<')
            arr = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, msg.width)
            with self._lock:
                self.latest_depth = np.ascontiguousarray(arr, dtype=np.float32)

        def _next_viewpoint(self):
            # Cierra la secuencia anterior si la había
            if self.seq_dir is not None:
                self._finalize_seq()
            self.vp_idx += 1
            if self.vp_idx >= len(viewpoints):
                self.get_logger().info("Todos los viewpoints completados.")
                raise SystemExit(0)
            x, y, yaw = viewpoints[self.vp_idx]
            if args.no_teleport:
                ok = True   # el robot ya está colocado; usamos su pose declarada
            else:
                ok = teleport_robot(self.gz, args.world, args.robot_model, x, y, yaw)
            if not ok and not args.no_teleport:
                self.get_logger().error(
                    f"Teleport FALLÓ para el modelo '{args.robot_model}' en el mundo "
                    f"'{args.world}'. NO se genera GT (la pose del robot sería incorrecta).\n"
                    f"  -> Comprueba el nombre real del robot en Gazebo:\n"
                    f"       gz model --list      (o)   gz topic -e -t /world/{args.world}/pose/info -n 1 | grep name\n"
                    f"     y reejecuta con --robot_model <NOMBRE_REAL>.\n"
                    f"  -> Si el robot ya está colocado y NO quieres teletransportarlo,\n"
                    f"     usa --no_teleport --robot_x X --robot_y Y --robot_yaw_deg YAW\n"
                    f"     indicando su pose real."
                )
                raise SystemExit(2)
            self.robot_xyz, self.robot_yaw = (x, y, 0.0), yaw
            name = f"seq{self.vp_idx+1:02d}"
            self.seq_dir = out_root / name
            (self.seq_dir / "img1").mkdir(parents=True, exist_ok=True)
            (self.seq_dir / "gt").mkdir(parents=True, exist_ok=True)
            self.frame_in_seq = 0
            self.lines = []
            self.settle_until = time.monotonic() + args.settle_time
            self.get_logger().info(f"[{name}] viewpoint ({x:.1f},{y:.1f},{math.degrees(yaw):.0f}°) "
                                   f"teleport={'OK' if ok else 'FALLO'}")

        def _finalize_seq(self):
            with open(self.seq_dir / "gt" / "gt.txt", "w") as f:
                f.write("\n".join(self.lines) + ("\n" if self.lines else ""))
            write_seqinfo(self.seq_dir, self.seq_dir.name, self.frame_in_seq,
                          args.rate, self.img_w, self.img_h)
            self.get_logger().info(f"[{self.seq_dir.name}] {self.frame_in_seq} frames, "
                                   f"{len(self.lines)} filas GT")

        def _on_info(self, msg: CameraInfo):
            with self._lock:
                if self.fx is not None:
                    return
                self.img_w, self.img_h = msg.width, msg.height
                self.fx, self.fy = msg.k[0], msg.k[4]
                self.cx, self.cy = msg.k[2], msg.k[5]
                self.get_logger().info(f"Cámara {self.img_w}x{self.img_h} "
                                       f"fx={self.fx:.1f} cx={self.cx:.1f}")

        def _on_image(self, msg: Image):
            with self._lock:
                if self.fx is None:
                    return
                cam = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                       "cam_x": self.cam_x, "cam_y": self.cam_y, "cam_z": self.cam_z}
                img_wh = (self.img_w, self.img_h)
                depth = self.latest_depth if (args.use_depth or args.depth_refine or args.save_depth) else None
            now = time.monotonic()
            if now - self.last_cap < self.rate_interval or now < self.settle_until:
                return
            self.last_cap = now

            sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            frame = imgmsg_to_cv2(msg)
            self.frame_in_seq += 1
            fr = self.frame_in_seq

            rows = compute_frame_gt(self.actors, self.id_map, self.robot_xyz,
                                    self.robot_yaw, cam, sim_time, img_wh,
                                    depth=depth, occ_margin=args.occ_margin,
                                    person_height=args.person_height, ground_z=args.ground_z,
                                    box_config=box_config, half_width=args.actor_half_width,
                                    depth_refine=args.depth_refine, refine_margin=args.refine_margin, back_margin=args.back_margin)
            for (tid, l, t, w, h) in rows:
                self.lines.append(f"{fr},{tid},{l},{t},{w},{h},1,1,1")

            cv2.imwrite(str(self.seq_dir / "img1" / f"{fr:06d}.jpg"), frame)

            # Modo depuración: guarda profundidad (PNG 16-bit, mm) + metadatos por
            # frame para reaplicar/afinar el refinado OFFLINE (tune_refine.py) sin
            # tener que regenerar el dataset cada vez.
            if args.save_depth and depth is not None:
                import numpy as np
                ddir = self.seq_dir / "depth"; ddir.mkdir(exist_ok=True)
                dmm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                cv2.imwrite(str(ddir / f"{fr:06d}.png"),
                            np.clip(dmm * 1000.0, 0, 65535).astype(np.uint16))
                with open(self.seq_dir / "meta.jsonl", "a") as mf:
                    mf.write(json.dumps({"frame": fr, "sim_time": sim_time,
                        "robot": [self.robot_xyz[0], self.robot_xyz[1], self.robot_yaw],
                        "cam": cam, "wh": list(img_wh)}) + "\n")

            if args.draw:
                vis = frame.copy()
                for (tid, l, t, w, h) in rows:
                    color = _ID_COLORS[(tid - 1) % len(_ID_COLORS)]
                    cv2.rectangle(vis, (l, t), (l + w, t + h), color, 2)
                    cv2.putText(vis, f"ID:{tid}", (l, max(0, t - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                vdir = self.seq_dir / "verify"
                vdir.mkdir(exist_ok=True)
                cv2.imwrite(str(vdir / f"{fr:06d}.jpg"), vis)

            if fr >= args.frames_per_seq:
                self._next_viewpoint()

    rclpy.init()
    node = MOTCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.seq_dir is not None and node.frame_in_seq > 0:
            node._finalize_seq()
        print(f"\nDataset MOT generado en: {out_root}")
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


def main():
    p = argparse.ArgumentParser(description="Generador de dataset de tracking multi-ID (MOTChallenge)")
    p.add_argument("--selftest", action="store_true", help="Valida la lógica offline (sin Gazebo)")
    p.add_argument("--sdf", help="SDF del mundo con <actor>")
    p.add_argument("--output_dir", help="Directorio de salida del dataset")
    p.add_argument("--viewpoints", default="",
                   help="Viewpoints estáticos del robot: 'x,y,yaw_deg  x,y,yaw_deg ...'")
    p.add_argument("--frames_per_seq", type=int, default=150, help="Frames por secuencia")
    p.add_argument("--rate", type=float, default=10.0, help="Tasa de captura (Hz)")
    p.add_argument("--settle_time", type=float, default=1.5, help="Espera tras teleport (s)")
    p.add_argument("--world", default="warehouse_unified", help="Nombre del mundo en Gazebo")
    p.add_argument("--robot_model", default="turtlebot4", help="Nombre del robot en Gazebo")
    p.add_argument("--camera_ns", default="/oakd/rgb/preview", help="Namespace del topic de cámara")
    p.add_argument("--cam_x", type=float, default=-0.037, help="Offset X cámara desde base_link")
    p.add_argument("--cam_z", type=float, default=0.19, help="Offset Z cámara desde base_link")
    p.add_argument("--no_teleport", action="store_true",
                   help="No teletransporta el robot; usa su pose real declarada con --robot_x/y/yaw_deg (1 secuencia)")
    p.add_argument("--robot_x", type=float, default=0.0, help="Pose real X del robot (con --no_teleport)")
    p.add_argument("--robot_y", type=float, default=0.0, help="Pose real Y del robot (con --no_teleport)")
    p.add_argument("--robot_yaw_deg", type=float, default=0.0, help="Yaw real del robot en grados (con --no_teleport)")
    p.add_argument("--draw", action="store_true",
                   help="Guarda copias con las cajas e IDs dibujados en seqNN/verify/ para QA visual")
    p.add_argument("--actor_half_width", type=float, default=None,
                   help="Semiancho del box 3D del actor en m (def. 0.25; prueba 0.18 para cajas más ceñidas)")
    p.add_argument("--actor_bottom_offset", type=float, default=None,
                   help="Distancia de la pose (pelvis) a los pies en m (def. 1.0)")
    p.add_argument("--actor_top_offset", type=float, default=None,
                   help="Distancia de la pose (pelvis) a la cabeza en m (def. 0.77)")
    p.add_argument("--use_depth", action="store_true",
                   help="Filtra actores ocluidos usando la imagen de profundidad (recomendado)")
    p.add_argument("--depth_topic", default="/oakd/rgb/preview/depth", help="Topic de profundidad (32FC1)")
    p.add_argument("--occ_margin", type=float, default=0.5,
                   help="Margen (m): si lo medido está más cerca que (dist_actor - margen), se descarta por oclusión")
    p.add_argument("--person_height", type=float, default=1.75,
                   help="Altura del actor en m para la caja (def. 1.75)")
    p.add_argument("--ground_z", type=float, default=0.0,
                   help="Z del suelo donde pisan los actores (def. 0.0)")
    p.add_argument("--box_config", default=None,
                   help="JSON actor->{half_width,person_height} de calibrate_actor_boxes.py (box por malla)")
    p.add_argument("--depth_refine", action="store_true",
                   help="Refina cada caja a la silueta real del actor con la profundidad (cajas centradas, sin fantasmas). Implica --use_depth.")
    p.add_argument("--save_depth", action="store_true",
                   help="Guarda profundidad (PNG 16-bit mm) + meta.jsonl por frame para afinar el refinado offline con tune_refine.py")
    p.add_argument("--back_margin", type=float, default=0.35,
                   help="Margen de profundidad HACIA ATRAS (m) desde la superficie frontal del actor: cubre el fondo del cuerpo (~0.35) sin coger la estanteria de detras")
    p.add_argument("--refine_margin", type=float, default=0.15,
                   help="Tolerancia de profundidad (m) para considerar un píxel parte del actor")
    args, _ = p.parse_known_args()

    apply_actor_box(args.actor_half_width, args.actor_bottom_offset, args.actor_top_offset)

    if args.selftest:
        return selftest()
    if not args.sdf or not args.output_dir:
        p.error("se requieren --sdf y --output_dir (o usa --selftest)")
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
