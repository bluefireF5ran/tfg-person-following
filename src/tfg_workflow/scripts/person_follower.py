#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# safe_person_following_node: safe person-following controller for TurtleBot4
# with integrated real-time telemetry to Weights & Biases (W&B) and direct
# Gazebo physics-level velocity injection to prevent firmware locks.

import argparse
import math
import sys
from enum import Enum

import rclpy
import rclpy.parameter
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
from vision_msgs.msg import Detection2DArray
from irobot_create_msgs.action import Undock

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────
# Resolución por defecto de la cámara OAK-D del TurtleBot4 en simulación.
# Configurable por CLI: la cámara se capturó a 640x480 para la fase de tracking.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HFOV_RAD = math.radians(69.0)

# Altura (px) del bounding box de una persona a la distancia de consigna (respaldo si
# no hay profundidad). Escala con la resolución (~400 px a 640x480).
BBOX_H_AT_TARGET = 400.0

# Fijado del objetivo por proximidad+permanencia (spec de Eugenio).
ACQUIRE_DWELL_S = 3.0        # s que la persona más cercana debe mantenerse para fijarse
ACQUIRE_MAX_RANGE = 2.0      # m: solo se considera objetivo si está a <= esta distancia
ACQUIRE_STILL_TOL_M = 0.7    # m: tolerancia de "quieto" en profundidad (la prof. medida baja
                             # al centrar un objetivo que estaba en el borde: ~range·cos θ)

# Adquisición tolerante a huecos POR TIEMPO (no por nº de frames): el candidato puede
# parpadear (queda en el borde del FOV y YOLO lo pierde a ratos); se mantiene la cuenta de
# permanencia mientras lleve menos de ACQ_GAP_S sin verse.
ACQ_GAP_S = 4.0              # s sin candidato en rango antes de reiniciar la cuenta

# Giro de centrado durante la adquisición: con detección fresca gira hacia el candidato
# (capado) para traerlo al CENTRO; en los huecos el giro DECAE a 0 (coast) para no
# sobrepasarlo con un cx obsoleto. (Antes se congelaba a 0 y, como el objetivo se cazaba en
# el borde del fotograma, nunca llegaba a centrarse → no fijaba nunca.)
ACQ_TURN_MAX = 0.6           # rad/s máx. de centrado mientras se adquiere
ACQ_COAST_S = 1.2            # s tras la última detección en que aún se centra (decayendo)

# Mantenimiento del objetivo (tracking visual por id) ante cruces/oclusiones.
# CLAVE para "no saltar al que se cruza" (spec de Eugenio): ante un hueco del id del objetivo
# NO se re-vincula de inmediato (eso permitía que un cruzante en ese frame robara el objetivo).
# Se COAST hasta que el id REAPAREZCA; solo si la ausencia se SOSTIENE > REBIND_MIN_LOST_S se
# plantea un re-vínculo, y SOLO a un candidato que esté QUIETO en la posición prevista.
# DISCRIMINADOR objetivo-vs-cruzante: el objetivo (al que seguimos centrado) se queda ~quieto
# en el punto previsto; un cruzante BARRE ese punto en 1-2 frames. Por eso no basta con que el
# candidato esté en el gate de cx/profundidad: debe PERMANECER ~quieto (dentro de ±DRIFT_PX de
# donde apareció) durante PERSIST_FRAMES consecutivos. Así un cruzante lento (que a 0.36 m/s y
# 3 m son ~5 px/frame, segundos dentro del gate) ya no roba el objetivo.
REBIND_MIN_LOST_S = 1.5      # s que el id debe llevar ausente antes de plantear re-vínculo
REBIND_CX_PX = 90            # px: ventana lateral alrededor de la última posición del objetivo
REBIND_DEPTH_M = 0.5         # m: ...y profundidad similar (un cruzante a otra dist. se ignora)
REBIND_PERSIST_FRAMES = 4    # frames consecutivos ~quietos en el punto previsto para aceptar
REBIND_DRIFT_PX = 8          # px: deriva máx. respecto al ancla para contar como "quieto"

# Pérdida REAL del objetivo fijado: ausente más de esto (agotados el coast y el re-vínculo) →
# soltar el fijado y volver a adquirir. Holgado para que un cruce normal (1-3 s) NO la dispare.
FOLLOW_LOST_TIMEOUT = 6.0

# Distancia al objetivo por LÁSER (instrucción de Eugenio + método del PDF relacionLaserImagen):
# la coordenada X de la caja se traduce a un ÁNGULO del láser (centro imagen = 0°) vía el FOV y
# se lee el rango del /scan en ese ángulo (`_lidar_range_at_pixel`). Se ACEPTA ese rango como
# distancia a la persona solo si CORROBORA la profundidad de cámara (|láser - prof| <= esto):
# es la asunción del PDF ("el haz incide en el objeto visto"). Si discrepa (en sim el LiDAR 2D
# atraviesa al actor sin colisión y mide la PARED de detrás) o es infinito, se cae a profundidad
# OAK-D. Así el láser es el método principal (robot real) y la profundidad el respaldo (sim).
LASER_AGREE_M = 0.5

# Evitación activa de obstáculos (LiDAR; las personas no las ve el LiDAR en sim).
AVOID_GAIN = 0.8             # rad/s de sesgo de giro hacia el lado más despejado
OBSTACLE_SLOW_DIST = 1.5     # m: distancia frontal a la que empieza el frenado progresivo (->0 en safe_dist)

# Guarda de RETROCESO (P2): el LiDAR 2D del TB4 no resumía la trasera, así que los retrocesos
# (mantener distancia si el objetivo queda más cerca que la consigna, o el desatasco de STOPPED)
# eran CIEGOS y podían chocar con un obstáculo de detrás. Se limita la velocidad de retroceso y
# NO se retrocede si la trasera está bloqueada (en su lugar se queda quieto / gira en el sitio).
MAX_REVERSE_SPEED = 0.25     # m/s: tope de velocidad hacia atrás (movimiento ciego = lento)
REAR_SAFE_M = 0.5            # m: holgura trasera mínima para permitir retroceder
REVERSE_DEADBAND = 0.5       # m: si el objetivo está <esto más cerca que la consigna, NO retroceder


class State(Enum):
    SEARCHING = 1
    FOLLOWING = 2
    STOPPED = 3
    ACQUIRING = 4


class PersonFollowerNode(Node):
    def __init__(self, safe_dist, target_dist, max_linear, max_angular, lost_timeout, skip_undock=True, enable_ramping=False, search_angular=1.2, cmd_topic="/cmd_vel",
                 acquire_dwell_s=ACQUIRE_DWELL_S, acquire_max_range=ACQUIRE_MAX_RANGE, avoid_dist=0.8):
        super().__init__("person_follower",
                         parameter_overrides=[
                             rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True),
                         ])

        self.safe_dist = safe_dist
        self.target_dist = target_dist
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.lost_timeout = lost_timeout
        self.search_angular = search_angular   # rad/s de giro en SEARCHING
        self.cmd_topic = cmd_topic
        self.acquire_dwell_s = acquire_dwell_s
        self.acquire_max_range = acquire_max_range
        self.avoid_dist = avoid_dist
        self.min_angular = 0.08
        self.bbox_scale = 1.0
        self.enable_ramping = enable_ramping

        self.declare_parameter("use_wandb", True)
        self.declare_parameter("wandb_project", "tfg_turtlebot_follower")
        self.declare_parameter("wandb_run_name", "visual_servo_hsv")
        self.declare_parameter("wandb_tag", "test")
        self.declare_parameter("sim_obstacle", False)

        self.use_wandb = self.get_parameter("use_wandb").get_parameter_value().bool_value
        self.wandb_project = self.get_parameter("wandb_project").get_parameter_value().string_value
        self.wandb_run_name = self.get_parameter("wandb_run_name").get_parameter_value().string_value
        self.wandb_tag = self.get_parameter("wandb_tag").get_parameter_value().string_value

        self.state = State.SEARCHING
        self.last_detection_time = self.get_clock().now()
        self.frontal_min_range = float("inf")
        self._stuck_since = None

        self.current_linear = 0.0
        self.current_angular = 0.0

        # Medidas del objetivo/candidato para el lazo de control.
        self._det_cx = IMAGE_WIDTH / 2.0
        self._det_bbox_h = 0.0
        self._det_dist = 0.0        # distancia (m) al objetivo (láser corroborado, o profundidad)
        self._dist_source = 'depth' # 'laser' (método del PDF) | 'depth' (respaldo, actor sin colisión)
        self._est_dist = 0.0        # distancia usada por el control (para log/telemetría/banco)
        self._det_count = 0
        self._locked = False        # True solo cuando el objetivo fijado está visible

        # Estado de fijado/seguimiento del objetivo (spec de Eugenio).
        self.target_id = None       # id BoT-SORT del objetivo fijado (None = adquiriendo)
        self._acq_since = None      # instante en que empezó el dwell del candidato
        self._acq_cx = IMAGE_WIDTH / 2.0   # X del candidato (continuidad espacial, no por id)
        self._acq_ref_depth = None  # profundidad ancla para el chequeo de "quieto"
        self._acq_last_seen = None  # último instante con candidato en rango (gracia por tiempo)
        self._tgt_lost_since = None # instante en que el id del objetivo dejó de verse (coast/re-vínculo)
        self._tgt_cx = None         # última X conocida del objetivo (re-vínculo espacial)
        self._tgt_depth = None      # última profundidad conocida del objetivo
        # Candidato a re-vínculo en evaluación (debe permanecer ~quieto unos frames).
        self._rebind_id = None
        self._rebind_anchor_cx = None
        self._rebind_frames = 0

        # Historial de movimiento del objetivo fijado (para predicción de posición durante
        # oclusiones/cruces cercanos). Se actualiza en cada frame visible.
        self._tgt_cx_history = []   # (timestamp_ns, cx)
        self._tgt_depth_history = []   # (timestamp_ns, depth)

        # Histéresis para STOPPED: entrar a safe_dist, salir a safe_dist * 1.5
        self._obstacle_recovery = safe_dist * 1.5
        # Cooldown para warnings del filtro de profundidad (evitar spam)
        self._last_depth_warn_ns = 0

        self._scan_msg = None
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.rear_min = float("inf")    # holgura trasera (para no retroceder a ciegas, P2)

        if self.use_wandb and WANDB_AVAILABLE:
            self.get_logger().info("Inicializando Weights & Biases...")
            # W&B es OPCIONAL: si no hay API key (p. ej. al clonar el repo en otra máquina sin
            # `wandb login`), wandb.init lanza UsageError. Lo capturamos para NO matar el nodo:
            # el seguimiento debe funcionar igual, solo sin telemetría.
            try:
                wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    tags=[self.wandb_tag],
                    config={
                        "safe_dist": self.safe_dist,
                        "target_dist": self.target_dist,
                        "max_linear": self.max_linear,
                        "max_angular": self.max_angular,
                        "lost_timeout": self.lost_timeout,
                        "bbox_scale": self.bbox_scale,
                        "enable_ramping": self.enable_ramping,
                    }
                )
                self.get_logger().info("Dashboard de telemetria W&B conectado!")
            except Exception as e:
                self.use_wandb = False
                self.get_logger().warn(
                    f"W&B desactivado (no se pudo inicializar: {e}). El seguimiento continúa "
                    f"sin telemetría. Para habilitarlo: `wandb login` o lanzar con use_wandb:=false.")
        elif self.use_wandb and not WANDB_AVAILABLE:
            self.get_logger().warn("Libreria 'wandb' no encontrada; se continúa sin telemetría.")

        # Publicar en /cmd_vel (entrada de motion_control del Create3), NO en
        # /diffdrive_controller/cmd_vel: publicar ahí pelea con los ceros que motion_control
        # emite continuamente para mantener el robot parado -> el robot se quedaba estático.
        # Con safety_override=full, motion_control reenvía /cmd_vel sin limitar.
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        # Publica el id del objetivo fijado (o -1) para que el detector lo resalte en verde.
        self.target_id_pub = self.create_publisher(Int32, "/follow_target_id", 10)

        self.create_subscription(Detection2DArray, "/person_detections", self._on_detections, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)

        self.timer = self.create_timer(0.1, self._control_loop)

        self._configure_system_params()

        self._undock_done = False
        self._skip_undock = skip_undock
        if skip_undock:
            self.get_logger().info("Omitiendo undock (robot spawned off-dock)")
            self._undock_done = True
        else:
            self._undock_client = ActionClient(self, Undock, "/undock")
            self._try_undock()

        self.get_logger().info(
            f"PersonFollower listo -- safe={safe_dist}m, target={target_dist}m, "
            f"max_v={max_linear}m/s, max_w={max_angular}rad/s"
        )

    def _try_undock(self):
        self.get_logger().info("Esperando al servidor de accion /undock (hasta 60s)...")
        if not self._undock_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().warn("Servidor /undock no disponible, forzando control directo...")
            self._undock_done = True
            return
        self.get_logger().info("Enviando goal de desatracado de forma asincrona...")
        future = self._undock_client.send_goal_async(Undock.Goal())
        future.add_done_callback(self._undock_response_cb)

    def _undock_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal de undock rechazado por la base (asumiendo ya desatracado)")
            self._undock_done = True
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._undock_result_cb)

    def _undock_result_cb(self, future):
        self.get_logger().info('Desatracado completado con exito! Iniciando lazo de control sincrono.')
        self._undock_done = True

    def _configure_system_params(self):
        # Parámetros del sistema a fijar. EL CLAVE: safety_override='full' en el
        # motion_control del Create3 -> sin él, la seguridad capa la velocidad lineal a
        # 0.306 m/s (los actores andan a ~0.36 m/s, el robot no los alcanza nunca) y mete
        # reflejos que frenan/bloquean la base. Con 'full' sube a 0.46 m/s y sin frenos.
        # Se reintenta porque /motion_control suele tardar en publicar su servicio.
        from rcl_interfaces.msg import ParameterType
        self._pending_params = [
            ('/motion_control', 'safety_override', 'full', ParameterType.PARAMETER_STRING),
            ('/motion_control', 'reflexes_enabled', False, ParameterType.PARAMETER_BOOL),
            ('/diffdrive_controller', 'cmd_vel_timeout', 5.0, ParameterType.PARAMETER_DOUBLE),
        ]
        self._param_clients = {}
        self._param_retries = 0
        self._param_timer = self.create_timer(2.0, self._try_set_pending_params)

    def _try_set_pending_params(self):
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter as RosParam
        from rcl_interfaces.msg import ParameterValue
        from rcl_interfaces.msg import ParameterType

        self._param_retries += 1
        still = []
        for (ns, name, value, ptype) in self._pending_params:
            client = self._param_clients.get(ns) or self.create_client(SetParameters, f'{ns}/set_parameters')
            self._param_clients[ns] = client
            if not client.service_is_ready():
                still.append((ns, name, value, ptype))
                continue
            pv = ParameterValue(type=ptype,
                                string_value=value if ptype == ParameterType.PARAMETER_STRING else '',
                                double_value=value if ptype == ParameterType.PARAMETER_DOUBLE else 0.0,
                                bool_value=value if ptype == ParameterType.PARAMETER_BOOL else False)
            req = SetParameters.Request()
            req.parameters = [RosParam(name=name, value=pv)]
            client.call_async(req).add_done_callback(
                lambda f, n=ns, p=name, v=value: self.get_logger().info(f'{n}: {p} -> {v}')
                if (f.result() and f.result().results[0].successful)
                else self.get_logger().warn(f'{n}: fallo al setear {p}'))
        self._pending_params = still
        if not still:
            self.get_logger().info("Parametros del sistema OK (Create3 safety_override=full).")
            self._param_timer.cancel()
        elif self._param_retries >= 30:   # ~60 s
            self.get_logger().warn(f"No se pudieron fijar: {[p[1] for p in still]} (servicios no disponibles)")
            self._param_timer.cancel()

    def _on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        self._scan_msg = msg
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        front_min = float("inf")
        left_min = float("inf")     # +30..+90 grados (holgura a la izquierda)
        right_min = float("inf")    # -90..-30 grados (holgura a la derecha)
        rear_min = float("inf")     # |angle| >= 90 grados (TODA la mitad trasera; sin hueco ciego)
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            angle = angle_min + i * angle_inc
            if abs(angle) < math.radians(30):
                front_min = min(front_min, r)
            elif math.radians(30) <= angle < math.radians(90):
                left_min = min(left_min, r)
            elif math.radians(-90) < angle <= math.radians(-30):
                right_min = min(right_min, r)
            elif abs(angle) >= math.radians(90):
                rear_min = min(rear_min, r)
        self.frontal_min_range = front_min
        self.left_min = left_min
        self.right_min = right_min
        self.rear_min = rear_min

    def _lidar_range_at_pixel(self, pixel_x: float) -> float:
        if self._scan_msg is None:
            return float("inf")
        msg = self._scan_msg
        norm_x = (pixel_x - IMAGE_WIDTH / 2.0) / (IMAGE_WIDTH / 2.0)
        person_angle = -norm_x * (HFOV_RAD / 2.0)

        idx = round((person_angle - msg.angle_min) / msg.angle_increment)
        n = len(msg.ranges)
        best = float("inf")
        for offset in range(-5, 6):
            j = idx + offset
            if 0 <= j < n:
                r = msg.ranges[j]
                if msg.range_min < r < msg.range_max:
                    best = min(best, r)
        return best

    @staticmethod
    def _parse_id(class_id: str) -> int:
        """'person:7' -> 7."""
        try:
            return int(class_id.split(":")[1])
        except (IndexError, ValueError):
            return -1

    def _reset_rebind(self):
        self._rebind_id = None
        self._rebind_anchor_cx = None
        self._rebind_frames = 0

    def _predict_target_cx(self, now_ns):
        """Predice la posición X del objetivo basada en su velocidad reciente.
        Si hay suficiente historial, extrapola linealmente; si no, usa la última conocida."""
        if len(self._tgt_cx_history) < 2 or self._tgt_cx is None:
            return self._tgt_cx
        t0, cx0 = self._tgt_cx_history[0]
        t1, cx1 = self._tgt_cx_history[-1]
        dt = (t1 - t0) * 1e-9
        if dt < 0.01:
            return cx1
        vel = (cx1 - cx0) / dt  # px/s
        dt_pred = (now_ns - t1) * 1e-9
        predicted = cx1 + vel * dt_pred
        return max(0.0, min(float(IMAGE_WIDTH), predicted))

    def _rebind_candidate(self, persons, lost_id_secs):
        """Re-vínculo del objetivo perdido SOLO a un candidato que esté QUIETO en la posición
        PREVISTA varios frames consecutivos (rechaza cruzantes, que barren ese punto). 
        Usa PREDICCIÓN de movimiento para centrar el gate (el objetivo no se queda quieto
        mientras está ocluido), y tolera MÚLTIPLES candidatos cercanos (cruce de actores):
        elige al más cercano al centro previsto y que tenga la profundidad más parecida.
        Devuelve la persona a la que re-vincular, o None (seguir coasteando)."""
        if lost_id_secs < REBIND_MIN_LOST_S or self._tgt_cx is None:
            return None
        # Centro del gate = posición PREDICHA por velocidad reciente (no la última conocida)
        now_ns = self.get_clock().now().nanoseconds
        pred_cx = self._predict_target_cx(now_ns)
        # P1: escalar la tolerancia de profundidad con el tiempo perdido (crece más lento
        # que la escala original: /4.5 en vez de /3.0, para no abrir el gate a profundidades
        # radicalmente distintas incluso tras oclusiones largas).
        depth_gate = REBIND_DEPTH_M * (1.0 + lost_id_secs / 4.5)
        # Gate espacial también se escala ligeramente con el tiempo perdido
        cx_gate = REBIND_CX_PX + int(lost_id_secs * 20)  # +20 px/s perdido
        # PASE 1: gate normal con profundidad (discrimina cruzantes a distinta distancia)
        near = [p for p in persons
                if abs(p['cx'] - pred_cx) <= cx_gate
                and (self._tgt_depth is None
                     or abs(p.get('dist', p['depth']) - self._tgt_depth) <= depth_gate)]
        # PASE 2 (fallback con gate ampliado, R3): si el gate de profundidad rechazó a todos
        # pero hay personas en un gate espacial+profundidad más amplio, reintentar.
        # El objetivo puede reaparecer a profundidad distinta si se alejó/acercó durante la
        # oclusión (cruce cercano). Se amplía el depth gate un 40% (1.4x) para cubrir el
        # movimiento del objetivo sin aceptar profundidades radicalmente distintas.
        fallback_mode = False
        if not near and self._tgt_depth is not None:
            cx_gate2 = cx_gate + 40
            depth_gate2 = depth_gate * 1.4
            near = [p for p in persons
                    if abs(p['cx'] - pred_cx) <= cx_gate2
                    and abs(p.get('dist', p['depth']) - self._tgt_depth) <= depth_gate2]
            near.sort(key=lambda p: abs(p['cx'] - pred_cx))
            if near:
                fallback_mode = True
        if not near:
            # Nadie en la zona prevista: reset y coast.
            self._reset_rebind()
            return None
        # Elegir al de profundidad más parecida, luego cx.
        near.sort(key=lambda p: (abs(p.get('dist', p['depth']) - (self._tgt_depth or 0)),
                                 abs(p['cx'] - pred_cx)))
        c = near[0]
        # Persistencia + quietud (±DRIFT del ancla respecto a la predicción).
        # En fallback (gate ampliado) se requiere el doble de persistencia.
        required = REBIND_PERSIST_FRAMES * (2 if fallback_mode else 1)
        if (self._rebind_id == c['id'] and self._rebind_anchor_cx is not None
                and abs(c['cx'] - self._rebind_anchor_cx) <= REBIND_DRIFT_PX + 4):
            self._rebind_frames += 1
        else:
            # Cambió el candidato, o se fue del drift: reiniciar.
            self._rebind_id = c['id']
            self._rebind_anchor_cx = c['cx']
            self._rebind_frames = 1
        return c if self._rebind_frames >= required else None

    def _on_detections(self, msg: Detection2DArray):
        """Adquisición por proximidad+permanencia y mantenimiento del objetivo por id
        (tracking visual), según la spec de Eugenio. La distancia viene de la profundidad
        OAK-D que publica el detector en results[0].pose.pose.position.z."""
        now = self.get_clock().now()
        persons = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            depth = hyp.pose.pose.position.z      # distancia (m) por profundidad OAK-D
            h = det.bbox.size_y
            # Distancia usada: PROFUNDIDAD si la hay; si no (depth=0 por cortes de la OAKD),
            # respaldo por tamaño del bbox para no bloquear el seguimiento.
            if depth > 0.05:
                dist = depth
            else:
                ratio = h / (BBOX_H_AT_TARGET * self.bbox_scale)
                dist = self.target_dist / max(ratio, 0.01)
            persons.append({
                'id': self._parse_id(hyp.hypothesis.class_id),
                'cx': det.bbox.center.position.x,
                'cy': det.bbox.center.position.y,
                'h': h, 'depth': depth, 'dist': dist,
            })
        self._det_count = len(persons)

        # ── FASE DE ADQUISICIÓN: aún no hay objetivo fijado ──────────────
        # Se fija a la persona más cercana (<acquire_max_range) que se mantenga QUIETA
        # ~acquire_dwell_s. Se sigue al candidato por CONTINUIDAD ESPACIAL (cx), no por id
        # BoT-SORT (que cambia), y "quieto" = profundidad estable (deja de acercarse).
        if self.target_id is None:
            self._locked = False
            in_range = [p for p in persons if 0.05 < p['dist'] <= self.acquire_max_range]
            if not in_range:
                # Hueco: mantener la cuenta de permanencia mientras el candidato lleve menos
                # de ACQ_GAP_S sin verse (parpadeos del borde del FOV / fallos puntuales de YOLO).
                if (self._acq_since is not None and self._acq_last_seen is not None
                        and (now - self._acq_last_seen).nanoseconds * 1e-9 > ACQ_GAP_S):
                    self._acq_since = None
                return
            nearest = min(in_range, key=lambda p: p['dist'])
            # El candidato es visible → el robot lo centra mientras corre la cuenta.
            self._det_cx, self._det_bbox_h, self._det_dist = nearest['cx'], nearest['h'], nearest['dist']
            self.last_detection_time = now
            # ¿Continúa la MISMA persona? Continuidad espacial (cx) + quietud en profundidad,
            # con referencia que DERIVA suave: al centrarlo, el cx y la profundidad cambian algo
            # (el id BoT-SORT no se exige, que cambia en vivo).
            consistent = (self._acq_since is not None
                          and abs(nearest['cx'] - self._acq_cx) < 220
                          and abs(nearest['dist'] - self._acq_ref_depth) < ACQUIRE_STILL_TOL_M)
            if consistent:
                self._acq_cx = 0.5 * self._acq_cx + 0.5 * nearest['cx']
                self._acq_ref_depth = 0.7 * self._acq_ref_depth + 0.3 * nearest['dist']
                self._acq_last_seen = now
                if (now - self._acq_since).nanoseconds * 1e-9 >= self.acquire_dwell_s:
                    self.target_id = nearest['id']             # ¡FIJADO!
                    self._tgt_cx, self._tgt_depth, self._tgt_lost_since = nearest['cx'], nearest['dist'], None
                    now_ns = now.nanoseconds
                    self._tgt_cx_history = [(now_ns, nearest['cx'])]
                    self._tgt_depth_history = [(now_ns, nearest['dist'])]
                    self._reset_rebind()
                    src = "profundidad OAK-D" if nearest['depth'] > 0.05 else "tamaño bbox (sin depth)"
                    self.get_logger().info(
                        f"Objetivo fijado id={self.target_id} a {nearest['dist']:.2f} m "
                        f"({src}, tras {self.acquire_dwell_s:.0f}s en el campo de visión)")
            else:
                # Nuevo candidato / saltó a otra persona → reiniciar la cuenta.
                self._acq_since = now
                self._acq_last_seen = now
                self._acq_cx = nearest['cx']
                self._acq_ref_depth = nearest['dist']
            return

        # ── FASE DE SEGUIMIENTO: objetivo fijado por id ──────────────────
        by_id = {p['id']: p for p in persons}
        tgt = by_id.get(self.target_id)
        if tgt is None:
            # El id del objetivo NO aparece este frame. Lo normal es un parpadeo de 1-2 frames
            # (churn de BoT-SORT / YOLO falla un instante): NO re-vincular de inmediato —si lo
            # hiciéramos, un cruzante que pase por delante en ese hueco nos robaría el objetivo
            # (justo el fallo a corregir). En su lugar COAST y dejar que el id REAPAREZCA solo;
            # y si hay que re-vincular, solo a quien permanezca QUIETO en el punto previsto.
            if self._tgt_lost_since is None:
                self._tgt_lost_since = now
            lost_id_secs = (now - self._tgt_lost_since).nanoseconds * 1e-9
            cand = self._rebind_candidate(persons, lost_id_secs)
            if cand is not None:
                self.target_id = cand['id']
                tgt = cand
                self.get_logger().info(
                    f"Objetivo re-vinculado a id={self.target_id} (quieto en la posición "
                    f"prevista, tras {lost_id_secs:.1f}s sin verlo)")
            else:
                # P1 (cruce cercano de actores): durante el coast, usar la detección MÁS
                # CERCANA a la posición predicha para centrar el giro, no el cx obsoleto.
                # Así el robot se orienta hacia donde está la persona AUNQUE el id haya cambiado
                # por el cruce. Buscamos en un gate amplio (~200px, sin gate de profundidad)
                # para mantener la orientación hacia la zona del cruce mientras el objetivo
                # reaparece. NO rebindeamos — solo alineamos el giro para no perder más al objetivo.
                now_ns = self.get_clock().now().nanoseconds
                pred_cx = self._predict_target_cx(now_ns)
                best = None
                best_dist = float("inf")
                for p in persons:
                    d = abs(p['cx'] - (pred_cx if pred_cx is not None else (self._tgt_cx or IMAGE_WIDTH/2)))
                    if d < 200 and d < best_dist:
                        best_dist = d
                        best = p
                if best is not None:
                    self._det_cx = best['cx']
                self._locked = False     # coast: no actualizamos last_detection_time
                return
        # Objetivo visible (o re-vinculado)
        self._tgt_lost_since = None
        self._reset_rebind()
        self._locked = True
        self._det_cx, self._det_bbox_h = tgt['cx'], tgt['h']
        # DISTANCIA AL OBJETIVO POR LÁSER (instrucción de Eugenio + método del PDF): se traduce
        # la X de la caja a un ángulo del láser y se lee el rango del /scan ahí. Se usa ese
        # rango SOLO si corrobora la profundidad de cámara (el haz incide en la persona); si no
        # (en sim el LiDAR 2D atraviesa al actor sin colisión y mide la pared de detrás, o da
        # inf), se cae a la profundidad OAK-D. Láser = principal (robot real); prof. = respaldo.
        d_depth = tgt['dist']
        d_laser = self._lidar_range_at_pixel(tgt['cx'])
        if math.isfinite(d_laser) and abs(d_laser - d_depth) <= LASER_AGREE_M:
            self._det_dist, self._dist_source = d_laser, 'laser'
        else:
            self._det_dist, self._dist_source = d_depth, 'depth'
        # Filtro anti-salto brusco de profundidad: si la profundidad medida cambia >30%
        # respecto a la media reciente (el actor camina a ~0.36 m/s, no puede saltar 1.3m
        # en 2s), es que la OAK-D midió el fondo (pared/estantería) en vez de la persona.
        # En ese caso, respaldar con la estimación por tamaño del bbox.
        now_ns = now.nanoseconds
        if self._tgt_depth_history and len(self._tgt_depth_history) >= 2:
            avg_depth = sum(d for _, d in self._tgt_depth_history[-5:]) / min(len(self._tgt_depth_history[-5:]), 5)
            if avg_depth > 0.5 and tgt['dist'] > 0.5:
                ratio = tgt['dist'] / avg_depth
                if ratio > 1.3 or ratio < 0.7:
                    bbox_ratio = tgt['h'] / (BBOX_H_AT_TARGET * self.bbox_scale)
                    bbox_dist = self.target_dist / max(bbox_ratio, 0.01)
                    if 0.3 < bbox_dist < 10.0:
                        if (now_ns - self._last_depth_warn_ns) > 2_000_000_000:
                            self._last_depth_warn_ns = now_ns
                            self.get_logger().warn(
                                f"Profundidad saltó de {avg_depth:.1f}m a {tgt['dist']:.1f}m "
                                f"(ratio={ratio:.2f}) -> usando bbox ({bbox_dist:.1f}m)")
                        tgt['dist'] = bbox_dist
                        tgt['depth'] = bbox_dist
        self._tgt_cx = tgt['cx']
        self._tgt_depth = tgt['dist']
        # Historial de movimiento (últimos ~0.5s para estimar velocidad)
        self._tgt_cx_history.append((now_ns, tgt['cx']))
        self._tgt_depth_history.append((now_ns, tgt['dist']))
        cutoff = now_ns - 600_000_000  # 0.6s
        self._tgt_cx_history = [(t, c) for t, c in self._tgt_cx_history if t > cutoff]
        self._tgt_depth_history = [(t, d) for t, d in self._tgt_depth_history if t > cutoff]
        self.last_detection_time = now

    def _control_loop(self):
        if not self._undock_done:
            return

        now = self.get_clock().now()
        lost_secs = (now - self.last_detection_time).nanoseconds * 1e-9

        sim_obstacle = self.get_parameter("sim_obstacle").get_parameter_value().bool_value
        if sim_obstacle:
            self.frontal_min_range = 0.3

        person_visible = lost_secs < 0.5          # solo para la telemetría/log
        obstacle_close = (self.frontal_min_range < self.safe_dist if self.state != State.STOPPED
                          else self.frontal_min_range < self._obstacle_recovery)

        # Pérdida REAL del objetivo fijado: ausente más de FOLLOW_LOST_TIMEOUT (ya agotados el
        # coast y el re-vínculo) → soltar el fijado y volver a adquirir.
        if self.target_id is not None and not self._locked and lost_secs > FOLLOW_LOST_TIMEOUT:
            self.get_logger().info(
                f"Objetivo id={self.target_id} perdido ({lost_secs:.1f}s) → re-adquirir")
            self.target_id = None
            self._acq_since = None
            self._acq_last_seen = None
            self._tgt_lost_since = None
            self._tgt_cx_history.clear()
            self._tgt_depth_history.clear()
            self._reset_rebind()

        # Estado determinado por la SUB-MÁQUINA de objetivo (NO por lost_secs, que provocaba la
        # oscilación SEARCHING<->FOLLOWING). STOPPED prevalece si hay un obstáculo pegado.
        prev_state = self.state
        if obstacle_close:
            self.state = State.STOPPED
        elif self.target_id is not None:
            self.state = State.FOLLOWING        # objetivo fijado (la rama gestiona lock/coast)
        elif self._acq_since is not None:
            self.state = State.ACQUIRING        # candidato en curso → centrar + contar permanencia
        else:
            self.state = State.SEARCHING        # nadie en rango → barrer despacio

        if self.state != prev_state:
            self.get_logger().info(
                f"[STATE] {prev_state.name} -> {self.state.name} "
                f"(front={self.frontal_min_range:.2f}m, lost={lost_secs:.1f}s)"
            )
            self._stuck_since = now if self.state == State.STOPPED else None

        des_linear = 0.0
        des_angular = 0.0
        est_dist = 0.0
        error_x = 0.0

        if self.state == State.STOPPED:
            stuck_secs = (now - self._stuck_since).nanoseconds * 1e-9 if self._stuck_since else 0.0
            if stuck_secs > 3.0 and not sim_obstacle:
                # Desatasco SEGURO: girar EN EL SITIO para reorientarse (NO retroceder). El
                # retroceso de desatasco era CIEGO por detrás y chocaba (P2/log 19:25); rotando
                # se reencuadra el objetivo y se reanuda hacia delante cuando el frente despeja.
                des_linear = 0.0
                des_angular = 0.6
            else:
                des_linear = 0.0
                des_angular = 0.0

        elif self.state == State.SEARCHING:
            des_linear = 0.0
            des_angular = self.search_angular

        elif self.state == State.ACQUIRING or (self.state == State.FOLLOWING and not self._locked):
            # Adquiriendo un candidato, o el objetivo fijado momentáneamente no visible (coast):
            # CENTRAR (suave) hacia su última posición conocida y NO avanzar. El giro decae con
            # el tiempo sin detección (coast) para no sobrepasar al objetivo con un cx obsoleto;
            # con detección fresca trae al objetivo del borde al centro. Así se acumula la cuenta
            # de permanencia sin oscilar.
            error_x = (self._det_cx - IMAGE_WIDTH / 2.0) / (IMAGE_WIDTH / 2.0)
            coast = max(0.0, 1.0 - lost_secs / ACQ_COAST_S)
            des_angular = -error_x * self.max_angular * 1.5
            des_angular = max(-ACQ_TURN_MAX, min(ACQ_TURN_MAX, des_angular)) * coast
            des_linear = 0.0

        elif self.state == State.FOLLOWING:   # objetivo fijado y visible (_locked)
            # Centrado del objetivo (visual servoing).
            error_x = (self._det_cx - IMAGE_WIDTH / 2.0) / (IMAGE_WIDTH / 2.0)
            des_angular = -error_x * self.max_angular * 1.5
            if abs(error_x) > 0.03 and abs(des_angular) < self.min_angular:
                des_angular = math.copysign(self.min_angular, des_angular)

            # Distancia al objetivo: láser corroborado o profundidad (ver _on_detections);
            # bbox como último respaldo si no hubiera ni láser ni profundidad.
            if self._det_dist > 0.05:
                est_dist = self._det_dist
            else:
                bbox_ratio = self._det_bbox_h / (BBOX_H_AT_TARGET * self.bbox_scale)
                est_dist = self.target_dist / max(bbox_ratio, 0.01)
            self._est_dist = est_dist
            dist_error = est_dist - self.target_dist
            # BANDA MUERTA de retroceso: si el objetivo está solo un poco MÁS CERCA que la
            # consigna, NO retroceder (quedarse quieto); retroceder a ciegas por acercamientos
            # pequeños era innecesario y peligroso. Solo se retrocede si se acerca de verdad.
            if -REVERSE_DEADBAND < dist_error < 0.0:
                des_linear = 0.0
            else:
                des_linear = dist_error * self.max_linear
                if dist_error > 0.2:
                    des_linear = max(des_linear, 0.15)
                des_linear *= max(0.4, 1.0 - 0.3 * abs(error_x))

            # Evitación de ENTORNO (el LiDAR NO ve a las personas; lo cercano por delante es
            # pilar/pared/estantería). DOS capas:
            # (1) FRENADO PROGRESIVO: la velocidad hacia delante se limita en proporción a la
            #     holgura frontal, EMPEZANDO PRONTO (OBSTACLE_SLOW_DIST=1.5 m) y llegando a 0 en
            #     safe_dist. Antes solo se capaba a 0.12 en la franja 0.5-0.8 m -> a 0.6 m/s
            #     llegaba tarde y EMBESTÍA el pilar (log 19:45). Ahora frena con tiempo.
            if des_linear > 0.0 and self.frontal_min_range < OBSTACLE_SLOW_DIST:
                v_cap = self.max_linear * (self.frontal_min_range - self.safe_dist) \
                        / max(0.1, OBSTACLE_SLOW_DIST - self.safe_dist)
                des_linear = max(0.0, min(des_linear, v_cap))
            # (2) SESGO DE GIRO hacia el lado más despejado para RODEAR el obstáculo. Se escala
            #     con el descentrado del objetivo (prioridad: no sacarlo del FOV ~69°).
            if self.frontal_min_range < self.avoid_dist:
                avoid_w = AVOID_GAIN if self.left_min >= self.right_min else -AVOID_GAIN
                if self._locked:
                    avoid_w *= max(0.15, 1.0 - abs(error_x))
                des_angular += avoid_w

        # GUARDA DE RETROCESO (P2, "sin chocar"): cualquier retroceso (mantener distancia con un
        # objetivo más cercano que la consigna, o el desatasco de STOPPED) es CIEGO porque la
        # cámara mira al frente y el control no veía la trasera. Se limita la velocidad de
        # retroceso y, si la trasera está bloqueada, NO se retrocede (queda en 0; el giro se
        # mantiene para escapar girando en el sitio en vez de embestir hacia atrás).
        # Además, si el objetivo está descentrado mientras retrocedemos, reducir el retroceso
        # para no perder el tracking (el giro combinado con retroceso puede sacarlo del FOV).
        if des_linear < 0.0:
            des_linear = max(des_linear, -MAX_REVERSE_SPEED)
            if self._locked and abs(error_x) > 0.3:
                des_linear *= 0.5   # retroceso suave si el target ya no está centrado
            if self.rear_min < REAR_SAFE_M:
                des_linear = 0.0

        if self.enable_ramping:
            ramp_rate_lin = 0.8
            ramp_rate_ang = 2.0
            dt = 0.1
            self.current_linear = self._ramp(self.current_linear, des_linear, ramp_rate_lin * dt)
            self.current_angular = self._ramp(self.current_angular, des_angular, ramp_rate_ang * dt)
        else:
            self.current_linear = des_linear
            self.current_angular = des_angular

        self.current_linear = max(-self.max_linear, min(self.max_linear, self.current_linear))
        self.current_angular = max(-self.max_angular, min(self.max_angular, self.current_angular))

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()   # sello con hora (sim) actual
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = self.current_linear
        msg.twist.angular.z = self.current_angular
        self.cmd_pub.publish(msg)

        # Publicar el id del objetivo (o -1) para que el detector lo resalte.
        tid = Int32()
        tid.data = int(self.target_id) if self.target_id is not None else -1
        self.target_id_pub.publish(tid)

        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "linear_velocity_cmd": self.current_linear,
                "angular_velocity_cmd": self.current_angular,
                "bbox_height_px": float(self._det_bbox_h) if person_visible else 0.0,
                "estimated_distance_m": float(est_dist) if person_visible else 0.0,
                "target_locked": 1.0 if self._locked else 0.0,
                "centering_error_cx": float(error_x) if person_visible else 0.0,
                "min_lidar_front_m": float(self.frontal_min_range),
                "state_code": self.state.value,
                "lost_duration_sec": float(lost_secs),
                "dist_source_laser": 1.0 if (self._locked and self._dist_source == 'laser') else 0.0,
            })

        now_sec = now.nanoseconds * 1e-9
        if not hasattr(self, "_last_log_time") or (now_sec - self._last_log_time) >= 2.0:
            self._last_log_time = now_sec
            tgt = f"id{self.target_id}" if self.target_id is not None else "ninguno"
            src = ("láser" if self._dist_source == 'laser' else "prof") if self._locked else "-"
            self.get_logger().info(
                f"[{self.state.name}] v={self.current_linear:.2f} w={self.current_angular:.2f} | "
                f"front={self.frontal_min_range:.2f}m | objetivo={tgt} {'(lock)' if self._locked else ''} | "
                f"dist={est_dist:.2f}m ({src}) | cx_err={error_x:.2f}"
            )

    @staticmethod
    def _ramp(current: float, target: float, max_step: float) -> float:
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + math.copysign(max_step, diff)


def main():
    parser = argparse.ArgumentParser(description="Safe person-following controller for TurtleBot4")
    parser.add_argument("--safe-dist", type=float, default=0.5)
    parser.add_argument("--target-dist", type=float, default=1.5)
    parser.add_argument("--max-linear", type=float, default=1.0)
    parser.add_argument("--max-angular", type=float, default=1.5)
    parser.add_argument("--lost-timeout", type=float, default=3.0)
    parser.add_argument("--search-angular", type=float, default=1.2,
                        help="Velocidad de giro en la fase de busqueda (rad/s)")
    parser.add_argument("--cmd-topic", default="/cmd_vel",
                        help="Topic de velocidad (def /cmd_vel; evita el conflicto con motion_control)")
    parser.add_argument("--acquire-dwell-s", type=float, default=ACQUIRE_DWELL_S,
                        help="Segundos que la persona mas cercana debe mantenerse para fijarse como objetivo")
    parser.add_argument("--acquire-max-range", type=float, default=ACQUIRE_MAX_RANGE,
                        help="Distancia maxima (m) para considerar a alguien como objetivo")
    parser.add_argument("--avoid-dist", type=float, default=0.8,
                        help="Distancia (m) a la que se empieza a esquivar obstaculos")
    parser.add_argument("--undock", action="store_true", default=False)
    parser.add_argument("--bbox-scale", type=float, default=1.0)
    parser.add_argument("--enable-ramping", action="store_true", default=False)

    rclpy.init(args=sys.argv)
    cleaned = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(cleaned)

    node = PersonFollowerNode(
        safe_dist=args.safe_dist,
        target_dist=args.target_dist,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        lost_timeout=args.lost_timeout,
        skip_undock=not args.undock,
        enable_ramping=args.enable_ramping,
        search_angular=args.search_angular,
        cmd_topic=args.cmd_topic,
        acquire_dwell_s=args.acquire_dwell_s,
        acquire_max_range=args.acquire_max_range,
        avoid_dist=args.avoid_dist,
    )
    node.bbox_scale = args.bbox_scale

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        # Carrera típica al cerrar (take_message mientras se destruye el contexto): no es
        # fatal, el cleanup del finally se ejecuta igual.
        try:
            node.get_logger().warn(f"spin terminado por excepción (probable cierre): {e}")
        except Exception:
            pass
    finally:
        try:
            if rclpy.ok():
                stop = TwistStamped()
                stop.header.stamp.sec = 0
                stop.header.stamp.nanosec = 0
                node.cmd_pub.publish(stop)
        except Exception:
            pass   # contexto ya invalidado por el SIGINT del launch

        if node.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
