#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# Banco de pruebas OFFLINE de la lógica de seguimiento (person_follower.py).
#
# MOTIVACIÓN: el ajuste fino del seguidor se hacía lanzando Gazebo una y otra vez (lento).
# Toda la lógica de decisión del follower (adquirir -> fijar -> mantener ante cruces ->
# re-adquirir) es función pura de (detecciones, reloj, LiDAR). Aquí se ejercita SIN Gazebo:
# se sintetizan mensajes Detection2DArray (mismo contrato que publica person_detector.py:
# bbox.center.x/y, bbox.size_x/y, results[0].hypothesis.class_id="person:{id}",
# results[0].pose.pose.position.z=profundidad) y se controla el reloj con un FakeClock.
#
# Cada escenario codifica un requisito de la spec del tutor como PASS/FAIL y vuelca un
# visor textual por frame (posiciones objetivo/cruzante, estado, target_id, v/w).
#
# USO (sin colcon, sin pytest; igual que el resto del TFG):
#     python3 src/tfg_workflow/scripts/test_follower_logic.py
#     python3 src/tfg_workflow/scripts/test_follower_logic.py --quiet   # solo resumen PASS/FAIL

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import rclpy
from rclpy.time import Time
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

SCRIPTS_DIR = Path(__file__).resolve().parent
FOLLOWER_PATH = SCRIPTS_DIR / "person_follower.py"


# ── Cargar person_follower.py como módulo (es un script, no un paquete instalado) ──
def _load_follower_module():
    spec = importlib.util.spec_from_file_location("person_follower_under_test", FOLLOWER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WANDB_AVAILABLE = False     # desactiva W&B (el constructor y el lazo lo comprueban)
    return mod


pf = _load_follower_module()


# ── Reloj controlable ───────────────────────────────────────────
# Con use_sim_time=True y sin /clock, el reloj real se queda en 0 y los dwell nunca corren.
# El nodo usa siempre self.get_clock().now(); lo sustituimos por este reloj que avanzamos a mano.
class FakeClock:
    def __init__(self):
        self.t_ns = 0

    def now(self) -> Time:
        return Time(nanoseconds=self.t_ns)

    def advance(self, dt_s: float):
        self.t_ns += int(round(dt_s * 1e9))


# ── Síntesis de detecciones (replica el contrato de person_detector.py:190-201) ──
def make_array(persons) -> Detection2DArray:
    """persons: lista de dicts {id, cx, depth, [cy], [h]} -> Detection2DArray."""
    arr = Detection2DArray()
    for p in persons:
        det = Detection2D()
        det.bbox.center.position.x = float(p["cx"])
        det.bbox.center.position.y = float(p.get("cy", 240.0))
        det.bbox.size_x = float(p.get("w", 180.0))
        det.bbox.size_y = float(p.get("h", 400.0))
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = f"person:{int(p['id'])}"
        hyp.pose.pose.position.z = float(p["depth"])
        det.results.append(hyp)
        arr.detections.append(det)
    return arr


# ── Banco: nodo follower real + reloj falso, conducido a mano (sin spin) ──
class Harness:
    def __init__(self):
        self.node = pf.PersonFollowerNode(
            safe_dist=0.5, target_dist=3.0, max_linear=0.6, max_angular=1.2,
            lost_timeout=3.0, skip_undock=True, search_angular=0.5,
            acquire_dwell_s=3.0, acquire_max_range=3.3, avoid_dist=1.2,
        )
        self.clock = FakeClock()
        self.node.get_clock = lambda: self.clock          # toda lectura de tiempo pasa por aquí
        # Base temporal limpia (el constructor leyó el reloj real -> reset a 0).
        self.node.last_detection_time = self.clock.now()
        self.node._acq_since = None
        self.node._acq_last_seen = None
        self.node._tgt_lost_since = None
        self.node.frontal_min_range = float("inf")
        self.trace = []

    def tick(self, dt, persons=None, front=float("inf"), laser_at_target=None, rear=float("inf")):
        """Avanza el reloj dt, (opcional) entrega detecciones y ejecuta un ciclo de control.
        `laser_at_target`: si se da, simula el rango del LiDAR en el ángulo de la caja del
        objetivo (relación del PDF) sustituyendo `_lidar_range_at_pixel`; por defecto el nodo
        devuelve inf (no hay /scan) y la distancia cae a profundidad, como hoy.
        `front`/`rear`: holgura LiDAR frontal/trasera (la trasera gatea el retroceso, P2)."""
        self.clock.advance(dt)
        self.node.frontal_min_range = front
        self.node.rear_min = rear
        if laser_at_target is not None:
            self.node._lidar_range_at_pixel = lambda cx, _v=laser_at_target: _v
        if persons is not None:
            self.node._on_detections(make_array(persons))
        self.node._control_loop()
        snap = {
            "t": self.clock.t_ns * 1e-9,
            "in": persons if persons is not None else [],
            "state": self.node.state.name,
            "tgt": self.node.target_id,
            "locked": self.node._locked,
            "v": self.node.current_linear,
            "w": self.node.current_angular,
            "dist_source": self.node._dist_source,
            "est_dist": self.node._est_dist,
            "ctrl_dist": self.node._ctrl_dist,
        }
        self.trace.append(snap)
        return snap

    def run(self, frames):
        """frames: lista de (dt, persons, front)."""
        for f in frames:
            dt = f[0]
            persons = f[1] if len(f) > 1 else None
            front = f[2] if len(f) > 2 else float("inf")
            self.tick(dt, persons, front)
        return self.trace

    def destroy(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass


# ── Helpers de escenario ─────────────────────────────────────────
def steady(n, dt, persons, front=float("inf")):
    """n frames idénticos."""
    return [(dt, [dict(p) for p in persons], front) for _ in range(n)]


def acquire_and_lock(h, cx=320.0, depth=2.8, tid=1, dt=0.08, secs=3.4):
    """Lleva el banco hasta tener el objetivo `tid` FIJADO (reutilizado por varios escenarios)."""
    n = int(secs / dt)
    h.run(steady(n, dt, [{"id": tid, "cx": cx, "depth": depth}]))
    return h.node.target_id


# ── Visor textual ────────────────────────────────────────────────
def _fmt_in(persons):
    if not persons:
        return "(nada)"
    return " ".join(f"id{int(p['id'])}@cx{int(p['cx'])},d{p['depth']:.1f}" for p in persons)


def print_trace(trace):
    """Imprime los frames RELEVANTES: primero, último, y cualquier cambio de estado/target."""
    last = None
    for i, s in enumerate(trace):
        change = (last is None or s["state"] != last["state"] or s["tgt"] != last["tgt"])
        is_edge = (i == 0 or i == len(trace) - 1)
        if change or is_edge:
            lock = " [LOCK]" if s["locked"] else ""
            print(f"    t={s['t']:5.2f}  in:[{_fmt_in(s['in']):<34}] "
                  f"state={s['state']:<9} tgt={str(s['tgt']):<4} "
                  f"v={s['v']:+.2f} w={s['w']:+.2f}{lock}")
        last = s


# ── Escenarios (cada uno: (nombre, función(h)->(ok, detalle))) ──
def sc1_acquire(h):
    tid = acquire_and_lock(h, cx=320, depth=2.8, tid=1)
    last = h.trace[-1]
    ok = (tid == 1 and last["state"] == "FOLLOWING" and last["locked"])
    return ok, f"target_id={tid} state={last['state']} locked={last['locked']} (esperado 1/FOLLOWING/True)"


def sc2_no_early_lock(h):
    h.run(steady(int(2.0 / 0.08), 0.08, [{"id": 1, "cx": 320, "depth": 2.8}]))
    ok = (h.node.target_id is None)
    return ok, f"target_id={h.node.target_id} tras 2.0s (esperado None: dwell 3s)"


def sc3_ignore_distinct_crosser(h):
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # Cruzante distinto (lejos en cx y depth) mientras el objetivo sigue visible.
    switched = []
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0},
                          {"id": 2, "cx": 120, "depth": 3.2}])
        if s["tgt"] != 1:
            switched.append(s["tgt"])
    ok = (h.node.target_id == 1 and not switched)
    return ok, f"target_id={h.node.target_id}; cambios espurios={switched} (esperado 1, ninguno)"


def sc4_brief_drop_overlap(h):
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # ~0.4s: SOLO el cruzante solapado (id1 ausente, churn breve).
    grabbed = []
    for _ in range(int(0.4 / 0.08)):
        s = h.tick(0.08, [{"id": 2, "cx": 330, "depth": 3.3}])
        if s["tgt"] not in (1, None) or s["tgt"] == 2:
            grabbed.append(s["tgt"])
    # vuelve id1
    for _ in range(int(0.5 / 0.08)):
        h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.1}])
    ok = (h.node.target_id == 1 and 2 not in grabbed)
    return ok, f"target_id final={h.node.target_id}; agarró cruzante={grabbed} (esperado 1, no 2)"


def sc5_occlusion_reappear_newid(h):
    # Cruzante REALISTA: ~5 px/frame (0.36 m/s a 3 m, 12.5 fps) -> permanece SEGUNDOS en el
    # gate de cx; es el caso que el código actual agarraba (y luego perseguía -> ACQUIRING).
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    grabbed_crosser = False
    cx = 270.0
    for _ in range(int(3.0 / 0.08)):    # 3 s de oclusión con el cruzante barriendo despacio
        s = h.tick(0.08, [{"id": 2, "cx": cx, "depth": 3.3}])
        cx += 5.0
        if s["tgt"] == 2:
            grabbed_crosser = True
    # id1 reaparece con id NUEVO (botsort le cambió el id), QUIETO en la posición prevista.
    reacquired_ok = False
    for _ in range(int(1.5 / 0.08)):
        s = h.tick(0.08, [{"id": 9, "cx": 320, "depth": 3.0}])
        if s["tgt"] == 9 and s["state"] == "FOLLOWING":
            reacquired_ok = True
    ok = (not grabbed_crosser and reacquired_ok and h.node.target_id == 9)
    return ok, (f"agarró_cruzante={grabbed_crosser} re-vinculó_a_9={reacquired_ok} "
                f"target_id final={h.node.target_id} (esperado: no cruzante, sí re-vínculo a 9)")


def sc5b_slow_crosser_sameid_returns(h):
    # Caso DOMINANTE real: botsort recupera el MISMO id tras la oclusión. Durante la oclusión
    # un cruzante lento se sienta en el gate -> NUNCA debe robarlo; al volver id1, se reanuda.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    grabbed_crosser = False
    cx = 270.0
    for _ in range(int(3.0 / 0.08)):
        s = h.tick(0.08, [{"id": 2, "cx": cx, "depth": 3.3}])
        cx += 5.0
        if s["tgt"] == 2:
            grabbed_crosser = True
    # Vuelve id1 (mismo id) quieto y centrado.
    resumed = False
    for _ in range(int(1.0 / 0.08)):
        s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0}])
        if s["tgt"] == 1 and s["state"] == "FOLLOWING":
            resumed = True
    ok = (not grabbed_crosser and resumed and h.node.target_id == 1)
    return ok, (f"agarró_cruzante={grabbed_crosser} reanudó_id1={resumed} "
                f"target_id final={h.node.target_id} (esperado: no cruzante, sigue 1)")


def sc6_real_loss_reacquire(h):
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # Nadie en escena durante > FOLLOW_LOST_TIMEOUT.
    for _ in range(int(9.0 / 0.1)):
        h.tick(0.1, persons=[])     # detección vacía: nadie
    ok = (h.node.target_id is None and h.node.state.name in ("ACQUIRING", "SEARCHING"))
    return ok, f"target_id={h.node.target_id} state={h.node.state.name} (esperado None + re-adquirir)"


def sc7_obstacle_stop(h):
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 3.0}], front=0.3)   # obstáculo pegado
    ok = (s["state"] == "STOPPED" and abs(s["v"]) < 1e-6)
    return ok, f"state={s['state']} v={s['v']:.3f} (esperado STOPPED, v=0)"


def sc8_crosser_other_depth(h):
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # id1 ausente y un cruzante en el MISMO cx pero a otra profundidad (1.5 m, mucho más cerca).
    grabbed = []
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 2, "cx": 320, "depth": 1.5}])
        if s["tgt"] == 2:
            grabbed.append(2)
    ok = (2 not in grabbed)
    return ok, f"agarró cruzante a 1.5m={bool(grabbed)} (esperado: no, gate de profundidad)"


# ── Distancia al objetivo por LÁSER (instrucción del tutor + método del PDF) ──
def scL1_laser_corroborates(h):
    # Caso robot real / objeto colisionable: el láser en el ángulo de la caja CORROBORA la
    # profundidad de cámara -> se usa el LÁSER como distancia (método del PDF).
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0}], laser_at_target=3.05)
    ok = (s["dist_source"] == "laser" and abs(s["est_dist"] - 3.05) < 0.1
          and s["state"] == "FOLLOWING")
    return ok, (f"dist_source={s['dist_source']} est_dist={s['est_dist']:.2f} "
                f"(esperado láser, ~3.05)")


def scL2_laser_inf_fallback_depth(h):
    # En sim el actor no tiene colisión -> el láser no devuelve eco (inf) -> cae a profundidad.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0}], laser_at_target=float("inf"))
    ok = (s["dist_source"] == "depth" and abs(s["est_dist"] - 3.0) < 0.1)
    return ok, (f"dist_source={s['dist_source']} est_dist={s['est_dist']:.2f} "
                f"(esperado prof, ~3.0)")


def scL3_laser_out_of_bounds_fallback_depth(h):
    # Si el láser da un valor IMPLAUSIBLE (p. ej. 15 m: el haz pasó de largo sin tocar a la
    # persona), se descarta por cota de cordura y se usa la profundidad de respaldo.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0}], laser_at_target=15.0)
    ok = (s["dist_source"] == "depth" and abs(s["est_dist"] - 3.0) < 0.1)
    return ok, (f"dist_source={s['dist_source']} est_dist={s['est_dist']:.2f} "
                f"(esperado prof ~3.0: láser 15 m fuera de cota)")


def scL4_laser_refines_noisy_depth(h):
    # La profundidad OAK-D es algo RUIDOSA (3.4 m) pero el láser (3.0 m) la CORROBORA dentro de la
    # tolerancia (|3.0-3.4| <= LASER_AGREE_M): el láser es la fuente PRINCIPAL (método del PDF) y
    # AFINA la distancia. (Si discrepara MUCHO, el haz toca otra cosa -> se cae a profundidad: L7.)
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.4}], laser_at_target=3.0)
    ok = (s["dist_source"] == "laser" and abs(s["est_dist"] - 3.0) < 0.1)
    return ok, (f"dist_source={s['dist_source']} est_dist={s['est_dist']:.2f} "
                f"(esperado láser 3.0 afinando prof ruidosa 3.4)")


# ── Guarda de RETROCESO (P2: "sin chocar" con obstáculos traseros) ──
def scR1_no_blind_reverse_when_rear_blocked(h):
    # Objetivo fijado MÁS CERCA que la consigna (1.5 m < 3.0 m): el control de distancia querría
    # retroceder. Con la trasera BLOQUEADA, la guarda debe impedir el retroceso ciego (v >= 0).
    acquire_and_lock(h, cx=320, depth=1.5, tid=1)
    s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 1.5}], rear=0.3)
    ok = (s["v"] >= -1e-6)
    return ok, f"v={s['v']:.3f} con trasera=0.3m (esperado >=0: no retrocede a ciegas)"


def scR2_reverse_capped_when_rear_clear(h):
    # Mismo caso pero trasera DESPEJADA: puede retroceder, pero CAPADO a -MAX_REVERSE_SPEED.
    acquire_and_lock(h, cx=320, depth=1.5, tid=1)
    s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 1.5}], rear=2.0)
    ok = (-pf.MAX_REVERSE_SPEED - 1e-6 <= s["v"] < 0.0)
    return ok, f"v={s['v']:.3f} con trasera=2.0m (esperado en [-{pf.MAX_REVERSE_SPEED:.2f}, 0))"


def sc9_avoidance_keeps_target_locked(h):
    """P1: evitación activa con objetivo CENTRADO no debe perder el fijado.
    Introduce un obstáculo en la franja de evitación y verifica que el lock se mantiene
    (el escalado de la evitación con error_x no debe sacar al objetivo del FOV)."""
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    lost_target = False
    for _ in range(int(3.0 / 0.1)):
        s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 3.0}], front=0.7)
        if s["tgt"] != 1 or not s["locked"]:
            lost_target = True
            break
    ok = (not lost_target and h.node.target_id == 1 and h.node._locked)
    return ok, (f"perdió_objetivo={lost_target} target_id={h.node.target_id} "
                f"(esperado no perder, 1)")


def sc10_avoidance_offcenter_target_stays_locked(h):
    """P1: objetivo DESCENTRADO + evitación → el lock se mantiene.
    La evitación se escala proporcional al descentrado (factor 1-|error_x|),
    con cota inferior de 0.15 para evitar rotaciones bruscas que saquen al
    objetivo del FOV."""
    acquire_and_lock(h, cx=260, depth=3.0, tid=1, secs=3.4)
    lost_target = False
    for _ in range(int(3.0 / 0.1)):
        s = h.tick(0.1, [{"id": 1, "cx": 260, "depth": 3.0}], front=0.7)
        if s["tgt"] != 1 or not s["locked"]:
            lost_target = True
            break
    ok = (not lost_target and h.node.target_id == 1 and h.node._locked)
    return ok, (f"perdió_objetivo={lost_target} target_id={h.node.target_id} "
                f"(esperado no perder, 1)")


def sc11_rebind_scaled_depth_after_occlusion(h):
    """P1: gate de profundidad ESCALADO con tiempo perdido permite re-vincular
    un objetivo que reaparece a distinta profundidad tras una oclusión larga.
    Sin escalado (REBIND_DEPTH_M=0.5 fijo), diff=0.8 rechazaría el re-vínculo;
    con escalado (0.5*(1+3/3)=1.0), se acepta."""
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # Oclusión de ~4s: solo un cruzante lejano (no en el gate) para que no
    # compita por el re-vínculo, pero el objetivo no aparece.
    for _ in range(int(4.0 / 0.08)):
        h.tick(0.08, [{"id": 2, "cx": 600, "depth": 5.0}])
    # Objetivo reaparece con id nuevo a 3.8m (diff=0.8 vs 3.0 original)
    # A t≈4s perdido: depth_gate = 0.5*(1+4/3) ≈ 1.17 > 0.8 → aceptado
    rebound = False
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 9, "cx": 320, "depth": 3.8}])
        if s["tgt"] == 9 and s["state"] == "FOLLOWING":
            rebound = True
    ok = rebound and h.node.target_id == 9
    return ok, (f"re-vinculó={rebound} target_id={h.node.target_id} "
                f"(esperado rebind a 9 con depth 3.8m)")


def sc12_rebind_rejects_outside_scaled_gate(h):
    """P1: incluso con gate escalado, un cruzante a profundidad MUY distinta
    NO debe robar el objetivo (la tolerancia escala, no desaparece)."""
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # Oclusión de 3s
    for _ in range(int(3.0 / 0.08)):
        h.tick(0.08, [{"id": 2, "cx": 600, "depth": 5.0}])
    # Cruzante en el gate espacial pero a 1.5m (diff=1.5) -> incluso escalado:
    # depth_gate = 0.5*(1+3/3)=1.0 -> 1.5 > 1.0 -> rechazado
    grabbed_crosser = False
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 3, "cx": 320, "depth": 1.5}])
        if s["tgt"] == 3:
            grabbed_crosser = True
    ok = (not grabbed_crosser)
    return ok, f"agarró_cruzante={grabbed_crosser} (esperado: no, depth 1.5m fuera del gate escalado)"


def sc13_close_crossing_keeps_target(h):
    """Cruce cercano: 2 actores se entrecruzan muy cerca → el robot NO debe perder el
    objetivo. El target id desaparece (oclusión por overlap), aparecen ids nuevos,
    pero el coast debe mantener la orientación hacia la zona de la agrupación y
    re-vincular al que reaparezca más cerca de la posición predicha."""
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # Simula un cruce cercano: el objetivo (id=1) y un cruzante (id=2) se solapan.
    # Durante 2s solo se ve al cruzante en el mismo gate (close crossing confunde al
    # tracker). El coast debe mantener cx cerca de la zona.
    for _ in range(int(2.0 / 0.08)):
        h.tick(0.08, [{"id": 2, "cx": 330, "depth": 3.2}])
    # Tras el cruce, el objetivo reaparece (con id nuevo por reassign de BoT-SORT)
    rebound = False
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 9, "cx": 320, "depth": 3.0}])
        if s["tgt"] == 9 and s["state"] == "FOLLOWING":
            rebound = True
    ok = rebound and h.node.target_id == 9
    return ok, (f"re-vinculó={rebound} target_id={h.node.target_id} "
                f"(esperado rebind a 9, close crossing)")


def sc14_close_crossing_multi_persistent(h):
    """Cruce cercano MULTIPLE: varios actores se entrecruzan. El robot debe:
    1. No dejarse robar por un cruzante que se queda en el gate
    2. Recuperar al objetivo cuando reaparezca"""
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    # 2s con el objetivo y un cruzante cerca
    for _ in range(int(2.0 / 0.08)):
        h.tick(0.08, [{"id": 1, "cx": 310, "depth": 3.1},
                       {"id": 2, "cx": 350, "depth": 3.5}])
    # 1s solo cruzantes (target desaparece)
    for _ in range(int(1.0 / 0.08)):
        h.tick(0.08, [{"id": 2, "cx": 350, "depth": 3.5}])
    # Objetivo reaparece con id nuevo
    rebound = False
    for _ in range(int(2.0 / 0.08)):
        s = h.tick(0.08, [{"id": 9, "cx": 310, "depth": 3.1}])
        if s["tgt"] == 9 and s["state"] == "FOLLOWING":
            rebound = True
    ok = rebound and h.node.target_id == 9
    return ok, (f"re-vinculó={rebound} target_id={h.node.target_id} "
                f"(esperado rebind a 9, multi close crossing)")


def scL5_reject_crosser_in_front(h):
    # Un cruzante/pilar se interpone DELANTE del objetivo: las medidas caen (láser=1.0, prof=1.2)
    # respecto a la distancia previa (~3 m). Por consistencia temporal se RECHAZAN y se MANTIENE
    # ~3 m (el robot NO recula por el cruce; la evitación gestiona el obstáculo aparte).
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    for _ in range(3):
        h.tick(0.08, [{"id": 1, "cx": 320, "depth": 3.0}], laser_at_target=3.0)
    s = h.tick(0.08, [{"id": 1, "cx": 320, "depth": 1.2}], laser_at_target=1.0)
    ok = (s["est_dist"] > 2.0 and s["dist_source"] == "hold")
    return ok, (f"dist_source={s['dist_source']} est_dist={s['est_dist']:.2f} "
                f"(esperado hold ~3.0, no la medida del cruzante)")


def scL6_unstick_stale_hold(h):
    # El láser "camina" hacia arriba (rebota en una pared lejana del circuito) y arrastra la
    # distancia hasta ~9 m en pasos <1 m (cada uno se acepta), mientras el objetivo REAL está a
    # ~2 m (profundidad). Con la previa anclada en ~9 m, toda lectura real (~2 m) dista >1 m y se
    # rechaza: sin re-sync el robot se queda CLAVADO en ~9 m y avanza a ciegas (log 17:08). Tras
    # DIST_HOLD_RESYNC_S debe RE-SINCRONIZAR a ~2 m.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    for dl in (3.8, 4.6, 5.4, 6.2, 7.0, 7.8, 8.6, 9.0):
        h.tick(0.1, [{"id": 1, "cx": 320, "depth": 2.0}], laser_at_target=dl)
    last = None
    for _ in range(int(1.4 / 0.1)):  # >DIST_HOLD_RESYNC_S manteniendo al objetivo a ~2 m
        last = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 2.0}], laser_at_target=float("inf"))
    ok = (last["est_dist"] < 3.0)
    return ok, (f"est_dist={last['est_dist']:.2f} dist_source={last['dist_source']} "
                f"(esperado re-sync a ~2.0, no clavado en ~9)")


def scL7_wall_in_bearing_uses_depth_no_reverse(h):
    # FALLO REAL (run 6f6s76od): al final del circuito el láser en el ángulo de la caja lee una
    # PARED cercana (~0.8 m) mientras el objetivo REAL se ALEJA (profundidad 3->6 m). Confiando en
    # el láser, dist_error<0 y el robot RETROCEDERÍA (v<0). Con la corroboración láser-prof, el
    # láser (que discrepa mucho de la profundidad) se DESCARTA y se usa la profundidad: no recula.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    last = None
    for d in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        last = h.tick(0.1, [{"id": 1, "cx": 320, "depth": d}], laser_at_target=0.8)
    ok = (last["dist_source"] != "laser" and last["est_dist"] > 2.5 and last["v"] >= -1e-6)
    return ok, (f"dist_source={last['dist_source']} est_dist={last['est_dist']:.2f} "
                f"v={last['v']:.3f} (esperado: NO láser, ~prof lejana, v>=0 sin recular)")


def scV1_distance_ema_smooths_control(h):
    # La distancia entra con JITTER (3.0<->3.6, saltos <DIST_MAX_STEP_M, así que se ACEPTAN): sin
    # suavizado, la distancia de control oscilaría 0.6 m y la velocidad daría acelerones/retrocesos
    # que descentran y pierden al objetivo (lo que el usuario observó con el láser activo). El EMA
    # debe reducir bastante el rango de la distancia de control.
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    cds = []
    for d in (3.6, 3.0, 3.6, 3.0, 3.6, 3.0, 3.6, 3.0):
        s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": d}], laser_at_target=float("inf"))
        cds.append(s["ctrl_dist"])
    rng = max(cds) - min(cds)
    ok = (rng < 0.3)   # entrada oscila 0.6 m; el EMA lo deja a la mitad o menos
    return ok, f"rango ctrl_dist={rng:.3f} m (entrada 0.6; esperado <0.3 por el EMA)"


def scV2_detector_silent_stops_no_blind_drive(h):
    # SEGURIDAD: el detector deja de publicar (se cayó) mientras el objetivo está fijado y alejado
    # (~4.5 m, querría AVANZAR). Sin arreglo, el follower seguía a 0.6 m/s con la distancia
    # congelada (caso real: 38 s a ciegas). Tras LOST_BLIND_STOP_S sin detecciones debe SOLTAR el
    # lock y PARAR (des_linear<=0).
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    for d in (3.5, 4.0, 4.5):                       # el objetivo se aleja (locked, querría avanzar)
        h.tick(0.1, [{"id": 1, "cx": 320, "depth": d}], laser_at_target=float("inf"))
    last = None
    for _ in range(int(2.5 / 0.1)):                 # detector MUDO >LOST_BLIND_STOP_S
        last = h.tick(0.1, None, front=float("inf"))
    ok = (not last["locked"] and last["v"] <= 1e-6)
    return ok, (f"locked={last['locked']} v={last['v']:.3f} "
                f"(esperado: suelta lock y v<=0, no avanza a ciegas con dist congelada)")


def scR3_stuck_unstuck_rotates_no_reverse(h):
    # Atascado por obstáculo frontal >3 s: el desatasco SOLO gira en el sitio (v=0, w!=0), NO
    # retrocede — ni siquiera con la trasera DESPEJADA (el retroceso ciego chocaba).
    acquire_and_lock(h, cx=320, depth=3.0, tid=1)
    last = None
    for _ in range(int(4.0 / 0.1)):
        last = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 3.0}], front=0.3, rear=2.0)
    ok = (last["state"] == "STOPPED" and last["v"] >= -1e-6 and abs(last["w"]) > 0.1)
    return ok, (f"state={last['state']} v={last['v']:.3f} w={last['w']:.2f} "
                f"(esperado STOPPED, v>=0 y w!=0: gira sin recular)")


def scA_progressive_slow_near_obstacle(h):
    # Fijar cerca (2.8 m) y ALEJAR el objetivo suavemente a 4.0 m (sin disparar el filtro
    # anti-salto de profundidad) para que quiera AVANZAR; luego, ante obstáculo frontal, la v
    # debe FRENAR progresivamente con la holgura (no embestir el pilar a 0.6 m/s, log 19:45).
    acquire_and_lock(h, cx=320, depth=2.8, tid=1)
    for d in (3.0, 3.3, 3.6, 3.9, 4.0, 4.0):
        h.tick(0.1, [{"id": 1, "cx": 320, "depth": d}], front=2.0)
    s_far = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 4.0}], front=2.0)   # frente despejado
    s_near = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 4.0}], front=1.0)  # obstáculo a 1.0 m
    s_close = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 4.0}], front=0.6)  # casi pegado
    ok = (s_far["v"] > 0.4 and s_near["v"] < 0.4 and 0.0 <= s_close["v"] < 0.15)
    return ok, (f"v(2.0m)={s_far['v']:.2f} v(1.0m)={s_near['v']:.2f} v(0.6m)={s_close['v']:.2f} "
                f"(esperado: rápido despejado, frenado progresivo cerca)")


def scR4_no_reverse_within_deadband(h):
    # Objetivo solo un POCO más cerca que la consigna (2.7 m vs 3.0, dentro de la banda muerta)
    # y trasera DESPEJADA: NO debe retroceder (la banda muerta lo evita; v>=0).
    acquire_and_lock(h, cx=320, depth=2.7, tid=1)
    s = h.tick(0.1, [{"id": 1, "cx": 320, "depth": 2.7}], rear=2.0)
    ok = (s["v"] >= -1e-6)
    return ok, f"v={s['v']:.3f} a 2.7m (esperado >=0: banda muerta, no retrocede por poco)"


# ── Tests para helpers PUROS de la capa LiDAR ──────────────────
def scH1_yaw_deg_from_quat(h):
    from geometry_msgs.msg import Quaternion
    cases = []
    for deg in (0, 90, 180, 270):
        half = math.radians(deg) * 0.5
        q = Quaternion(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half))
        result = pf.PersonFollowerNode._yaw_deg_from_quat(q)
        cases.append((deg, result))
    ok = all(abs(r - d) < 1.0 for d, r in cases)
    details = " ".join(f"{d}°->{r:.1f}°" for d, r in cases)
    return ok, details


def scH2_median_range_to_bearing(h):
    pts = [(0.0, 0.0, d, float(ang)) for ang, d in
           [(0, 2.0), (45, 3.0), (90, 4.0), (180, 5.0), (270, 6.0)]]
    r0 = pf.PersonFollowerNode._median_range_to_bearing(pts, 0.0)
    r45 = pf.PersonFollowerNode._median_range_to_bearing(pts, 45.0)
    r_wide = pf.PersonFollowerNode._median_range_to_bearing(pts, 91.0)
    r_none = pf.PersonFollowerNode._median_range_to_bearing(pts, 999.0)
    ok = (abs(r0 - 2.0) < 0.01 and abs(r45 - 3.0) < 0.01
          and abs(r_wide - 4.0) < 0.01 and math.isinf(r_none))
    return ok, f"0°->{r0} 45°->{r45} 91°(win=4, esperado 90->4.0)={r_wide} 999°->{r_none}"


def scH3_frontal_min_filter(h):
    # El corredor frontal (helper real _frontal_min_in_corridor) solo cuenta puntos POR DELANTE
    # (x>AHEAD_X_M) y dentro del ancho del robot (|y|<=FRONT_HALFWIDTH_M): ignora los de detrás,
    # los demasiado cercanos al eje pero a un lado, y sobre todo los LATERALES (cruzante que pasa
    # al costado) que un x>0.40 a secas contaría como frontales por su distancia euclídea.
    # El punto LATERAL está más CERCA (d=0.602) que el del corredor (d=0.608): un criterio x>0.40
    # a secas devolvería 0.602 (frenaría por el cruzante de al lado); el corredor lo descarta y
    # devuelve 0.608. Así el test DISTINGUE corredor de "x>0.40 a secas".
    d_lat = math.hypot(0.45, 0.40)    # 0.602; |y|=0.40 > FRONT_HALFWIDTH_M -> fuera del corredor
    d_cor = math.hypot(0.60, 0.10)    # 0.608; |y|=0.10 -> dentro del corredor (el válido más cercano)
    pts = [
        (0.38, 0.00, 0.38, 0.0),      # x<AHEAD_X_M -> ignorado
        (-0.50, 0.00, 0.50, 180.0),   # detrás -> ignorado
        (0.45, 0.40, d_lat, 42.0),    # lateral más cercano -> DEBE ignorarse
        (0.60, 0.10, d_cor, 9.0),     # corredor -> el mínimo válido
    ]
    fmin = pf.PersonFollowerNode._frontal_min_in_corridor(pts)
    ok = abs(fmin - d_cor) < 0.01
    return ok, (f"frontal_min={fmin:.3f} (esperado {d_cor:.3f}: corredor; ignora x<{pf.AHEAD_X_M}, "
                f"detrás, y el lateral más cercano |y|>{pf.FRONT_HALFWIDTH_M})")


SCENARIOS = [
    ("1. Adquirir y fijar", sc1_acquire),
    ("2. No fijar antes de 3s", sc2_no_early_lock),
    ("3. Ignorar cruzante distinto", sc3_ignore_distinct_crosser),
    ("4. Drop breve + cruzante solapado (no cambiar de objetivo)", sc4_brief_drop_overlap),
    ("5. Oclusion (cruzante LENTO) + reaparicion id nuevo (no volver a ACQUIRING)", sc5_occlusion_reappear_newid),
    ("5b. Cruzante LENTO en el gate + vuelve mismo id (no robar objetivo)", sc5b_slow_crosser_sameid_returns),
    ("6. Perdida real -> re-adquirir", sc6_real_loss_reacquire),
    ("7. Parada por obstaculo", sc7_obstacle_stop),
    ("8. Cruzante a otra profundidad, mismo cx", sc8_crosser_other_depth),
    ("L1. Distancia por LASER cuando corrobora la profundidad (PDF)", scL1_laser_corroborates),
    ("L2. Laser no ve al actor (inf) -> cae a profundidad", scL2_laser_inf_fallback_depth),
    ("L3. Laser implausible (fuera de cota) -> cae a profundidad", scL3_laser_out_of_bounds_fallback_depth),
    ("L4. Laser PRINCIPAL afina la profundidad ruidosa cuando corrobora", scL4_laser_refines_noisy_depth),
    ("L5. Cruzante/pilar delante -> rechaza el salto y MANTIENE (no recula)", scL5_reject_crosser_in_front),
    ("L6. Distancia clavada (hold obsoleto) -> RE-SINCRONIZA tras 1s (no avanza a ciegas)", scL6_unstick_stale_hold),
    ("L7. Pared cercana en el bearing del objetivo -> usa profundidad, NO retrocede", scL7_wall_in_bearing_uses_depth_no_reverse),
    ("V1. Jitter de distancia -> el EMA suaviza la distancia de control (no pierde el objetivo)", scV1_distance_ema_smooths_control),
    ("V2. Detector mudo (caído) -> suelta lock y PARA, no avanza a ciegas (seguridad)", scV2_detector_silent_stops_no_blind_drive),
    ("9. Evitacion activa con objetivo centrado -> NO pierde lock (P1)", sc9_avoidance_keeps_target_locked),
    ("10. Evitacion con objetivo descentrado -> NO pierde lock (P1)", sc10_avoidance_offcenter_target_stays_locked),
    ("11. Rebind con depth gate escalado tras occlusion larga (P1)", sc11_rebind_scaled_depth_after_occlusion),
    ("12. Rebind rechaza cruzante fuera del gate escalado (P1)", sc12_rebind_rejects_outside_scaled_gate),
    ("13. Close crossing: 2 actores se cruzan cerca -> rebind al correcto", sc13_close_crossing_keeps_target),
    ("14. Close crossing multiple -> no robo, rebind al correcto", sc14_close_crossing_multi_persistent),
    ("R1. Objetivo muy cerca + trasera bloqueada -> NO retrocede (P2)", scR1_no_blind_reverse_when_rear_blocked),
    ("R2. Objetivo muy cerca + trasera libre -> retroceso CAPADO (P2)", scR2_reverse_capped_when_rear_clear),
    ("R3. Desatasco STOPPED -> gira en el sitio, NO recula (P2)", scR3_stuck_unstuck_rotates_no_reverse),
    ("R4. Objetivo un poco cerca (banda muerta) -> NO retrocede (P2)", scR4_no_reverse_within_deadband),
    ("A. Frenado progresivo ante obstaculo frontal (no embestir el pilar)", scA_progressive_slow_near_obstacle),
    ("H1. Yaw desde quaternion (0/90/180/270°)", scH1_yaw_deg_from_quat),
    ("H2. Mediana de rango en bearing con nube sintética", scH2_median_range_to_bearing),
    ("H3. Corredor frontal: ignora x<AHEAD_X_M, traseros y laterales |y|>FRONT_HALFWIDTH_M", scH3_frontal_min_filter),
]


def main():
    parser = argparse.ArgumentParser(description="Banco offline de la logica del follower")
    parser.add_argument("--quiet", action="store_true", help="solo el resumen PASS/FAIL")
    args = parser.parse_args()

    rclpy.init()
    results = []
    try:
        for name, fn in SCENARIOS:
            h = Harness()
            try:
                ok, detail = fn(h)
            except Exception as e:
                ok, detail = False, f"EXCEPCION: {e!r}"
            results.append((name, ok, detail))
            tag = "PASS" if ok else "FAIL"
            print(f"\n=== [{tag}] {name} ===")
            print(f"    -> {detail}")
            if not args.quiet or not ok:
                print_trace(h.trace)
            h.destroy()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass

    n_pass = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 64)
    print(f"RESUMEN: {n_pass}/{len(results)} escenarios PASS")
    for name, ok, _ in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 64)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
