# Safe person-following for a mobile robot (simulated TurtleBot 4)

A simulated TurtleBot 4 picks one person out of a warehouse where other people are walking,
locks onto them and follows them through a full circuit without colliding: detection,
tracking, target selection and control on **ROS 2 Jazzy** and **Gazebo Harmonic**.

The closed loop was developed and evaluated **entirely in simulation**. The only real data are
the four real video sequences used to compare the trackers.

Final-year project (TFG), Computer Engineering, University of Granada.

<!-- Demo video: drag multi_vista.mp4 into the GitHub web editor on this line. -->

---

## TL;DR

| | |
|---|---|
| Detector | YOLO11s, retrained on the simulated warehouse: **mAP@0.5 = 0.913**, F1 = 0.864 |
| Tracker | StrongSORT: **19 ID switches** and **1.45 px** localisation error on four real sequences |
| Behaviour | **117 randomised runs** in Gazebo; target held for the whole circuit in **104 (89 %)** |
| Control | **93.5 ± 5.3 %** of the time in `FOLLOWING`, centring error **0.050 ± 0.025** |
| Safety | in simulation, stops with **0.418 m** of clearance from 0.6 m/s in **0.10 s** |
| Budget | **64 FPS** end-to-end against a **10 Hz** control loop |

Every number is measured, not estimated: detection and tracking on an NVIDIA RTX 2070,
behaviour over 117 simulated runs logged to Weights & Biases with the simulation clock.

---

## The main decision: the chosen tracker is not the one with the best MOTA

Five multi-object trackers, evaluated on **ROBOT7**: four real video sequences of people being
followed, not simulation. Matching is by Euclidean distance between centres with a per-box
threshold of half its diagonal.

| Tracker | TP | FP | MISS | ID switches | MOTA | MOTP (px) |
|---|---|---|---|---|---|---|
| ByteTrack | **8861** | 15 | **574** | 28 | **0.935** | 4.55 |
| BoT-SORT | 8829 | 25 | 599 | 35 | 0.930 | 3.89 |
| **StrongSORT** (chosen) | 8804 | **2** | 640 | **19** | 0.930 | **1.45** |
| DeepOCSORT | 8690 | 28 | 723 | 50 | 0.915 | 3.77 |
| OCSORT | 8659 | 12 | 763 | 41 | 0.914 | 3.34 |

**ByteTrack wins on MOTA. StrongSORT is the one I chose.**

MOTA aggregates false positives, misses and ID switches into a single number, and it is
dominated by detection quality, which is identical here because all five run on the same
YOLO11s output. What decides this application is different: the robot must stay locked onto
**one specific person** while others walk between it and the target. Every ID switch is a
chance to start following a stranger.

StrongSORT has **19 ID switches against ByteTrack's 28**, **2 false positives against 15**, and
localises **3× more precisely** (1.45 px vs 4.55 px). It gives up 0.005 of MOTA for that. For a
robot that has to keep following the right person, that is not a close call.

The cost was measured, not assumed:

| Tracker | ReID | Association (ms) | End-to-end (FPS) |
|---|---|---|---|
| ByteTrack | no | 0.51 | 127.5 |
| OCSORT | no | 0.62 | 125.6 |
| BoT-SORT | yes | 7.58 | 67.0 |
| DeepOCSORT | yes | 8.13 | 64.7 |
| **StrongSORT** | yes | 8.23 | **64.2** |

Appearance-based re-identification costs about 8 ms per frame and halves the throughput. That
would matter if the budget were tight. It is not: the control loop runs at **10 Hz**, so 64 FPS
leaves a factor of six in hand. Spending throughput that would otherwise be idle to buy identity
stability is the right trade here, and the numbers are what make it arguable rather than a
preference.

---

## Detection: six models, same conditions

All six trained for 100 epochs on the same dataset of images captured in the Gazebo warehouse;
latency measured at 320 × 240 on an RTX 2070.

| Model | Params | mAP@0.5 | Precision | Recall | F1 | ms/frame | FPS |
|---|---|---|---|---|---|---|---|
| YOLO11n | 2.6 M | 0.817 | 0.797 | 0.706 | 0.749 | **4.3** | **232** |
| **YOLO11s** (chosen) | 9.4 M | **0.913** | **0.905** | **0.827** | **0.864** | 7.1 | 140 |
| YOLO11m | 20.1 M | 0.859 | 0.830 | 0.769 | 0.798 | 15.0 | 67 |
| YOLO12n | 2.6 M | 0.829 | 0.831 | 0.725 | 0.774 | 5.7 | 174 |
| YOLO12s | 9.4 M | 0.841 | 0.850 | 0.735 | 0.788 | 8.9 | 113 |
| YOLO26n | 2.4 M | 0.817 | 0.800 | 0.701 | 0.747 | 4.5 | 222 |

Two things worth noting. **Bigger is not better**: YOLO11m has more than twice the parameters
of YOLO11s and scores *worse* (0.859 vs 0.913) at twice the latency; on a dataset this size the
larger model overfits. And the newer generation did not win: YOLO12s and YOLO26n both landed
below YOLO11s. Training length mattered more than architecture: going from 50 to 100 epochs
took YOLO11s from **0.852 to 0.913**.

Recall sits at 0.827, below 90 %. On a single image that is the main weakness; on a video
stream a tracker can carry an identity through the frames where the detector misses, which is
why the behaviour is evaluated as well as the detector.

**Which weights run live.** `person_follow.launch.py` defaults to the stock COCO `yolo11s.pt`
for live following. The OAK-D camera is mounted low and at the 3 m following distance it often
cuts the target off at the head; the retrained `yolo11s_100ep.pt` is tuned to the warehouse
dataset and detects those partial views worse. Pass `model:=` to switch.

---

## Behaviour: 117 runs, not one demo

The full behaviour was evaluated in Gazebo over **117 runs** on a closed warehouse circuit, each
starting from a randomised initial pose, with other people crossing the robot's path. Runs were
logged to Weights & Biases under the simulation clock.

In **104 of 117 (89 %)** the robot acquired the target and held it for the entire circuit. In the
remaining 13, a very close crossing fully occluded the target and caused a loss or a transient
identity confusion. Metrics below are over the 104 sustained runs.

| Metric | Value |
|---|---|
| Angular centring error (normalised MAE) | 0.050 ± 0.025 |
| Distance error vs 3.0 m setpoint | 0.825 ± 0.416 m |
| Mean linear / angular velocity | 0.26 ± 0.04 m/s · 0.09 ± 0.04 rad/s |
| Target losses per run (`FOLLOWING`→`SEARCHING`) | 0.1 ± 0.3 |
| Re-acquisition time when lost | 4.99 ± 5.51 s |
| Time in `FOLLOWING` | 93.5 ± 5.3 % |

The 89 % is reported rather than tuned away. The failure mode is specific and understood (full
occlusion at close range) and it follows from the 10 Hz loop, which is discussed under
limitations.

### Safety layer

The LiDAR layer overrides the follower unconditionally. Measured in simulation:

| Metric | Value |
|---|---|
| Frontal clearance when stopped | 0.418 m |
| Approach speed | 0.600 m/s |
| Braking time (cruise → stop) | 0.10 s |
| Control cycles in safety stop | 13 / 500 (2.6 %) |

It stops without contact, and it engages rarely enough (2.6 % of cycles) that it does not
interfere with normal following.

---

## How it works

Two ROS 2 nodes:

- **`person_detector.py`** subscribes to the OAK-D RGB stream, runs YOLO11s with StrongSORT,
  and publishes every person on `/person_detections` with a distance attached.
- **`person_follower.py`** consumes `/person_detections` and `/scan`, picks and locks a target,
  and publishes `/cmd_vel`.

```
SEARCHING ──candidate in range──> ACQUIRING ──held ≥3 s──> FOLLOWING
    ^                                  │                        │
    └──────────target lost─────────────┴────────────────────────┘

  any state ──obstacle < threshold──> STOPPED ──path clear──> back
```

The target is fixed by **proximity plus dwell time**: the nearest detected person who stays in
view for 3 s becomes the target, and from then on the robot follows that *identity*, not that
*position*. Brief disappearances do not trigger re-selection: the robot holds its last command
and, only if the absence persists, re-binds to whoever remains still at the predicted position,
explicitly rejecting whoever is crossing through it.

Defaults: `safe_dist = 0.5 m`, `target_dist = 3.0 m`, `max_linear = 0.6 m/s`,
`max_angular = 1.2 rad/s`, control loop 10 Hz.

### Three details that were not obvious

- **Distance comes from the laser, read as a median.** Taking the *minimum* range across the
  beams covering the target picks up whatever thin object is between robot and person: a pillar
  edge, a passer-by's arm. The median across the target's angular span is stable.
- **Distance is checked for temporal consistency.** A single frame where the reading jumps is
  rejected rather than acted on, which is what stops a crossing person from yanking the
  controller.
- **Gazebo must run on the NVIDIA GPU or the LiDAR silently lies.** If `GpuLidar` falls back to
  software rendering it returns a *fixed* range: the robot sees no obstacles and no people, with
  no error message. `run_tb4_sim.sh` exports the offload variables for this reason. If `ros2
  topic echo /scan` gives a constant value, this is why.

---

## Running it

Requires Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic with `ros_gz`, the TurtleBot 4 simulation
packages and Python 3.12. An NVIDIA GPU is strongly recommended (see the LiDAR note above).
The repository root is the colcon workspace; several analysis scripts default to `~/tfg_ws`,
so clone it there.

```bash
git clone https://github.com/bluefireF5ran/tfg-person-following.git ~/tfg_ws
cd ~/tfg_ws
pip install -r requirements.txt      # ultralytics, boxmot, motmetrics, wandb, ...
colcon build --symlink-install
source install/setup.bash
```

Then, in three terminals:

```bash
# 1) warehouse simulation with animated people
bash src/tfg_workflow/scripts/run_tb4_sim.sh warehouse_unified

# 2) perception + control (StrongSORT by default; botsort:=true for BoT-SORT)
ros2 launch src/tfg_workflow/launch/person_follow.launch.py
#    with the retrained detector instead of the COCO weights:
#    ros2 launch src/tfg_workflow/launch/person_follow.launch.py model:=src/tfg_workflow/yolo11s_100ep.pt

# 3) optional: watch the detections, locked target highlighted in green
ros2 run rqt_image_view rqt_image_view /person_detections/image
```

Weights & Biases is optional: without `wandb login`, or without the `wandb` package, the
follower runs normally and skips telemetry.

---

## Repository layout

```
src/tfg_workflow/
  scripts/
    person_detector.py       YOLO + StrongSORT -> /person_detections
    person_follower.py       safe-following controller -> /cmd_vel
    test_follower_logic.py   offline test bench for the controller
    train_yolo.py            detector training
    evaluate_mot.py          CLEAR MOT evaluation (py-motmetrics)
    run_all_boxmot.py        the five-tracker comparison above
    capture_mot_dataset.py   sequence capture with geometric ground truth
    ...                      benchmarking, telemetry analysis, campaign scripts
  launch/                    person_follow, sim_with_people, spawn
  worlds/                    warehouse_unified.sdf
  models/                    animated actors used in the scenario
  python_lib/controller_manager/
                             copy of ROS 2 controller_manager with longer service
                             timeouts, made by patch_controller_manager.py
  yolo11s_100ep.pt           the retrained detector (mAP@0.5 = 0.913)
```

MOT datasets and W&B logs are not versioned: they are large and regenerable with the capture
and ground-truth scripts. Code comments are in Spanish.

---

## Limitations

- **Simulation only.** The detector was trained on simulated images and the closed loop was
  developed and evaluated in Gazebo. Only the tracker comparison uses real data (ROBOT7).
- **10 Hz caps how fast the target can move.** The TurtleBot 4 LiDAR refreshes at about 10 Hz,
  so the control loop matches it. Between two cycles a fast-moving person, or one crossing very
  close, can travel far enough to leave the field of view or defeat re-association. Walking pace
  has comfortable margin; running does not. Raising it needs a faster sensor or a motion
  prediction stage, not a tuning change.
- **ROBOT7 is not public.** It was supplied by the project advisor, so the tracker table cannot
  be reproduced from this repository alone. The simulated evaluation can.
- **Two actors share one mesh.** Two of the simulated people are visually identical, which is
  intrinsically unsolvable for appearance-based re-identification. It puts a floor under the ID
  switch counts rather than inflating them.
- **Single detector.** The tracker comparison holds YOLO11s fixed, so it isolates association,
  but the conclusions are conditional on that detector.

---

## License

MIT (see `LICENSE`). `src/tfg_workflow/python_lib/controller_manager/` is a modified copy of
the ROS 2 `controller_manager` package (PAL Robotics) and keeps its Apache-2.0 license headers.

## Context

University of Granada, E.T.S. de Ingenierías Informática y de Telecomunicación, 2025-2026.
Advisor: Eugenio Aguirre Molina.

**Francisco Merchán Cordero**: [pacomerchanc@gmail.com](mailto:pacomerchanc@gmail.com)
