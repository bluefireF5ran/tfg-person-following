#!/usr/bin/env python3
"""Analyze person_follower.py log output and extract evaluation metrics.

Usage:
    # Capture log while running follower
    bash scripts/run_pipeline.sh follow-launch 2>&1 | tee /tmp/follower.log

    # Analyze
    python3 scripts/analyze_follower_log.py --log /tmp/follower.log
"""
import argparse
import re
from collections import Counter
from pathlib import Path


FOLLOWER_RE = re.compile(
    r"\[(\w+)\]\s+v=([\d.-]+)\s+w=([\d.-]+)\s+"
    r"front=([\d.]+|inf)m\s+bbox_h=([\d.]+)px\s+"
    r"est_d=([\d.]+)m\s+lidar=([\d.]+|inf)m\s+"
    r"cx_err=([\d.-]+)\s+dets=(\d+)\s+track=\((\w)\)"
)
STATE_RE = re.compile(r"\[STATE\] (\w+) \u2192 (\w+)")
TS_RE = re.compile(r"\[INFO\] \[(\d+\.\d+)\]")


def parse_log(path: str) -> dict:
    lines = Path(path).read_text().splitlines()

    # Data accumulators
    state_segments = []
    distances = []
    speeds = []
    angulars = []
    safety_count = 0
    track_losses = 0
    transitions = 0
    first_ts = None
    last_ts = None

    for line in lines:
        ts_m = TS_RE.search(line)
        if not ts_m:
            continue
        ts = float(ts_m.group(1))
        if first_ts is None:
            first_ts = ts
        last_ts = ts

        # State transition
        m = STATE_RE.search(line)
        if m:
            transitions += 1
            state_segments.append((ts, m.group(2)))
            continue

        # Follower status line
        m = FOLLOWER_RE.search(line)
        if m:
            state = m.group(1)
            state_segments.append((ts, state))

            if state == "FOLLOWING":
                distances.append(float(m.group(6)))
            if state == "STOPPED":
                safety_count += 1

            speeds.append(float(m.group(2)))
            angulars.append(float(m.group(3)))
            continue

        # Tracker losses
        if "Lost target after" in line:
            track_losses += 1

    if not state_segments:
        return {"error": "No follower log lines found"}

    # Compute total duration
    total = last_ts - first_ts if first_ts and last_ts and last_ts > first_ts else 0.001

    # Time per state (based on segment start times)
    time_per_state = Counter()
    for i in range(len(state_segments)):
        st = state_segments[i][1]
        if i < len(state_segments) - 1:
            dur = state_segments[i + 1][0] - state_segments[i][0]
        else:
            dur = last_ts - state_segments[i][0]
        if dur > 0:
            time_per_state[st] += dur

    # Count segments per state
    state_counts = Counter(st for _, st in state_segments)

    result = {
        "total_duration_s": round(total, 1),
        "state_transitions": transitions,
        "track_losses": track_losses,
        "safety_activations": safety_count,
    }

    for state_name in ["SEARCHING", "FOLLOWING", "STOPPED"]:
        t = time_per_state.get(state_name, 0.0)
        result[f"time_{state_name.lower()}_s"] = round(t, 1)
        result[f"pct_{state_name.lower()}"] = round(t / total * 100, 1)
        result[f"count_{state_name.lower()}"] = state_counts.get(state_name, 0)

    if distances:
        n = len(distances)
        mean_d = sum(distances) / n
        var_d = sum((d - mean_d) ** 2 for d in distances) / n
        result["dist_mean"] = round(mean_d, 2)
        result["dist_std"] = round(var_d ** 0.5, 2)
        result["dist_min"] = round(min(distances), 2)
        result["dist_max"] = round(max(distances), 2)
        result["dist_error"] = round(abs(mean_d - 1.5), 2)

    if speeds:
        result["speed_mean"] = round(sum(speeds) / len(speeds), 2)

    if angulars:
        result["angular_mean"] = round(sum(angulars) / len(angulars), 2)

    return result


def print_report(stats: dict):
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    print("=" * 55)
    print("  ANALISIS DE LOG DEL SEGUIDOR")
    print("=" * 55)
    print(f"  Duracion total:        {stats['total_duration_s']:>8.1f} s")

    print(f"\n  -- Tiempo por estado --")
    for state_name in ["FOLLOWING", "SEARCHING", "STOPPED"]:
        t = stats[f"time_{state_name.lower()}_s"]
        pct = stats[f"pct_{state_name.lower()}"]
        cnt = stats[f"count_{state_name.lower()}"]
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {state_name:<12s} {t:>7.1f}s  {pct:>5.1f}%  {bar}  ({cnt}x)")

    print(f"  Transiciones de estado: {stats['state_transitions']}")
    print(f"  Perdidas de target:     {stats['track_losses']}")
    print(f"  Safety activations:     {stats['safety_activations']}")

    if "dist_mean" in stats:
        print(f"\n  -- Distancia estimada (FOLLOWING) --")
        print(f"  Media:       {stats['dist_mean']:>6.2f} m")
        print(f"  Desviacion:  {stats['dist_std']:>6.2f} m")
        print(f"  Min:         {stats['dist_min']:>6.2f} m")
        print(f"  Max:         {stats['dist_max']:>6.2f} m")
        print(f"  Error medio: {stats['dist_error']:>6.2f} m  (target=1.5m)")

    if "speed_mean" in stats:
        print(f"\n  -- Velocidades --")
        print(f"  Vel. lineal media:  {stats['speed_mean']:>6.2f} m/s")
        print(f"  Vel. angular media: {stats['angular_mean']:>6.2f} rad/s")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Analyze person_follower log")
    parser.add_argument("--log", required=True, help="Path to log file")
    args = parser.parse_args()

    stats = parse_log(args.log)
    print_report(stats)


if __name__ == "__main__":
    main()
