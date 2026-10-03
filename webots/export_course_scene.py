#!/usr/bin/env python3
"""Export a train/dev course-bank record (movers included) as a Webots world.

Reuses the native exporter webots/metal_scene.py (read-only) for template
handling, static obstacle nodes and the RaptorCrazyflie customData, then
appends a small supervisor Robot (DEF CourseMoverDriver, controller
"course_obstacle") whose customData carries the moving-obstacle schedule.
After writing, the world is re-parsed and compared against the bank record:
static geometry, mover tables, and sampled positions must match the Metal
kinematics center + velocity * t exactly.  Results go to
results/mimo-courses/same-motion-checks.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webots"))
sys.path.insert(0, str(ROOT))

import metal_scene as ms                     # noqa: E402  (native, read-only)
from navigation_courses import HOLD_S        # noqa: E402

OUT_WORLDS = ROOT / "webots/worlds"
CHECKS_PATH = ROOT / "results/course-export/same-motion-checks.json"
SAMPLE_STEP_S = 0.5


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_record(record: dict) -> list[dict]:
    """metal_scene.validate_record plus moving-obstacle acceptance."""
    if record.get("family") not in (14, 15, 16):
        raise ValueError("unsupported spatial family")
    if record.get("room_bounds") != ms.ROOM_BOUNDS:
        raise ValueError("record room bounds differ from the Webots template")
    if abs(finite(record.get("body_radius_m"), "body_radius_m") - ms.BODY_RADIUS_M) > 1e-6:
        raise ValueError("record body radius differs from the RaptorCrazyflie sphere")
    wind = record.get("wind") or [0.0, 0.0, 0.0]
    if any(abs(finite(w, "wind")) > 1e-8 for w in wind):
        raise ValueError("zero-wind records only")
    obstacles = record.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles or len(obstacles) > 16:
        raise ValueError("obstacles must be 1..16 shapes")
    for index, obstacle in enumerate(obstacles):
        label = f"obstacles[{index}]"
        if obstacle.get("kind") not in (0, 1, 2):
            raise ValueError(f"{label}.kind must be 0, 1 or 2")
        half = [finite(v, f"{label}.half_extent") for v in obstacle["half_extent"]]
        velocity = [finite(v, f"{label}.velocity") for v in obstacle["velocity"]]
        center = [finite(v, f"{label}.center") for v in obstacle["center"]]
        extents = half if obstacle["kind"] == 0 else (
            [half[0]] * 3 if obstacle["kind"] == 1 else [half[0], half[0], half[2]])
        if any(d <= 0.0 for d in extents):
            raise ValueError(f"{label} has a non-positive collision dimension")
        for axis, name in enumerate(("x", "y", "z")):
            extent = extents[axis]
            if (center[axis] - extent < ms.ROOM_BOUNDS[name][0] - 1e-4 or
                    center[axis] + extent > ms.ROOM_BOUNDS[name][1] + 1e-4):
                raise ValueError(f"{label} extends outside the fixed room")
        # a mover must remain fully inside the room for the whole hold window
        if any(velocity):
            for t in [i * 4.0 for i in range(int(HOLD_S / 4.0) + 1)]:
                for axis, name in enumerate(("x", "y", "z")):
                    pos = center[axis] + t * velocity[axis]
                    extent = extents[axis]
                    if pos - extent < ms.ROOM_BOUNDS[name][0] - 1e-4 or \
                            pos + extent > ms.ROOM_BOUNDS[name][1] + 1e-4:
                        raise ValueError(f"{label} leaves the room at t={t}s")
    return obstacles


def mover_custom_data(obstacles: list[dict]) -> str:
    entries = []
    for index, obstacle in enumerate(obstacles):
        if not any(obstacle["velocity"]):
            continue
        # repr() round-trips the double exactly, so the parsed schedule is
        # bit-identical to the bank record
        center = ",".join(repr(float(v)) for v in obstacle["center"])
        velocity = ",".join(repr(float(v)) for v in obstacle["velocity"])
        entries.append(f"{index:02d}:{center}:{velocity}")
    if not entries:
        raise ValueError("record has no moving obstacles; nothing for the driver")
    return "movers=" + "|".join(entries)


def parse_world(world_text: str) -> tuple[dict[int, dict], str]:
    """Re-read solids and the driver customData from the written world."""
    solids: dict[int, dict] = {}
    pattern = re.compile(
        r"DEF ChallengeObstacle(\d+) Solid \{.*?translation "
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?) (-?\d+\.?\d*(?:[eE][-+]?\d+)?) "
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?).*?geometry (\w+) \{ (.*?)\}"
        r".*?boundingObject \w+ \{ (.*?)\}",
        re.S)
    for match in pattern.finditer(world_text):
        idx = int(match.group(1))
        translation = [float(match.group(i)) for i in (2, 3, 4)]
        solids[idx] = {"translation": translation, "geometry": match.group(5),
                       "shape": match.group(6)}
    driver = re.search(r'DEF CourseMoverDriver Robot \{[^}]*?customData "([^"]+)"',
                       world_text)
    if driver is None:
        raise ValueError("exported world is missing the CourseMoverDriver node")
    return solids, driver.group(1)


def geometry_matches(solid: dict, obstacle: dict) -> bool:
    kind = obstacle["kind"]
    half = obstacle["half_extent"]
    size = solid["shape"].strip()
    if kind == 0:
        want = [2 * half[0], 2 * half[1], 2 * half[2]]
        got = [float(v) for v in
               re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", size)[:3]]
    elif kind == 1:
        got = [float(re.search(r"radius ([-0-9.eE+]+)", size).group(1))]
        want = [half[0]]
    else:
        got = [float(re.search(r"radius ([-0-9.eE+]+)", size).group(1)),
               float(re.search(r"height ([-0-9.eE+]+)", size).group(1))]
        want = [half[0], 2 * half[2]]
    return all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(got, want))


def run_checks(record: dict, world_text: str) -> dict:
    obstacles = record["obstacles"]
    solids, driver_raw = parse_world(world_text)
    errors: list[str] = []

    static_checked = 0
    for index, obstacle in enumerate(obstacles):
        solid = solids.get(index)
        if solid is None:
            errors.append(f"obstacle {index} missing from world")
            continue
        if any(abs(a - b) > 1e-6 for a, b in
               zip(solid["translation"], obstacle["center"])):
            errors.append(f"obstacle {index} translation != record center")
        if not geometry_matches(solid, obstacle):
            errors.append(f"obstacle {index} geometry != record half_extent")
        static_checked += 1

    if not driver_raw.startswith("movers="):
        errors.append("driver customData missing movers=")
        movers: list[tuple[int, list[float], list[float]]] = []
    else:
        movers = []
        for entry in driver_raw[len("movers="):].split("|"):
            if not entry:
                continue
            idx_s, c_s, v_s = entry.split(":")
            movers.append((int(idx_s), [float(x) for x in c_s.split(",")],
                           [float(x) for x in v_s.split(",")]))

    record_movers = {i: o for i, o in enumerate(obstacles) if any(o["velocity"])}
    if set(i for i, _, _ in movers) != set(record_movers):
        errors.append("driver mover set != record mover set")
    max_error = 0.0
    samples = int(HOLD_S / SAMPLE_STEP_S)
    for idx, center, velocity in movers:
        obstacle = obstacles[idx]
        for k in range(3):
            max_error = max(max_error, abs(center[k] - obstacle["center"][k]),
                            abs(velocity[k] - obstacle["velocity"][k]))
        # Metal kinematics vs driver schedule over the full hold window
        for s in range(samples + 1):
            t = s * SAMPLE_STEP_S
            for k in range(3):
                metal = obstacle["center"][k] + t * obstacle["velocity"][k]
                driver_pos = center[k] + t * velocity[k]
                max_error = max(max_error, abs(metal - driver_pos))

    return {
        "failure_id": record["failure_id"],
        "static_geometry_checked": static_checked,
        "movers_checked": len(movers),
        "sample_times_s": samples + 1,
        "sample_step_s": SAMPLE_STEP_S,
        "hold_s": HOLD_S,
        "max_abs_error_m": max_error,
        "errors": errors,
        "passed": not errors and max_error <= 1e-9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("failure_id", help="train/dev failure_id from the course bank")
    parser.add_argument("--bank", type=Path,
                        default=ROOT / "evidence/inputs/course-bank-v1.jsonl")
    parser.add_argument("--policy", default="../assets/navigation.bin",
                        help="policy path relative to the Webots project root")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=OUT_WORLDS)
    args = parser.parse_args()

    record, bank_bytes, source_line = ms.read_record(args.bank, args.failure_id)
    obstacles = validate_record(record)
    if not any(any(o["velocity"]) for o in obstacles):
        raise SystemExit(f"{args.failure_id} has no movers; "
                         "use the native metal_scene.py exporter instead")

    world_text = ms.scene_world(record, obstacles, args.policy, args.max_steps)
    driver = (
        "\nDEF CourseMoverDriver Robot { name \"course_mover_driver\" "
        "controller \"course_obstacle\" supervisor TRUE translation 0 0 0 "
        f'customData "{mover_custom_data(obstacles)}" }}'
    )
    world_text = world_text.rstrip("\n") + "\n" + driver

    report = run_checks(record, world_text)
    if not report["passed"]:
        print(json.dumps(report, indent=2))
        raise SystemExit("same-motion checks FAILED; world not written")

    name = f"metal_{record['failure_id']}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    world_path = args.out_dir / f"{name}.wbt"
    world_bytes = world_text.encode("utf-8")
    world_path.write_bytes(world_bytes)
    metadata = {
        "schema": "course-webots-scene-v1",
        "scene_name": name,
        "failure_id": record["failure_id"],
        "split": record["split"],
        "family": record["family"],
        "movers": [
            {"obstacle_index": i, "center": o["center"], "velocity": o["velocity"]}
            for i, o in enumerate(obstacles) if any(o["velocity"])
        ],
        "source_bank": str(args.bank.resolve()),
        "source_bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "source_record_sha256": hashlib.sha256(source_line).hexdigest(),
        "world_sha256": hashlib.sha256(world_bytes).hexdigest(),
        "controller": "webots/controllers/course_obstacle/course_obstacle.py",
        "same_motion_check": report,
    }
    (args.out_dir / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    checks = []
    if CHECKS_PATH.exists():
        checks = json.loads(CHECKS_PATH.read_text())
    checks = [c for c in checks if c["failure_id"] != report["failure_id"]]
    checks.append(report)
    CHECKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKS_PATH.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")

    print(f"exported {record['failure_id']} movers={report['movers_checked']} "
          f"static={report['static_geometry_checked']} "
          f"max_abs_error_m={report['max_abs_error_m']:.3g}")
    print(f"world={world_path}")
    print(f"checks={CHECKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
