#!/usr/bin/env python3
"""Matched independent Webots flight test for the frozen local-waypoint actors.

The mission is a paired comparison of two frozen 184-input NAVPOL1 actors on
sixteen predeclared local DEV tasks, flown in Webots with the real RAPTOR
policy, the 0.18 m collision sphere, the 100 Hz RAPTOR / 20 Hz navigation
period and nominal motors:

  arrival  assets/navigation-arrival-experimental.bin   (frozen warmstart)
  fast     results/omp-local-waypoints/export/navigation-local-waypoint-fast.bin

Every record comes from the kept source bank bytes (`WPBANK1`), never from a
regenerated approximation. The selection rule is fixed in code and written to
`selection-manifest.json` before any Webots process is started:

    dev-a bank, fixed file order, first 16 records.

The dev-a bank is ordered round-robin over its six families, so that rule is
"first two tasks per family, continued in fixed file order until sixteen" and
is balanced across families without ever looking at a result.

Grading is the shared source function `navigation_task_step`
(navigation_tasks.hpp) called inside the mission-owned controller, so a Webots
receipt is produced by the same code the Metal source simulator uses:
0.35 m goal radius, speed <= 0.5 m/s, 0.2 s stable hold, 400 navigation ticks
(20 s), contact is a collision.

Usage:
    python3 webots/local_waypoint_transfer.py select    # write the manifest
    python3 webots/local_waypoint_transfer.py flight    # run the 32 flights
    python3 webots/local_waypoint_transfer.py report    # paired report + hashes
    python3 webots/local_waypoint_transfer.py video     # one native recording
    python3 webots/local_waypoint_transfer.py all
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import struct
import subprocess
import sys
import time

WEBOTS_DIR = pathlib.Path(__file__).resolve().parent
WORKTREE = WEBOTS_DIR.parent
sys.path.insert(0, str(WEBOTS_DIR))
import metal_scene  # noqa: E402  (obstacle nodes + the bounded-room template)

OUT_DIR = pathlib.Path(os.environ.get("RL_LOCAL_TRANSFER_OUT", str(WORKTREE / "results/local-waypoint-transfer")))
WEBOTS = pathlib.Path(os.environ.get("WEBOTS_EXECUTABLE", "/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots"))
GPU_LOCK = pathlib.Path("/Users/muadhsambul/RL/results/metal-training.lock")
PORT = "23456"
WORLD_DIR = WEBOTS_DIR / "worlds"
FLIGHT_DIR = OUT_DIR / "flights"

BANK = WORKTREE / "evidence/inputs/local-waypoint/banks/dev-a.bin"
BANK_MANIFEST = WORKTREE / "evidence/inputs/local-waypoint/banks/dev-a-128.json"
SOURCE_EVALS = WORKTREE / "results/omp-local-waypoints/evals"
FAST_EXPORT = WORKTREE / "assets/navigation-local-waypoint-experimental.bin"
FAST_CHECKPOINT = WORKTREE / "assets/checkpoints/local-waypoint-fast-experimental.bin.best"
ARRIVAL_EXPORT = WORKTREE / "assets/navigation-arrival-experimental.bin"
ARRIVAL_CHECKPOINT = WORKTREE / "assets/checkpoints/arrival-open-domain-experimental.bin.best"

# Task contract shared with the source (navigation_tasks.hpp::default config).
GOAL_RADIUS_M = 0.35
STABLE_SPEED_MPS = 0.5
STABLE_HOLD_S = 0.2
NAV_TICKS = 400
NAV_PERIOD_S = 0.05
MAX_STEPS = 2000          # 100 Hz RAPTOR steps == 400 navigation ticks == 20 s
BODY_RADIUS_M = 0.18
ROOM_BOUNDS = {"x": (-2.0, 14.0), "y": (-5.0, 5.0), "z": (0.0, 5.0)}

SELECTION_RULE = (
    "dev-a bank, fixed file order, first 16 records (indices 0..15). The bank is "
    "ordered round-robin over its six families, so this equals the first two tasks "
    "per family continued in fixed file order until sixteen: no result, score or "
    "outcome was read to choose a task."
)
MAX_STEPS_RUNTIME = 300          # wall-clock bound per flight, seconds
LOCK_WAIT_S = 900
INVALID_MARKERS = (
    "EXTERNPROTO", "could not be found", "cannot be found", "No such file",
    "does not exist", "Unable to load", "unknown controller", "controller not found",
    "segmentation fault", "controller exited with status",
    "policy declares sensor profile", "missing motor", "RAPTOR weights not found",
    "expected 20x16", "navigation policy magic mismatch", "invalid navigation actor weight",
    "policy_version must be", "sensor_profile must be", "gravity element must be",
)

ACTORS = (
    ("arrival", ARRIVAL_EXPORT, ARRIVAL_CHECKPOINT),
    ("fast", FAST_EXPORT, FAST_CHECKPOINT),
)
SOURCE_EVAL_FILE = {"arrival": "arrival-deva-mode17.csv", "fast": "fast-deva-mode17.csv"}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64_file(path: pathlib.Path) -> int:
    digest = 14695981039346656037
    for byte in path.read_bytes():
        digest ^= byte
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------- bank
#
# One kept DEV record, decoded from the source bank bytes:
#   BankFileHeader 88 bytes ("WPBANK1", version, period, count, entry_bytes, sha256)
#   BankEntry     756 bytes (WWorld 676 + start/goal state 80)
# WWorld carries `count` used obstacles; the source fills all 16 slots for the
# static scatter families but only the first `count` participate in rays,
# clearance and collision, so only those may be exported.
OBSTACLE = struct.Struct("<I9f")
BANK_HEADER = struct.Struct("<8sIIII64s")
WORLD_BODY = struct.Struct("III6f")
ENTRY_BODY = struct.Struct("10f6f4I")
ENTRY_BYTES = 756


def read_bank(path: pathlib.Path) -> list[dict]:
    data = path.read_bytes()
    magic, version, period, count, entry_bytes, entry_sha = BANK_HEADER.unpack_from(data, 0)
    if magic.rstrip(b"\0") != b"WPBANK1" or version != 1 or entry_bytes != ENTRY_BYTES:
        raise ValueError(f"{path}: unexpected bank header {magic!r} v{version} entry_bytes={entry_bytes}")
    expected = BANK_HEADER.size + count * ENTRY_BYTES
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes, expected {expected}")
    if hashlib.sha256(data[BANK_HEADER.size:]).hexdigest() != entry_sha.decode().rstrip("\0"):
        raise ValueError(f"{path}: bank entry hash mismatch")
    records = []
    for index in range(count):
        offset = BANK_HEADER.size + index * ENTRY_BYTES
        raw = data[offset:offset + ENTRY_BYTES]
        values = [OBSTACLE.unpack_from(raw, i * OBSTACLE.size) for i in range(16)]
        world = WORLD_BODY.unpack_from(raw, 16 * OBSTACLE.size)
        body = ENTRY_BODY.unpack_from(raw, 16 * OBSTACLE.size + WORLD_BODY.size)
        used, seed, family = world[0:3]
        records.append({
            "index": index,
            "world_count": used,
            "world_seed": seed,
            "family": family,
            "goal": list(world[3:6]),
            "wind": list(world[6:9]),
            "obstacles": [
                {"kind": values[i][0], "center": list(values[i][1:4]),
                 "half_extent": list(values[i][4:7]), "velocity": list(values[i][7:10])}
                for i in range(16)
            ],
            "start_position": list(body[0:3]),
            "start_velocity": list(body[3:6]),
            "goal_position": list(body[6:9]),
            "start_yaw": body[9],
            "clearance_start": body[10],
            "clearance_goal": body[11],
            "direct_clearance": body[12],
            "witness_clearance": body[13],
            "witness_length": body[14],
            "initial_distance": body[15],
            "family_repeat": body[16],
            "scene_seed": body[17],
            "route_class": body[18],
            "attempts": body[19],
            "record_sha256": hashlib.sha256(raw).hexdigest(),
        })
    if not records:
        raise ValueError(f"{path}: empty bank")
    return records


def usable_obstacles(record: dict) -> list[dict]:
    """Return the obstacles the source actually uses, or raise an exclusion."""
    obstacles = []
    for slot, obstacle in enumerate(record["obstacles"]):
        if slot >= record["world_count"]:
            break
        center = [finite(v, "center") for v in obstacle["center"]]
        half = [finite(v, "half_extent") for v in obstacle["half_extent"]]
        velocity = [finite(v, "velocity") for v in obstacle["velocity"]]
        kind = obstacle["kind"]
        if kind not in (0, 1, 2):
            raise ValueError(f"obstacle[{slot}]: unsupported kind {kind}")
        if any(component != 0.0 for component in velocity):
            raise ValueError(f"obstacle[{slot}]: moving geometry is not exportable as a static Webots Solid")
        dimensions = half if kind == 0 else ([half[0]] * 3 if kind == 1 else [half[0], half[0], half[2]])
        if any(component <= 0.0 for component in dimensions):
            raise ValueError(f"obstacle[{slot}]: non-positive collision dimension")
        for axis, name in enumerate("xyz"):
            if kind == 0:
                extent = half[axis]
            elif kind == 1:
                extent = half[0]
            else:
                extent = half[0] if axis < 2 else half[2]
            if center[axis] - extent < ROOM_BOUNDS[name][0] - 1e-4 or \
               center[axis] + extent > ROOM_BOUNDS[name][1] + 1e-4:
                raise ValueError(f"obstacle[{slot}]: extends outside the fixed room bounds")
        obstacles.append({"kind": kind, "center": center, "half_extent": half, "velocity": velocity})
    if any(component != 0.0 for component in record["wind"]):
        raise ValueError("record declares wind; this exporter supports static zero-wind tasks only")
    if len(obstacles) != record["world_count"]:
        raise ValueError("obstacle export did not cover the declared world_count")
    return obstacles


def validate_task(record: dict) -> None:
    start = [finite(v, "start") for v in record["start_position"]]
    goal = [finite(v, "goal") for v in record["goal_position"]]
    velocity = [finite(v, "start_velocity") for v in record["start_velocity"]]
    yaw = finite(record["start_yaw"], "start_yaw")
    if not (-math.pi - 1e-6 <= yaw <= math.pi + 1e-6) and not (0.0 <= yaw <= 2.0 * math.pi):
        raise ValueError("start_yaw outside the representable SFRotation range")
    for name, value in zip("xyz", start):
        if not (ROOM_BOUNDS[name][0] < value < ROOM_BOUNDS[name][1]):
            raise ValueError(f"start {name} outside the room")
    for name, value in zip("xyz", goal):
        if not (ROOM_BOUNDS[name][0] < value < ROOM_BOUNDS[name][1]):
            raise ValueError(f"goal {name} outside the room")
    if math.dist(start, goal) < 1.0 - 1e-6 or math.dist(start, goal) > 3.0 + 1e-6:
        raise ValueError(f"goal distance {math.dist(start, goal):.4f} outside the 1-3 m local band")
    if math.sqrt(sum(v * v for v in velocity)) > 1.0 + 1e-6:
        raise ValueError("start speed above the 1 m/s bank envelope")
    distance = math.dist(start, goal)
    if abs(distance - record["initial_distance"]) > 1e-4:
        raise ValueError("record initial_distance does not match its endpoints")


# ------------------------------------------------------------------ assets
def read_navpol1(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    if data[:8] != b"NAVPOL1\0":
        raise ValueError(f"{path}: not a NAVPOL1 asset (magic {data[:8]!r})")
    version, obs, hidden, action, mode, weights = struct.unpack_from("<6I", data, 8)
    max_speed, = struct.unpack_from("<f", data, 8 + 24)
    rows, cols = struct.unpack_from("<2I", data, 36)
    range_max, nav_period, native_period = struct.unpack_from("<3f", data, 44)
    memory_frames, body_frame = struct.unpack_from("<2I", data, 56)
    if (version, obs, hidden, action, mode, weights) != (1, 184, 64, 4, 17, 12104):
        raise ValueError(f"{path}: unexpected NAVPOL1 metadata {(version, obs, hidden, action, mode, weights)}")
    if (rows, cols, memory_frames, body_frame) != (16, 20, 8, 0x00554C46):
        raise ValueError(f"{path}: unexpected sensor grid / frame tag")
    if abs(nav_period - 0.05) > 1e-6 or abs(range_max - 12.0) > 1e-6:
        raise ValueError(f"{path}: unexpected navigation period or range maximum")
    weights_bytes = data[8 + 56:8 + 56 + weights * 4]
    for value in struct.unpack(f"<{weights}f", weights_bytes):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite actor weight")
    if data[-16:-8] != b"NAVSRC1\0":
        raise ValueError(f"{path}: missing source provenance trailer")
    source_hash, = struct.unpack_from("<Q", data, len(data) - 8)
    return {
        "path": str(path), "sha256": sha256_file(path), "bytes": len(data),
        "magic": "NAVPOL1", "version": version, "observation_count": obs,
        "hidden_count": hidden, "action_count": action, "policy_mode": mode,
        "weight_count": weights, "max_speed_mps": max_speed,
        "sensor_grid": f"{rows}x{cols}", "range_max_m": range_max,
        "navigation_period_s": nav_period, "native_period_s": native_period,
        "memory_frames": memory_frames, "body_frame": "FLU",
        "source_checkpoint_hash_fnv1a64": source_hash,
    }


def check_source_provenance(asset: dict, checkpoint: pathlib.Path) -> dict:
    declared = asset["source_checkpoint_hash_fnv1a64"]
    actual = fnv1a64_file(checkpoint)
    if declared != actual:
        raise ValueError(f"{asset['path']}: export provenance does not match {checkpoint}")
    return {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
            "fnv1a64": actual, "matches_export": True}


# -------------------------------------------------------------- selection
def select_records() -> tuple[list[dict], dict]:
    """Fix the sixteen predeclared tasks, labeling any unexportable task here.

    An exclusion is recorded before a single Webots process starts and drops
    that task from BOTH actors; it is never replaced by another task and never
    counted as a success.
    """
    if not BANK.is_file():
        raise SystemExit(f"missing kept source bank {BANK}")
    bank_sha = sha256_file(BANK)
    bank_manifest = json.loads(BANK_MANIFEST.read_text())
    if bank_manifest.get("bank_sha256") != bank_sha:
        raise SystemExit(f"{BANK} does not hash to its own manifest ({bank_manifest.get('bank_sha256')})")
    records = read_bank(BANK)
    if len(records) < 16:
        raise SystemExit("dev-a bank holds fewer than sixteen records")
    exclusions = []
    kept = []
    for record in records[:16]:
        try:
            validate_task(record)
            obstacles = usable_obstacles(record)
        except ValueError as error:
            exclusions.append({"index": record["index"], "family": record["family"],
                               "family_name": family_name(record["family"]),
                               "scene_seed": record["scene_seed"], "reason": str(error)})
            continue
        record["obstacles_used"] = obstacles
        record["family_name"] = family_name(record["family"])
        kept.append(record)
    if not kept:
        raise SystemExit("no predeclared task could be exported: " + json.dumps(exclusions))
    return kept, {"bank_path": str(BANK), "bank_sha256": bank_sha,
                  "bank_manifest": str(BANK_MANIFEST), "bank_records": len(records),
                  "selection_rule": SELECTION_RULE,
                  "predeclared_tasks": len(records[:16]), "kept_tasks": len(kept),
                  "exclusions_before_flights": exclusions}


def family_name(family: int) -> str:
    return {0: "open", 1: "boxes", 2: "poles", 4: "doorway", 5: "table",
            14: "bent_hallway", 15: "connected_rooms", 16: "vertical_choices",
            }.get(family, f"family{family}")


def source_rows(records: list[dict]) -> dict:
    """Read the kept source evaluation rows for exactly these bank records."""
    rows = {}
    for actor, filename in SOURCE_EVAL_FILE.items():
        path = SOURCE_EVALS / filename
        with path.open(newline="") as handle:
            table = list(csv.DictReader(handle))
        by_env = {int(row["env"]): row for row in table}
        selected = []
        for record in records:
            row = by_env.get(record["index"])
            if row is None:
                raise SystemExit(f"{path}: no row for env {record['index']}")
            if int(row["family"]) != record["family"] or int(row["scene_seed"]) != record["scene_seed"]:
                raise SystemExit(f"{path}: env {record['index']} is not the kept bank record")
            if abs(float(row["start_x"]) - record["start_position"][0]) > 1e-6 or \
               abs(float(row["goal_z"]) - record["goal_position"][2]) > 1e-6 or \
               abs(float(row["start_yaw"]) - record["start_yaw"]) > 1e-6:
                raise SystemExit(f"{path}: env {record['index']} geometry differs from the bank bytes")
            selected.append({
                "env": record["index"], "family": record["family"],
                "family_name": record["family_name"], "route_class": record["route_class"],
                "success": int(row["success"]), "collision": int(row["collision"]),
                "timeout": int(row["timeout"]), "time_s": float(row["time_s"]),
                "path_m": float(row["path_m"]), "mean_speed_mps": float(row["mean_speed_mps"]),
                "min_clearance_m": float(row["min_clearance_m"]),
                "final_speed_mps": float(row["final_speed_mps"]),
                "stable_hold_s": float(row["stable_hold_s"]),
            })
        rows[actor] = {"file": str(path), "sha256": sha256_file(path), "rows": selected,
                       "full_split": {
                           "tasks": len(table),
                           "success": sum(int(row["success"]) for row in table),
                           "collision": sum(int(row["collision"]) for row in table),
                           "timeout": sum(int(row["timeout"]) for row in table),
                           "mean_arrival_s": (sum(float(row["time_s"]) for row in table if int(row["success"]))
                                              / max(1, sum(int(row["success"]) for row in table))),
                       }}
    return rows


def write_selection_manifest() -> None:
    if not SOURCE_EVALS.is_dir():
        SOURCE_EVALS.mkdir(parents=True, exist_ok=True)
        table = list(csv.DictReader((WORKTREE / "evidence/inputs/local-waypoint/evaluation-records.csv").open()))
        for actor in ("arrival", "fast"):
            filename = f"{actor}-deva-mode17.csv"
            selected = [dict(row) for row in table if row["record_file"] == filename]
            if len(selected) != 128:
                raise SystemExit(f"published comparator {filename} must contain128records")
            for row in selected:
                del row["record_file"]
            with (SOURCE_EVALS / filename).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
                writer.writeheader(); writer.writerows(selected)
    records, bank_info = select_records()
    assets = {}
    for actor, export, checkpoint in ACTORS:
        header = read_navpol1(export)
        header["provenance"] = check_source_provenance(header, checkpoint)
        assets[actor] = header
    source = source_rows(records)
    manifest = {
        "schema": "local-waypoint-transfer-selection-v1",
        "written_before_any_flight": True,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_rule": SELECTION_RULE,
        "bank": bank_info,
        "task_contract": {
            "goal_radius_m": GOAL_RADIUS_M, "stable_speed_mps": STABLE_SPEED_MPS,
            "stable_hold_s": STABLE_HOLD_S, "nav_period_s": NAV_PERIOD_S,
            "nav_ticks": NAV_TICKS, "budget_s": NAV_TICKS * NAV_PERIOD_S,
            "raptor_steps": MAX_STEPS, "raptor_hz": 100, "navigation_hz": 20,
            "collision_sphere_m": BODY_RADIUS_M,
            "grading": "navigation_tasks.hpp::navigation_task_step inside the mission-owned controller",
        },
        "physics": {
            "policy": "frozen RAPTOR at 100 Hz, nominal rotors, L2F motor lag",
            "collision": "0.18 m sphere in the RaptorCrazyflie PROTO",
            "sensors": "20x16 RangeFinder, ego GPS/IMU/Gyro, legacy profile (0.75, body origin)",
            "initial_state": "world file declares pose/yaw; Supervisor applies the bank start velocity physically at reset",
        },
        "actors": assets,
        "source_comparator": source,
        "planned_flights": [
            {"slug": flight_slug(record, actor), "index": record["index"],
             "family": record["family"], "family_name": record["family_name"],
             "route_class": record["route_class"], "scene_seed": record["scene_seed"],
             "actor": actor, "world": world_path(flight_slug(record, actor)).name}
            for record in records for actor, _, _ in ACTORS
        ],
        "primary_endpoint": "success count difference, fast minus arrival, over the sixteen paired tasks",
        "transfer_claim_rule": "a source gain counts as transferred only if the fast actor wins the primary endpoint over the paired flights with a valid denominator of sixteen tasks per actor",
        "video_rule": "if the transfer claim holds, record one native Webots movie of the first manifest-order fast-actor flight that reached stable hold; the recording is a re-run of that same world and must reach the same terminal verdict",
    }
    write_json(OUT_DIR / "selection-manifest.json", manifest)
    print(f"selection manifest={OUT_DIR / 'selection-manifest.json'} tasks={len(records)} "
          f"flights={len(manifest['planned_flights'])} bank={bank_info['bank_sha256'][:16]}")


def flight_slug(record: dict, actor: str) -> str:
    return f"lwt-env{record['index']:03d}-{record['family_name']}-{actor}"


# ------------------------------------------------------------------ worlds
def world_path(slug: str) -> pathlib.Path:
    return WORLD_DIR / f"local-waypoint-transfer-{slug}.wbt"


def write_world(record: dict, actor: str, policy_path: pathlib.Path, movie: pathlib.Path | None,
                slug: str | None = None) -> pathlib.Path:
    """Write one flight world. `slug` names both the world file and the
    controller's artifacts, so a recording run can never write into the
    scored flight's receipt paths."""
    slug = slug or flight_slug(record, actor)
    template = metal_scene.BOUNDED_TEMPLATE.read_text()
    marker = "# WEBOTS_OBSTACLES_INSERTION_POINT"
    if template.count(marker) != 1:
        raise ValueError("bounded room template must contain one obstacle insertion marker")
    text, replaced = re.subn(r"\bbasicTimeStep\s+10\b", "basicTimeStep 1", template, count=1)
    if replaced != 1:
        raise ValueError("bounded room template must specify one 10 ms physics step")
    text = text.replace(marker, metal_scene.obstacle_nodes(record["obstacles_used"]), 1)
    start = record["start_position"]
    yaw = record["start_yaw"]
    velocity = record["start_velocity"]
    policy = policy_path.resolve().relative_to(WORKTREE).as_posix()
    config = {
        "phase": "navigation",
        "seed": str(record["scene_seed"]),
        "policy": f"../{policy}",
        "speed": "1.5",
        "distance": f"{record['initial_distance']:.9g}",
        "goal": ",".join(f"{value:.9g}" for value in record["goal_position"]),
        "max_steps": str(MAX_STEPS),
        "profile": "hover",
        "motor_sampling": "average",
        "goal_objective": "hold",
        "sensor_profile": "legacy",
        "policy_version": "1",
        "start_yaw": f"{yaw:.9g}",
        "start_position": ",".join(f"{value:.9g}" for value in start),
        "start_velocity": ",".join(f"{value:.9g}" for value in velocity),
        "run_slug": slug,
        "capture_trajectory": "1",
    }
    if movie is not None:
        config["movie_file"] = str(movie)
        config["view_snapshot_file"] = str(movie.with_suffix(".view.png"))
    custom = ";".join(f"{key}={value}" for key, value in config.items())
    robot = (f'RaptorCrazyflie {{ translation {metal_scene.vrml_float(start[0])} {metal_scene.vrml_float(start[1])} '
             f'{metal_scene.vrml_float(start[2])} rotation 0 0 1 {metal_scene.vrml_float(yaw)} '
             f'name "RaptorCrazyflie" controller "local_waypoint_transfer" supervisor TRUE '
             f'customData "{custom}" }}')
    text, replaced = re.subn(r'RaptorCrazyflie \{[^}]*\}', robot, text, count=1)
    if replaced != 1:
        raise ValueError("template must contain exactly one RaptorCrazyflie instance")
    text = re.sub(r'title "[^"]*"', f'title "{slug}"', text, count=1)
    if movie is not None:
        midpoint = [(start[i] + record["goal_position"][i]) * 0.5 for i in range(3)]
        midpoint[2] += 0.6
        camera = (-1.5, -3.5, 4.5)
        text = benchmark_recording_viewpoint(text, midpoint, camera)
    path = world_path(slug)
    path.write_text(text)
    return path


def benchmark_recording_viewpoint(source: str, target, camera) -> str:
    """A fixed wide 3D view of the route; observer only, never an actor input."""
    forward = [target[i] - camera[i] for i in range(3)]
    length = math.sqrt(sum(value * value for value in forward))
    forward = [value / length for value in forward]
    left = (-forward[1], forward[0], 0.0)
    left_norm = math.sqrt(sum(value * value for value in left))
    left = (left[0] / left_norm, left[1] / left_norm, 0.0)
    up = (forward[1] * left[2] - forward[2] * left[1],
          forward[2] * left[0] - forward[0] * left[2],
          forward[0] * left[1] - forward[1] * left[0])
    matrix = [[forward[0], left[0], up[0]], [forward[1], left[1], up[1]], [forward[2], left[2], up[2]]]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        qw, qx, qy, qz = 0.25 * scale, (matrix[2][1] - matrix[1][2]) / scale, \
            (matrix[0][2] - matrix[2][0]) / scale, (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
        qw, qx, qy, qz = (matrix[1][0] + matrix[0][1]) / scale, 0.25 * scale, \
            (matrix[0][2] + matrix[2][0]) / scale, (matrix[2][1] - matrix[1][2]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
        qw, qx, qy, qz = (matrix[0][2] - matrix[2][0]) / scale, (matrix[1][0] + matrix[0][1]) / scale, \
            0.25 * scale, (matrix[2][1] - matrix[1][2]) / scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    angle = 2 * math.acos(max(-1.0, min(1.0, qw)))
    sine = math.sqrt(max(1e-16, 1.0 - qw * qw))
    axis = (qx / sine, qy / sine, qz / sine)
    viewpoint = "DEF RLRecordingViewpoint Viewpoint { position " + " ".join(f"{v:.9f}" for v in camera)
    viewpoint += " orientation " + " ".join(f"{v:.9f}" for v in axis) + f" {angle:.9f} fieldOfView 1.2 }}"
    text, replaced = re.subn(r"Viewpoint\s*\{[^}]*\}", viewpoint, source, count=1)
    if replaced != 1:
        raise ValueError("recording world must contain one Viewpoint node")
    return text


# ------------------------------------------------------------------- runs
def acquire_lock(fd: int, deadline: float) -> bool:
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.5)


def webots_command(world: pathlib.Path, record_movie: bool) -> list[str]:
    command = [str(WEBOTS), f"--port={PORT}", "--batch"]
    if record_movie:
        command += ["--mode=realtime"]
    else:
        command += ["--minimize", "--mode=fast", "--no-rendering"]
    command += ["--stdout", "--stderr", str(world)]
    return command


def run_flight(world: pathlib.Path, flight_dir: pathlib.Path, record_movie: bool) -> dict:
    flight_dir.mkdir(parents=True, exist_ok=True)
    slug = flight_dir.name
    controller_dir = WEBOTS_DIR / "results"
    receipt_path = controller_dir / f"{slug}.episode.json"
    trace_path = controller_dir / f"{slug}-trace.csv"
    marker_path = controller_dir / f"{slug}.exit.marker"
    for stale in (receipt_path, trace_path, marker_path):
        stale.unlink(missing_ok=True)
    log_path = flight_dir / "webots.log"
    status = {"invalid": False, "reason": ""}
    command = webots_command(world, record_movie)
    with log_path.open("w") as log:
        environment = os.environ.copy()
        if sys.platform == "darwin" and not record_movie:
            # Batch/minimize still lets Qt promote each new macOS instance.
            # Keep its Cocoa/OpenGL backend but disable foreground promotion.
            environment["QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM"] = "1"
        process = subprocess.Popen(command, cwd=str(WEBOTS_DIR), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   close_fds=True, env=environment)
        write_json(flight_dir / "process.json", {"pid": process.pid,
                   "argv": command, "cwd": str(WEBOTS_DIR), "state": "running"})
        try:
            deadline = time.monotonic() + MAX_STEPS_RUNTIME
            while True:
                if process.poll() is not None:
                    break
                if time.monotonic() > deadline:
                    status.update(invalid=True, reason="wall-clock timeout")
                    break
                try:
                    seen = log_path.read_text(errors="replace")
                except OSError:
                    seen = ""
                marker = next((item for item in INVALID_MARKERS if item in seen), None)
                if marker:
                    status.update(invalid=True, reason=f"log marker: {marker}")
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=15)
            write_json(flight_dir / "process.json", {"pid": process.pid,
                       "argv": command, "cwd": str(WEBOTS_DIR),
                       "state": "exited", "exit_code": process.returncode})
    returncode = process.returncode
    log_text = log_path.read_text(errors="replace")
    if not status["invalid"]:
        marker = next((item for item in INVALID_MARKERS if item in log_text), None)
        if marker:
            status.update(invalid=True, reason=f"log marker: {marker}")
    receipt = None
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        (flight_dir / "episode.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if trace_path.is_file():
        (flight_dir / "trace.csv").write_text(trace_path.read_text())
    if marker_path.is_file():
        (flight_dir / "exit.marker").write_text(marker_path.read_text())
    if status["invalid"]:
        return {"valid": False, "invalid_reason": status["reason"], "exit_code": returncode,
                "receipt": receipt}
    if returncode != 0:
        return {"valid": False, "invalid_reason": f"Webots exit code {returncode}",
                "exit_code": returncode, "receipt": receipt}
    if receipt is None:
        return {"valid": False, "invalid_reason": "no controller episode receipt",
                "exit_code": returncode, "receipt": None}
    if not receipt.get("navigation_loaded"):
        return {"valid": False, "invalid_reason": "navigation policy did not load",
                "exit_code": returncode, "receipt": receipt}
    if not receipt.get("raptor_loaded"):
        return {"valid": False, "invalid_reason": "RAPTOR weights did not load",
                "exit_code": returncode, "receipt": receipt}
    if receipt.get("grading") != "navigation_task_step":
        return {"valid": False, "invalid_reason": "receipt is not graded by the shared task function",
                "exit_code": returncode, "receipt": receipt}
    if receipt.get("sensor_profile") != "legacy" or receipt.get("policy_version") != 1:
        return {"valid": False, "invalid_reason": "receipt left the legacy 184-input contract",
                "exit_code": returncode, "receipt": receipt}
    if not trace_path.is_file():
        return {"valid": False, "invalid_reason": "controller did not write its flight telemetry",
                "exit_code": returncode, "receipt": receipt}
    terminal = sum(int(receipt.get(key, 0)) for key in ("success", "collision", "timeout"))
    if terminal < 1:
        return {"valid": False, "invalid_reason": "receipt has no terminal outcome",
                "exit_code": returncode, "receipt": receipt}
    if receipt.get("success") and receipt.get("collision"):
        return {"valid": False, "invalid_reason": "receipt claims success and contact",
                "exit_code": returncode, "receipt": receipt}
    if int(receipt.get("nav_task_ticks", 0)) < 1:
        return {"valid": False, "invalid_reason": "shared grader never stepped",
                "exit_code": returncode, "receipt": receipt}
    return {"valid": True, "invalid_reason": "", "exit_code": returncode, "receipt": receipt}


# -------------------------------------------------------------- telemetry
def clearance_m(position, obstacles: list[dict]) -> float:
    """world.hpp::wclearance translated to Python (room walls, obstacles, 0.18 m body)."""
    x, y, z = position
    best = min(x + 2.0, 14.0 - x, y + 5.0, 5.0 - y, z, 5.0 - z) - BODY_RADIUS_M
    for obstacle in obstacles:
        center, half = obstacle["center"], obstacle["half_extent"]
        qx, qy, qz = x - center[0], y - center[1], z - center[2]
        kind = obstacle["kind"]
        if kind == 1:
            distance = math.sqrt(qx * qx + qy * qy + qz * qz) - half[0]
        elif kind == 2:
            radial = math.sqrt(qx * qx + qy * qy) - half[0]
            axial = abs(qz) - half[2]
            distance = (math.sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
                        + min(max(radial, axial), 0.0))
        else:
            kx, ky, kz = abs(qx) - half[0], abs(qy) - half[1], abs(qz) - half[2]
            distance = (math.sqrt(max(kx, 0.0) ** 2 + max(ky, 0.0) ** 2 + max(kz, 0.0) ** 2)
                        + min(max(kx, ky, kz), 0.0))
        best = min(best, distance - BODY_RADIUS_M)
    return best


def read_trace(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty flight trace")
    samples = []
    for row in rows:
        sample = {key: float(row[key]) for key in row if key not in ("step",)}
        if not all(math.isfinite(value) for value in sample.values()):
            raise ValueError(f"{path}: non-finite telemetry at step {row['step']}")
        samples.append(sample)
    return samples


def applied_reset_velocity(log_path: pathlib.Path, record: dict) -> list[float]:
    """Read the Supervisor's t=0 verdict for the physical start velocity.

    The first telemetry row is written AFTER the first 10 ms physics step, so
    gravity has already added 9.81e-2 m/s to vz by then. The reset velocity is
    only observable at t=0, which the controller prints from
    wb_supervisor_node_get_velocity immediately after setting it.
    """
    if not log_path.is_file():
        raise SystemExit(f"{log_path}: flight log missing; cannot validate the reset state")
    text = log_path.read_text(errors="replace")
    match = re.search(r"WEBOTS_INITIAL_VELOCITY requested=([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+) "
                      r"applied=([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)", text)
    if match is None:
        raise SystemExit(f"{log_path}: controller never reported its applied reset velocity")
    for marker in ("WEBOTS_POLICY ", "WEBOTS_MODELS ", "WEBOTS_SENSOR_PROFILE "):
        if marker not in text:
            raise SystemExit(f"{log_path}: controller did not report {marker.strip()}")
    applied = [float(match.group(4 + axis)) for axis in range(3)]
    requested = [float(match.group(1 + axis)) for axis in range(3)]
    expected = record["start_velocity"]
    error = max(abs(applied[axis] - expected[axis]) for axis in range(3))
    requested_error = max(abs(requested[axis] - expected[axis]) for axis in range(3))
    if error > 1e-5 or requested_error > 1e-5:
        raise SystemExit(f"{log_path}: applied reset velocity {applied} does not match "
                         f"the bank record {expected}")
    return applied


def flight_record(record: dict, actor: str, run: dict) -> dict:
    slug = flight_slug(record, actor)
    receipt = run["receipt"] or {}
    flight_dir = FLIGHT_DIR / slug
    row = {
        "slug": slug, "index": record["index"], "family": record["family"],
        "family_name": record["family_name"], "scene_seed": record["scene_seed"],
        "route_class": record["route_class"], "actor": actor,
        "valid": run["valid"], "invalid_reason": run["invalid_reason"],
        "exit_code": run["exit_code"],
        "goal_distance_m": record["initial_distance"],
        "start_clearance_m": record["clearance_start"],
        "world": str(world_path(slug)), "flight_dir": str(flight_dir),
    }
    if not run["valid"]:
        row.update({"success": "", "collision": "", "timeout": ""})
        return row
    samples = read_trace(flight_dir / "trace.csv")
    first = samples[0]
    start_position = record["start_position"]
    applied_velocity = applied_reset_velocity(flight_dir / "webots.log", record)
    yaw = 2.0 * math.atan2(first["q_z"], first["q_w"])
    yaw_error = abs(math.atan2(math.sin(yaw - record["start_yaw"]), math.cos(yaw - record["start_yaw"])))
    position_error = math.dist((first["x"], first["y"], first["z"]), start_position)
    # The trace starts after the first 10 ms physics step, so the body has had
    # exactly one step of thrust, gravity and propeller reaction torque.
    if position_error > 0.03 or yaw_error > 0.02:
        raise SystemExit(f"{slug}: start pose does not follow the bank record "
                         f"(position {position_error:.5f} m, yaw {yaw_error:.5f} rad after one step)")
    if abs(first["goal_error"] - record["initial_distance"]) > 0.05:
        raise SystemExit(f"{slug}: first telemetry goal distance differs from the bank record")
    min_clearance = min(clearance_m((sample["x"], sample["y"], sample["z"]),
                                    record["obstacles_used"]) for sample in samples)
    time_s = float(receipt["time_s"])
    row.update({
        "success": int(bool(receipt["success"])), "collision": int(bool(receipt["collision"])),
        "timeout": int(bool(receipt["timeout"])),
        "time_s": time_s,
        "arrival_s": time_s if receipt["success"] else "",
        "path_m": float(receipt["path_m"]),
        "mean_speed_mps": float(receipt["path_m"]) / time_s if time_s > 1e-9 else 0.0,
        "peak_speed_mps": float(receipt["peak_speed_mps"]),
        "final_speed_mps": float(receipt["final_world_speed_mps"]),
        "min_clearance_m": min_clearance,
        "min_sensor_range_m": float(receipt["min_sensor_range_m"]),
        "final_error_m": float(receipt["final_error_m"]),
        "stable_hold_s": float(receipt["stable_hold_s"]),
        "nav_task_ticks": int(receipt["nav_task_ticks"]),
        "first_entry_s": float(receipt["goal_radius_entry_first_time_s"]),
        "first_entry_count": int(receipt["goal_radius_entry_count"]),
        "raptor_steps": int(receipt["steps"]),
        "trace_samples": len(samples),
        "initial_position_error_m": position_error,
        "initial_velocity_applied_mps": applied_velocity,
        "initial_velocity_error_mps": max(abs(applied_velocity[i] - record["start_velocity"][i])
                                          for i in range(3)),
        "initial_yaw_error_rad": yaw_error,
        "start_speed_mps": math.sqrt(sum(v * v for v in record["start_velocity"])),
    })
    return row


def run_all_flights() -> None:
    manifest_path = OUT_DIR / "selection-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("run `select` first: the selection manifest must exist before any flight")
    manifest = json.loads(manifest_path.read_text())
    records, _ = select_records()
    record_by_index = {record["index"]: record for record in records}
    if not WEBOTS.is_file():
        raise SystemExit(f"Webots executable missing: {WEBOTS}")
    lock_fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    os.set_inheritable(lock_fd, False)
    if not acquire_lock(lock_fd, time.monotonic() + LOCK_WAIT_S):
        os.close(lock_fd)
        raise SystemExit("could not acquire the shared lock within the bounded wait")
    rows = []
    try:
        for planned in manifest["planned_flights"]:
            record = record_by_index[planned["index"]]
            actor = planned["actor"]
            policy = dict((name, pathlib.Path(path)) for name, path, _ in ACTORS)[actor]
            world = write_world(record, actor, policy, movie=None)
            slug = flight_slug(record, actor)
            started = time.monotonic()
            run = run_flight(world, FLIGHT_DIR / slug, record_movie=False)
            row = flight_record(record, actor, run)
            row["wall_s"] = round(time.monotonic() - started, 3)
            row["world_sha256"] = sha256_file(world)
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("slug", "actor", "valid", "success", "collision", "timeout",
                               "arrival_s", "min_clearance_m", "invalid_reason")}), flush=True)
    finally:
        os.close(lock_fd)
    write_json(OUT_DIR / "flights.json", rows)
    valid = [row for row in rows if row["valid"]]
    print(f"flights rows={len(rows)} valid={len(valid)} output={OUT_DIR / 'flights.json'}")


# ------------------------------------------------------------------ report
def actor_summary(rows: list[dict], actor: str) -> dict:
    flights = [row for row in rows if row["actor"] == actor and row["valid"]]
    failures = [row for row in rows if row["actor"] == actor and not row["valid"]]
    successes = [row for row in flights if row["success"] == 1]
    collisions = [row for row in flights if row["collision"] == 1]
    arrivals = [row["arrival_s"] for row in flights if row["arrival_s"] != ""]
    return {
        "flights": len([row for row in rows if row["actor"] == actor]),
        "valid": len(flights), "invalid": len(failures),
        "success": len(successes), "collision": len(collisions),
        "timeout": len(flights) - len(successes) - len(collisions),
        "success_rate": len(successes) / len(flights) if flights else 0.0,
        "collision_rate": len(collisions) / len(flights) if flights else 0.0,
        "mean_arrival_s": sum(arrivals) / len(arrivals) if arrivals else "",
        "mean_path_m": sum(row["path_m"] for row in flights) / len(flights) if flights else 0.0,
        "mean_speed_mps": sum(row["mean_speed_mps"] for row in flights) / len(flights) if flights else 0.0,
        "min_clearance_m": min((row["min_clearance_m"] for row in flights), default=None),
        "arrival_successes": [row["index"] for row in successes],
        "collisions": [row["index"] for row in collisions],
        "timeouts": [row["index"] for row in flights if row["timeout"] == 1],
        "invalid_reasons": [row["invalid_reason"] for row in failures],
    }


def reference_error_m(flight_dir: pathlib.Path, record: dict) -> float:
    """Distance between the RAPTOR pose reference the controller seeded and
    the bank record's start pose. The source seeds it at the start pose; any
    nonzero value here would be a fixed reference error injected at reset."""
    log_path = flight_dir / "webots.log"
    if not log_path.is_file():
        raise SystemExit(f"{log_path}: flight log missing; cannot verify the seeded reference")
    match = re.search(r"WEBOTS_REFERENCE start=([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+) yaw=([-\d.eE+]+)",
                      log_path.read_text(errors="replace"))
    if match is None:
        raise SystemExit(f"{log_path}: controller never reported its seeded RAPTOR reference")
    seeded = [float(match.group(1 + axis)) for axis in range(3)]
    yaw = float(match.group(4))
    pose_error = math.dist(seeded, record["start_position"])
    yaw_error = abs(math.atan2(math.sin(yaw - record["start_yaw"]), math.cos(yaw - record["start_yaw"])))
    # The controller prints these with %g, so the observable precision is ~1e-5.
    if pose_error > 1e-3 or yaw_error > 1e-4:
        raise SystemExit(f"{log_path}: seeded reference {seeded}/{yaw} does not match the bank record")
    return pose_error


def task_comparison(rows: list[dict], manifest: dict, records: list[dict]) -> list[dict]:
    """Join Webots, source-Metal and bank rows for the same record identifier."""
    record_by_index = {record["index"]: record for record in records}
    flights = {(row["index"], row["actor"]): row for row in rows if row["valid"]}
    source = manifest["source_comparator"]
    source_by_actor = {actor: {row["env"]: row for row in payload["rows"]}
                       for actor, payload in source.items()}
    comparison = []
    for index in sorted(record_by_index):
        record = record_by_index[index]
        entry = {"index": index, "family": record["family"],
                 "family_name": record["family_name"], "route_class": record["route_class"],
                 "goal_distance_m": record["initial_distance"],
                 "start_clearance_m": record["clearance_start"]}
        for actor in ("arrival", "fast"):
            flight = flights.get((index, actor))
            origin = source_by_actor[actor][index]
            entry[f"webots_{actor}"] = {
                "success": flight["success"], "collision": flight["collision"],
                "timeout": flight["timeout"], "arrival_s": flight["arrival_s"],
                "path_m": flight["path_m"], "min_clearance_m": flight["min_clearance_m"],
                "final_speed_mps": flight["final_speed_mps"],
            } if flight else None
            entry[f"source_{actor}"] = {
                "success": origin["success"], "collision": origin["collision"],
                "timeout": origin["timeout"], "arrival_s": origin["time_s"],
                "path_m": origin["path_m"], "min_clearance_m": origin["min_clearance_m"],
                "final_speed_mps": origin["final_speed_mps"],
            }
        comparison.append(entry)
    return comparison


def flight_diagnosis(rows: list[dict], records: list[dict]) -> dict:
    record_by_index = {record["index"]: record for record in records}
    per_flight = []
    for row in rows:
        if not row["valid"]:
            continue
        record = record_by_index[row["index"]]
        flight_dir = pathlib.Path(row["flight_dir"])
        receipt = json.loads((flight_dir / "episode.json").read_text())
        per_flight.append({
            "slug": row["slug"], "actor": row["actor"],
            "success": row["success"], "collision": row["collision"],
            "tracking_rms_mps": float(receipt["tracking_rms_mps"]),
            "reference_error_m": reference_error_m(flight_dir, record),
            "path_ratio": row["path_m"] / row["goal_distance_m"],
            "first_entry_count": int(receipt["goal_radius_entry_count"]),
            "first_entry_s": float(receipt["goal_radius_entry_first_time_s"]),
            "min_clearance_m": row["min_clearance_m"],
            "mean_speed_mps": row["mean_speed_mps"],
            "arrival_s": row["arrival_s"],
            "min_sensor_range_m": row["min_sensor_range_m"],
        })
    aggregate = {}
    for actor in ("arrival", "fast"):
        selected = [row for row in per_flight if row["actor"] == actor]
        aggregate[actor] = {
            "flights": len(selected),
            "mean_tracking_rms_mps": sum(row["tracking_rms_mps"] for row in selected) / len(selected),
            "max_reference_error_m": max(row["reference_error_m"] for row in selected),
            "mean_path_ratio": sum(row["path_ratio"] for row in selected) / len(selected),
            "mean_first_entries": sum(row["first_entry_count"] for row in selected) / len(selected),
            "mean_min_clearance_m": sum(row["min_clearance_m"] for row in selected) / len(selected),
            "mean_min_sensor_range_m": sum(row["min_sensor_range_m"] for row in selected) / len(selected),
            "contact_mean_clearance_m": (sum(row["min_clearance_m"] for row in selected if row["collision"])
                                         / max(1, sum(1 for row in selected if row["collision"])))
            if any(row["collision"] for row in selected) else None,
            "success_mean_clearance_m": (sum(row["min_clearance_m"] for row in selected if row["success"])
                                         / max(1, sum(1 for row in selected if row["success"])))
            if any(row["success"] for row in selected) else None,
        }
    return {"per_flight": per_flight, "aggregate": aggregate}


def build_report() -> None:
    rows = json.loads((OUT_DIR / "flights.json").read_text())
    manifest = json.loads((OUT_DIR / "selection-manifest.json").read_text())
    kept = manifest["bank"]["kept_tasks"]
    valid = [row for row in rows if row["valid"]]
    if len(valid) != 2 * kept:
        raise SystemExit(f"{2 * kept} valid flights required for the paired report, got {len(valid)}; "
                         "failures stay in flights.json")
    summary = {actor: actor_summary(rows, actor) for actor, _, _ in ACTORS}
    fast, arrival = summary["fast"], summary["arrival"]
    paired = []
    by_key = {(row["index"], row["actor"]): row for row in valid}
    for planned in manifest["planned_flights"][::2]:
        record = by_key[(planned["index"], "arrival")]
        other = by_key[(planned["index"], "fast")]
        paired.append({
            "index": planned["index"], "family": record["family"],
            "family_name": record["family_name"], "route_class": record["route_class"],
            "arrival": {key: record[key] for key in
                        ("success", "collision", "timeout", "arrival_s", "mean_speed_mps",
                         "min_clearance_m", "final_speed_mps", "path_m")},
            "fast": {key: other[key] for key in
                     ("success", "collision", "timeout", "arrival_s", "mean_speed_mps",
                      "min_clearance_m", "final_speed_mps", "path_m")},
            "success_difference": other["success"] - record["success"],
            "collision_difference": other["collision"] - record["collision"],
        })
    source = manifest["source_comparator"]
    source_summary = {}
    for actor, payload in source.items():
        source_rows_data = payload["rows"]
        source_summary[actor] = {
            "tasks": len(source_rows_data),
            "success": sum(row["success"] for row in source_rows_data),
            "collision": sum(row["collision"] for row in source_rows_data),
            "timeout": sum(row["timeout"] for row in source_rows_data),
            "mean_arrival_s": sum(row["time_s"] for row in source_rows_data
                                  if row["success"]) / max(1, sum(row["success"] for row in source_rows_data)),
            "min_clearance_m": min(row["min_clearance_m"] for row in source_rows_data),
        }
    success_difference = fast["success"] - arrival["success"]
    collision_difference = fast["collision"] - arrival["collision"]
    denominator_ok = kept == 16
    transfer = success_difference > 0 and denominator_ok
    records, _ = select_records()
    payload = {
        "schema": "local-waypoint-transfer-results-v1",
        "paired_tasks": len(paired),
        "predeclared_tasks": 16,
        "kept_tasks": kept,
        "exclusions_before_flights": manifest["bank"]["exclusions_before_flights"],
        "valid_flights": len(valid),
        "webots": summary,
        "source_comparator": source_summary,
        "source_comparator_files": {actor: {"file": source[actor]["file"],
                                            "sha256": source[actor]["sha256"]}
                                    for actor in source},
        "primary_endpoint": {
            "name": "success count difference, fast minus arrival",
            "fast_minus_arrival_success": success_difference,
            "fast_minus_arrival_collision": collision_difference,
            "denominator_ok": denominator_ok,
            "transfer_observed": transfer,
            "source_same_16_difference": source_summary["fast"]["success"] - source_summary["arrival"]["success"],
            "source_full_split_difference": (manifest["source_comparator"]["fast"]["full_split"]["success"]
                                             - manifest["source_comparator"]["arrival"]["full_split"]["success"]),
        },
        "task_comparison": task_comparison(rows, manifest, records),
        "diagnosis": flight_diagnosis(rows, records),
        "paired": paired,
    }
    write_json(OUT_DIR / "paired-results.json", payload)
    with (OUT_DIR / "flights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "source-comparator.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actor", "env", "family", "family_name",
                                                    "route_class", "success", "collision", "timeout",
                                                    "time_s", "path_m", "mean_speed_mps",
                                                    "min_clearance_m", "final_speed_mps", "stable_hold_s"])
        writer.writeheader()
        for actor in source:
            for row in source[actor]["rows"]:
                writer.writerow({"actor": actor, **row})
    print(f"report transfer={transfer} fast={fast['success']}/{fast['valid']} "
          f"arrival={arrival['success']}/{arrival['valid']} output={OUT_DIR / 'paired-results.json'}")


def write_video() -> None:
    manifest = json.loads((OUT_DIR / "selection-manifest.json").read_text())
    results = json.loads((OUT_DIR / "paired-results.json").read_text())
    if not results["primary_endpoint"]["transfer_observed"]:
        print("no transfer claim: video skipped, failure traces and denominators preserved")
        return
    rows = json.loads((OUT_DIR / "flights.json").read_text())
    by_key = {(row["index"], row["actor"]): row for row in rows if row["valid"]}
    target = None
    for planned in manifest["planned_flights"]:
        if planned["actor"] != "fast":
            continue
        row = by_key.get((planned["index"], "fast"))
        if row and row["success"] == 1:
            target = row
            break
    if target is None:
        print("no fast-actor stable hold available for the predeclared recording")
        return
    records, _ = select_records()
    record = next(item for item in records if item["index"] == target["index"])
    # The recording lives beside, never on top of, the scored flight evidence.
    slug = f"{target['slug']}-recording"
    recording_dir = FLIGHT_DIR / slug
    recording_dir.mkdir(parents=True, exist_ok=True)
    movie = recording_dir / "flight.mp4"
    world = write_world(record, "fast", pathlib.Path(FAST_EXPORT), movie=movie, slug=slug)
    lock_fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    os.set_inheritable(lock_fd, False)
    if not acquire_lock(lock_fd, time.monotonic() + LOCK_WAIT_S):
        os.close(lock_fd)
        raise SystemExit("could not acquire the shared lock for the recording")
    try:
        run = run_flight(world, recording_dir, record_movie=True)
    finally:
        os.close(lock_fd)
    receipt = run["receipt"] or {}
    verdict = {"success": int(bool(receipt.get("success"))),
               "collision": int(bool(receipt.get("collision"))),
               "timeout": int(bool(receipt.get("timeout")))}
    scored = {key: target[key] for key in ("success", "collision", "timeout")}
    receipt["recording"] = {
        "movie": str(movie),
        "movie_bytes": movie.stat().st_size if movie.is_file() else 0,
        "view_snapshot": str(recording_dir / "flight.view.png"),
        "scored_flight": target["slug"],
        "scored_verdict": scored,
        "recording_verdict": verdict,
        "reproduced_terminal": verdict == scored,
        "rerun": True,
        "rule": manifest["video_rule"],
        "world": str(world),
        "world_sha256": sha256_file(world),
    }
    write_json(recording_dir / "recording.json", receipt)
    if verdict != scored:
        raise SystemExit(f"the recording re-run did not reproduce {target['slug']}: "
                         f"{verdict} vs {scored}")
    print(f"video movie={movie} bytes={receipt['recording']['movie_bytes']} "
          f"reproduced_terminal={receipt['recording']['reproduced_terminal']}")


def write_manifest() -> None:
    """Hash every produced artifact and record the exact reproduction commands."""
    commands = [
        "WEBOTS_HOME=/Users/muadhsambul/embodied/work/Webots.app "
        "make -C webots/controllers/local_waypoint_transfer",
        "python3 webots/local_waypoint_transfer.py select",
        "python3 webots/local_waypoint_transfer.py flight",
        "python3 webots/local_waypoint_transfer.py report",
        "python3 webots/local_waypoint_transfer.py video",
        "python3 webots/local_waypoint_transfer.py manifest",
    ]
    artifacts = []
    # The mission owns these outputs. Harness/session files that live in the
    # same directory and keep changing are not mission artifacts.
    harness_names = {"events.jsonl", "mimo-events.jsonl", "mimo-stderr.log",
                     "stderr.log", "brief.md", "resume-note.md", "manifest.json"}
    for path in sorted(OUT_DIR.rglob("*")):
        if not path.is_file() or path.name in harness_names:
            continue
        if "sessions" in path.relative_to(OUT_DIR).parts:
            continue
        artifacts.append({"path": str(path.relative_to(OUT_DIR)), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    worlds = [{"path": str(path.name), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
              for path in sorted(WORLD_DIR.glob("local-waypoint-transfer-*.wbt"))]
    controller = WEBOTS_DIR / "controllers/local_waypoint_transfer/local_waypoint_transfer.cpp"
    binary = WEBOTS_DIR / "controllers/local_waypoint_transfer/local_waypoint_transfer"
    runner = WEBOTS_DIR / "local_waypoint_transfer.py"
    manifest = {
        "schema": "local-waypoint-transfer-manifest-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commands": commands,
        "webots": {"executable": str(WEBOTS), "port": int(PORT),
                   "batch_flags": "--minimize --batch --mode=fast --no-rendering",
                   "recording_flags": "--batch --mode=realtime (visible window, authorized exception)"},
        "lock": str(GPU_LOCK),
        "runner": {"source": str(runner), "source_sha256": sha256_file(runner)},
        # The core headers are shared and were being edited concurrently by the
        # root project during this mission, so record exactly what the shipped
        # binary was compiled against. navigation_tasks.hpp is the grading code
        # both the source simulator and this controller execute.
        "compiled_against": {
            name: sha256_file(WORKTREE / name)
            for name in ("navigation_tasks.hpp", "sensor_profile.hpp", "deployment.hpp",
                         "guidance.hpp", "physics.hpp", "raptor.hpp", "world.hpp",
                         "assets/raptor.bin")
        },
        "controller": {"source": str(controller), "source_sha256": sha256_file(controller),
                       "binary": str(binary), "binary_sha256": sha256_file(binary),
                       "graded_by": "navigation_tasks.hpp::navigation_task_step",
                       "sensor_profile": "legacy (NAV_SENSOR_PROFILE=1)"},
        "worlds": worlds,
        "artifacts": artifacts,
    }
    write_json(OUT_DIR / "manifest.json", manifest)
    print(f"manifest artifacts={len(artifacts)} worlds={len(worlds)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "flight", "report", "video", "manifest", "all"))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.command in ("select", "all"):
        write_selection_manifest()
    if args.command in ("flight", "all"):
        run_all_flights()
    if args.command in ("report", "all"):
        build_report()
    if args.command in ("video", "all"):
        write_video()
    if args.command == "manifest":
        write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
