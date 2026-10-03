#!/usr/bin/env python3
"""Cold generator for longer, narrower, time-varying navigation courses.

Produces a challenge-bank-compatible JSONL (schema_version=1, families 14-16)
that loads in the existing Metal CLI (`bank-eval` / `bank-witness`) without any
change to world.hpp, challenge_evaluation.hpp or the simulator.

What this adds over challenge_bank.py
--------------------------------------
* longer routes: measured witness path length, not a renamed 8 m goal
* narrow passages: doorway/lane/tunnel/window clearances driven by difficulty,
  always >= 0.06 m for the witness centreline (0.18 m body + 0.04 m margin is
  the hard contract floor of 0.04 m)
* moving obstacles: kind 1/2 bodies with non-zero `velocity`.  world.hpp moves
  them as wc(o,t) = centre + velocity*t with t = run elapsed seconds, and that
  time argument is live in depth rays, clearance and collision, so movers are
  real in every observation and in contact scoring.
* both mirror directions (coordinate_transform=mirror_y on odd indices)
* difficulty is a *stratified* parameter (12 easy / 12 medium / 12 hard per
  split per family) that monotonically drives gap width, lane width, vertical
  window, detour magnitude and mover count.
* temporal feasibility is a separate, labelled check (static witness only
  proves geometry).  Movers are phased relative to a declared nominal traversal
  speed so the declared witness route stays feasible for a privileged route
  follower while a passage meaningfully faster than nominal falls inside the
  mover blocking window.  The timing model is calibrated from a mover-free
  pilot witness run on TRAIN (`calibrate`), never from DEV outcomes.

Contract facts verified against the read-only sources (see report.md):
  * loader accepts exact family labels/contracts, coordinate_transform
    identity|mirror_y, seed == xorshift32(base_seed + i*747796405 + 2891336453),
    difficulty in [0,1) with matching band, <=16 obstacles, positive sizes,
    in-room geometry *at t=0 only*, start/goal clearance > 0.04 at t=0,
    declared direct_route_clearance < -0.02 and witness_min_clearance > 0.04,
    witness_route >= 2 points with exact start/goal endpoints.
  * unknown extra record fields are ignored by the loader, so the richer
    metadata below (archetype/course_tags/timing/movers) costs nothing.

Unknown / labelled limits (also in report.md):
  * static witness clearance is declared at t=0; mover-aware clearance is
    stored separately as witness_nominal_timing_clearance_m.
  * mover velocity is not in any policy observation; it is only inferable from
    depth history.

Example:
    python3 navigation_courses.py build --out results/mimo-courses/course-bank-v1.jsonl
    python3 navigation_courses.py build --no-movers --splits train \
        --out results/mimo-courses/pilot-static-train.jsonl
    python3 navigation_courses.py calibrate \
        --witness results/mimo-courses/pilot-witness-train.csv \
        --out results/mimo-courses/timing-model.json
    python3 navigation_courses.py build --timing-model results/mimo-courses/timing-model.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import struct
import sys
from collections import Counter
from typing import Any

from challenge_bank import (
    BODY_RADIUS_M,
    FAMILY_CONTRACTS,
    FAMILY_NAMES,
    MASK32,
    ROOM,
    SIM_ENV_OFFSET,
    SIM_ENV_STRIDE,
    SPLITS,
    WorldRng,
    add_box as _cb_add_box,
    add_doorway as _cb_add_doorway,
    add_table as _cb_add_table,
    box,
    f32,
    scene_hash,
    sim_world_seed,
    source_hash,
    xorshift32,
)

# ---------------------------------------------------------------- constants --

MARGIN_M = 0.04                      # declared safety margin (challenge contract)
MIN_WITNESS_CLEARANCE_M = 0.09       # generator target (contract floor is 0.04)
MIN_STATIC_MOVER_CLEARANCE_M = 0.12   # placement floor; adds margin for ~0.10 m follower corner-cut
MIN_TURN_CLEARANCE_M = 0.25          # sharp route turns must sit in open space
MAX_SHARP_TURN_DEG = 35.0            # turns above this need MIN_TURN_CLEARANCE_M
FILLET_RADIUS_M = 0.6                # rounded-corner radius for witness routes
MAX_OBSTACLES = 16
EVAL_BUDGET_S = 20.0                 # bank-eval: 400 steps * 0.05 s
HOLD_S = 20.0                        # movers must stay inside the room this long
MAX_ROUTE_LEN_M = 32.0

DEFAULT_TIMING = {
    "nominal_speed_mps": 0.70,   # declared nominal traversal speed of the witness route
    "inv_spread_s_per_m": 0.22,  # conservative max |1/v_act - 1/v_cal| per metre arc
    "margin_s": 0.30,            # fixed slack added to the model error bound
    "damp": 1.30,                # multiplies the measured spread before use
    "source": "default (uncalibrated conservative prior)",
}

BANDS = ("easy", "medium", "hard")

FAMILY_CONTRACT_MIRROR = {
    14: "connected partitions force south turn, east hall, then north turn",
    15: FAMILY_CONTRACTS[15],
    16: FAMILY_CONTRACTS[16],
}


def add_box(obstacles: list[dict[str, Any]], center, half) -> None:
    _cb_add_box({"obstacles": obstacles}, center, half)


def add_doorway(obstacles: list[dict[str, Any]], wall_x: float, gap_y: float,
                gap_z: float, half_gap: float, half_gap_z: float) -> None:
    _cb_add_doorway({"obstacles": obstacles}, wall_x, gap_y, gap_z, half_gap, half_gap_z)


def add_table(obstacles: list[dict[str, Any]], x: float, y: float, half_x: float,
              half_y: float, top_z: float) -> None:
    _cb_add_table({"obstacles": obstacles}, x, y, half_x, half_y, top_z)

# ------------------------------------------------------------ numeric utils --


def sub(a: float, b: float) -> float:
    return f32(a - b)


def add(a: float, b: float) -> float:
    return f32(a + b)


def mul(a: float, b: float) -> float:
    return f32(a * b)


def sphere(center: tuple[float, float, float], radius: float) -> dict[str, Any]:
    return {
        "kind": 1,
        "center": [f32(v) for v in center],
        "half_extent": [f32(radius), f32(radius), f32(radius)],
        "velocity": [0.0, 0.0, 0.0],
    }


def cylinder(center: tuple[float, float, float], radius: float, half_z: float) -> dict[str, Any]:
    return {
        "kind": 2,
        "center": [f32(v) for v in center],
        "half_extent": [f32(radius), f32(radius), f32(half_z)],
        "velocity": [0.0, 0.0, 0.0],
    }


def jitter(seed: int, index: int, tag: str) -> float:
    digest = hashlib.sha256(f"{seed}|{index}|{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def difficulty_of(seed: int, index: int, per_split: int) -> float:
    """Stratified difficulty: exactly per_split/3 levels in each band."""
    per_band = per_split // 3
    band = (index // per_band) % 3
    rank = index % per_band
    return (band + (rank + jitter(seed, index, "difficulty")) / per_band) / 3.0


def band_of(difficulty: float) -> str:
    if difficulty < 1.0 / 3.0:
        return "easy"
    if difficulty < 2.0 / 3.0:
        return "medium"
    return "hard"


def stable_base_seed(master_seed: int, split: str, family: int, per_split: int,
                     distance: float) -> int:
    key = f"mimo-courses-v1|{master_seed}|{split}|{family}|{per_split}|{distance:.4f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


# ------------------------------------------------------------- clearance math --


def point_clearance(world: dict[str, Any], p: list[float] | tuple[float, ...],
                    time: float = 0.0) -> float:
    """Exact mirror of world.hpp wclearance(): kind aware and time aware."""
    x, y, z = p
    best = min(x + 2.0, 14.0 - x, y + 5.0, 5.0 - y, z, 5.0 - z) - BODY_RADIUS_M
    for obstacle in world["obstacles"]:
        c = obstacle["center"]
        v = obstacle["velocity"]
        qx = x - (c[0] + time * v[0])
        qy = y - (c[1] + time * v[1])
        qz = z - (c[2] + time * v[2])
        size = obstacle["half_extent"]
        kind = obstacle["kind"]
        if kind == 1:
            distance = math.sqrt(qx * qx + qy * qy + qz * qz) - size[0]
        elif kind == 2:
            a = math.sqrt(qx * qx + qy * qy) - size[0]
            b = abs(qz) - size[2]
            distance = math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2) + min(max(a, b), 0.0)
        else:
            kx, ky, kz = abs(qx) - size[0], abs(qy) - size[1], abs(qz) - size[2]
            outside = math.sqrt(max(kx, 0.0) ** 2 + max(ky, 0.0) ** 2 + max(kz, 0.0) ** 2)
            distance = outside + min(max(kx, ky, kz), 0.0)
        best = min(best, distance - BODY_RADIUS_M)
    return f32(best)


def route_polyline_length(route: list[list[float]]) -> float:
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
    return total


def route_samples(route: list[list[float]], spacing: float = 0.05):
    """Yield (arc_length_at_sample, point). Arc length measured from route[0]."""
    arc = 0.0
    for a, b in zip(route, route[1:]):
        seg = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        count = max(1, int(math.ceil(seg / spacing)))
        for j in range(count):
            t = j / count
            p = [a[i] + (b[i] - a[i]) * t for i in range(3)]
            yield arc + seg * t, p
        arc += seg
    yield arc, list(route[-1])


def point_at_arc(route: list[list[float]], target: float) -> list[float]:
    arc = 0.0
    for a, b in zip(route, route[1:]):
        seg = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        if arc + seg >= target:
            t = (target - arc) / seg if seg > 0 else 0.0
            return [a[i] + (b[i] - a[i]) * t for i in range(3)]
        arc += seg
    return list(route[-1])


def route_metrics(world: dict[str, Any], route: list[list[float]]) -> dict[str, Any]:
    """Static (t=0) clearance, sample count, and measured path length."""
    minimum = math.inf
    samples = 0
    for _, p in route_samples(route):
        minimum = min(minimum, point_clearance(world, p, 0.0))
        samples += 1
    return {
        "clearance": f32(minimum),
        "samples": samples,
        "length": f32(route_polyline_length(route)),
    }


def route_nominal_timing(world: dict[str, Any], route: list[list[float]],
                         nominal_speed: float) -> dict[str, Any]:
    """Clearance along the route when each point is reached at t = arc/v_nom."""
    minimum = math.inf
    worst_arc = 0.0
    for arc, p in route_samples(route):
        clearance = point_clearance(world, p, arc / nominal_speed)
        if clearance < minimum:
            minimum = clearance
            worst_arc = arc
    return {"clearance": f32(minimum), "arc_m": f32(worst_arc)}


def fillet_route(route: list[list[float]], radius: float = FILLET_RADIUS_M,
                 max_turn_deg: float = MAX_SHARP_TURN_DEG,
                 step_deg: float = 30.0,
                 max_sag_m: float = 0.12) -> list[list[float]]:
    """Round sharp interior corners of a witness route.

    The goal-script follower turns as soon as it is within 0.35 m of a route
    vertex, so a sharp corner cuts across it; inside a corridor that ends in
    a collision.  Each corner above ``max_turn_deg`` is replaced by a tangent
    arc: per-segment turns stay within ``step_deg`` and the arc never passes
    closer than ``max_sag_m`` to the original vertex.  Degenerate corners
    (near-reversals) are left untouched; validate() then forces them to sit
    in open space instead.
    """
    if len(route) < 3:
        return [list(p) for p in route]
    out: list[list[float]] = [list(route[0])]
    for i in range(1, len(route) - 1):
        p0, p1, p2 = route[i - 1], route[i], route[i + 1]
        v0 = [p0[k] - p1[k] for k in range(3)]
        v2 = [p2[k] - p1[k] for k in range(3)]
        n0 = math.sqrt(sum(c * c for c in v0))
        n2 = math.sqrt(sum(c * c for c in v2))
        if n0 < 1e-9 or n2 < 1e-9:
            out.append(list(p1))
            continue
        u = [c / n0 for c in v0]
        w = [c / n2 for c in v2]
        dot = max(-1.0, min(1.0, sum(u[k] * w[k] for k in range(3))))
        ang = math.degrees(math.acos(dot))
        turn = 180.0 - ang
        if turn <= max_turn_deg:
            out.append(list(p1))
            continue
        bis = [u[k] + w[k] for k in range(3)]
        bn = math.sqrt(sum(c * c for c in bis))
        if bn < 1e-6:
            out.append(list(p1))
            continue
        half = math.radians(ang) / 2.0
        tangent = radius * math.tan(half)
        sag_t = max_sag_m * math.cos(half) / max(1e-6, 1.0 - math.sin(half))
        tangent = min(tangent, sag_t, 0.4 * n0, 0.4 * n2)
        curve = tangent * math.tan(half)
        reach = tangent / math.cos(half)
        centre = [p1[k] + (bis[k] / bn) * reach for k in range(3)]
        start = [p1[k] + u[k] * tangent for k in range(3)]
        end = [p1[k] + w[k] * tangent for k in range(3)]
        rs = [start[k] - centre[k] for k in range(3)]
        re = [end[k] - centre[k] for k in range(3)]
        steps = max(2, int(math.ceil(turn / step_deg)))
        out.append(start)
        for k in range(1, steps):
            s = k / steps
            q = [rs[j] + (re[j] - rs[j]) * s for j in range(3)]
            qn = math.sqrt(sum(c * c for c in q))
            if qn < 1e-9:
                continue
            out.append([centre[j] + (q[j] / qn) * curve for j in range(3)])
        out.append(end)
    out.append(list(route[-1]))
    return out


def route_direction(route: list[list[float]], arc: float) -> tuple[float, float, float]:
    a = point_at_arc(route, max(0.0, arc - 0.25))
    b = point_at_arc(route, arc + 0.25)
    d = [b[i] - a[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in d)) or 1.0
    return (d[0] / norm, d[1] / norm, d[2] / norm)


# ------------------------------------------------------------ timing model ---


def load_timing_model(path: pathlib.Path | None) -> dict[str, Any]:
    model = dict(DEFAULT_TIMING)
    if path is not None:
        model.update(json.loads(path.read_text()))
    return model


def error_bound_s(arc_m: float, model: dict[str, Any]) -> float:
    return model["damp"] * model["inv_spread_s_per_m"] * arc_m + model["margin_s"]


# ------------------------------------------------------------- archetypes ----


def goal_from_rng(rng: WorldRng, span_lo: float = 9.0, span_hi: float = 10.0) -> dict[str, float]:
    span = span_lo + (span_hi - span_lo) * rng.uniform()
    return {
        "span": span,
        "gx": span * (0.90 + 0.07 * rng.uniform()),
        "gy": (rng.uniform() - 0.5) * 3.0,
        "gz": 1.10 + 0.80 * rng.uniform(),
    }


def build_family14(rng: WorldRng, difficulty: float, archetype: int) -> dict[str, Any]:
    g = goal_from_rng(rng)
    gx, gy, gz = g["gx"], g["gy"], g["gz"]
    obstacles: list[dict[str, Any]] = []
    tags = ["long"]
    params: dict[str, Any] = {"archetype": "hall_gate_tunnel" if archetype == 0 else "hall_s_corridors"}

    lane_w = 1.22 - 0.06 * difficulty          # 1.16 .. 1.22  -> corridor slack 0.31 .. 0.34
    pole_gap = 0.64 - 0.08 * difficulty        # 0.56 .. 0.64  -> gate clearance 0.10 .. 0.14
    pole_r, pole_h = 0.14, 1.10

    if archetype == 0:
        wall_x = gx * (0.20 + 0.05 * rng.uniform())
        gap_x = gx * (0.58 + 0.06 * rng.uniform())
        lane_edge = 5.0 - lane_w
        route_y = (lane_edge + 5.18) / 2.0
        # north partition + lane south barrier (the "connected partitions")
        add_box(obstacles, (wall_x, (lane_edge - 5.0) / 2.0, 2.5),
                (0.15, (lane_edge + 5.0) / 2.0, 2.5))
        add_box(obstacles, ((wall_x + gap_x) / 2.0, lane_edge + 0.03, 2.5),
                ((gap_x - wall_x) / 2.0 + 0.15, 0.15, 2.5))
        # narrow pole gate on the northward approach column
        x_gate = 0.75 + 0.30 * rng.uniform()
        y_pole = 2.05 + 0.50 * rng.uniform()
        for sx in (-1.0, 1.0):
            obstacles.append(cylinder((x_gate + sx * (pole_gap / 2.0 + pole_r), y_pole, pole_h),
                                      pole_r, pole_h))
        # low tunnel inside the lane: floor .. ceiling squeeze, kept clear of
        # the partition crossing (wall_x) and of the south turn (gap_x)
        tun_x0 = wall_x + 1.60
        tun_len = min(1.40 + 0.80 * rng.uniform(), gap_x - 0.90 - tun_x0)
        tun_x1 = tun_x0 + tun_len
        tun_z = 0.56 + 0.12 * difficulty        # route flies at tun_z/2 -> 0.10 .. 0.16
        add_box(obstacles, ((tun_x0 + tun_x1) / 2.0, (lane_edge + 5.0) / 2.0, (tun_z + 5.0) / 2.0),
                (tun_len / 2.0, lane_w / 2.0, (5.0 - tun_z) / 2.0))
        route = [
            [0.0, 0.0, 1.5],
            [x_gate, 0.0, 1.5],
            [x_gate, y_pole - 0.6, 1.5],
            [x_gate, y_pole + 0.6, 1.5],
            [x_gate, route_y, 1.5],
            [tun_x0 - 1.7, route_y, 1.5],
            [tun_x0 - 0.7, route_y, tun_z / 2.0],
            [tun_x1 + 0.6, route_y, tun_z / 2.0],
            [tun_x1 + 1.3, route_y, 1.5],
            [gap_x + 0.7, route_y, 1.5],
            [gap_x + 0.7, gy, 1.5],
            [gx, gy, gz],
        ]
        tags += ["narrow", "gate", "tunnel"]
        params.update({"lane_width_m": lane_w, "partition_x_m": wall_x, "turn_gap_x_m": gap_x,
                       "pole_gate_gap_m": pole_gap, "tunnel_x_m": [tun_x0, tun_x1],
                       "tunnel_height_m": tun_z})
    else:
        wall_x = gx * (0.20 + 0.05 * rng.uniform())
        gap_x = gx * (0.34 + 0.04 * rng.uniform())
        wall_x2 = gx * (0.62 + 0.04 * rng.uniform())
        gap_x2 = gx * (0.86 + 0.04 * rng.uniform())
        lane_edge = 5.0 - lane_w
        route_y = (lane_edge + 5.18) / 2.0
        lane2_edge = -5.0 + lane_w
        route2_y = (lane2_edge - 5.18) / 2.0
        add_box(obstacles, (wall_x, (lane_edge - 5.0) / 2.0, 2.5),
                (0.15, (lane_edge + 5.0) / 2.0, 2.5))
        add_box(obstacles, ((wall_x + gap_x) / 2.0, lane_edge + 0.03, 2.5),
                ((gap_x - wall_x) / 2.0 + 0.15, 0.15, 2.5))
        add_box(obstacles, (wall_x2, (lane2_edge + 5.0) / 2.0, 2.5),
                (0.15, (5.0 - lane2_edge) / 2.0, 2.5))
        add_box(obstacles, ((wall_x2 + gap_x2) / 2.0, lane2_edge - 0.03, 2.5),
                ((gap_x2 - wall_x2) / 2.0 + 0.15, 0.15, 2.5))
        mid_x = (gap_x + wall_x2) / 2.0
        # approach column must stay clear of the partition and its lane barrier
        x_col = min(0.90 + 0.30 * rng.uniform(), wall_x - 0.65)
        y_mid = (route_y + route2_y) / 2.0
        for sx in (-1.0, 1.0):
            obstacles.append(cylinder((mid_x + sx * (pole_gap / 2.0 + pole_r), y_mid, pole_h),
                                      pole_r, pole_h))
        route = [
            [0.0, 0.0, 1.5],
            [x_col, 0.0, 1.5],
            [x_col, route_y, 1.5],
            [wall_x - 0.4, route_y, 1.5],
            [wall_x + 0.4, route_y, 1.5],
            [gap_x + 0.7, route_y, 1.5],
            [mid_x, route_y, 1.5],
            [mid_x, y_mid + 0.7, 1.5],
            [mid_x, y_mid - 0.7, 1.5],
            [mid_x, route2_y, 1.5],
            [wall_x2 - 0.5, route2_y, 1.5],
            [wall_x2 + 0.5, route2_y, 1.5],
            [gap_x2 + 0.6, route2_y, 1.5],
            [gx, gy, gz],
        ]
        tags += ["narrow", "gate", "two-corridor"]
        params.update({"lane_width_m": lane_w, "partition_x_m": wall_x, "turn_gap_x_m": gap_x,
                       "partition_2_x_m": wall_x2, "turn_gap_2_x_m": gap_x2,
                       "lane_2_width_m": lane_w, "pole_gate_gap_m": pole_gap})

    return {"goal": [gx, gy, gz], "obstacles": obstacles, "route": route,
            "parameters": params, "tags": tags}


def build_family15(rng: WorldRng, difficulty: float, archetype: int) -> dict[str, Any]:
    g = goal_from_rng(rng)
    gx, gy, gz = g["gx"], g["gy"], g["gz"]
    obstacles: list[dict[str, Any]] = []
    tags = ["long", "narrow", "doors"]

    wall_1 = gx * (0.13 + 0.03 * rng.uniform())
    wall_2 = gx * (0.80 + 0.04 * rng.uniform())
    offset = 1.70 + 1.00 * difficulty + 0.30 * rng.uniform()
    gap_1, gap_2 = -offset, offset
    gap_half = 0.275 + 0.055 * (1.0 - difficulty)  # width 0.55 .. 0.66 -> clearance 0.095 .. 0.15
    gap_z, gap_half_z = 1.50, 0.80

    table_x = gx * 0.42
    table_y = (rng.uniform() - 0.5) * 0.20
    table_half_x = min(0.45, gx * 0.06)
    table_half_y = 0.60 + 0.25 * difficulty
    table_top = 1.45 + 0.30 * difficulty
    side = 1.0 if rng.uniform() >= 0.5 else -1.0
    side_y = table_y + side * (table_half_y + 0.45)   # approach side, clearance 0.27
    pass_y = table_y - side * (table_half_y + 0.45)   # far side after the U detour
    # west-end U detour: route swings past the table's west end, so every
    # corner sits in open space (no reversals); the west arm is clamped east of
    # doorway 1 because its side boxes run from the gap to the room edge
    jog_x = table_x - table_half_x - 0.90
    arm_x = jog_x - max(0.20, min(0.65, jog_x - wall_1 - 0.80))

    add_doorway(obstacles, wall_1, gap_1, gap_z, gap_half, gap_half_z)
    add_doorway(obstacles, wall_2, gap_2, gap_z, gap_half, gap_half_z)
    add_table(obstacles, table_x, table_y, table_half_x, table_half_y, table_top)

    route = [
        [0.0, 0.0, 1.5],
        [wall_1 - 1.25, gap_1, 1.5],
        [wall_1 - 0.45, gap_1, 1.5],
        [wall_1 + 0.45, gap_1, 1.5],
        [wall_1 + 1.25, gap_1, 1.5],
        [jog_x, side_y, 1.5],
        [arm_x, side_y, 1.5],
        [arm_x, pass_y, 1.5],
        [jog_x, pass_y, 1.5],
    ]

    if archetype == 0:
        add_box(obstacles, (gx * 0.62, 3.85, 0.55), (0.22, 0.40, 0.55))
        route.append([table_x + table_half_x + 0.6, pass_y, 1.5])
        params_arch = "doors_table_side"
        tags += ["detour"]
    else:
        # ceiling box over the west part of the table-side passage: route dives
        # to zb/2 beneath it, 0.30 m clear of the table on the passage side
        bxx0 = table_x - table_half_x - 0.15
        bxx1 = bxx0 + 1.0
        zb = 0.90 + 0.12 * difficulty            # route flies at zb/2 -> 0.27 .. 0.33 clear
        low_z = zb / 2.0
        add_box(obstacles, ((bxx0 + bxx1) / 2.0, pass_y - side * 0.20, (zb + 5.0) / 2.0),
                (0.5, 0.5, (5.0 - zb) / 2.0))
        route += [
            [table_x - table_half_x - 0.6, pass_y, low_z],
            [bxx1 + 0.45, pass_y, low_z],
            [bxx1 + 0.85, pass_y, 1.5],
            [table_x + table_half_x + 1.1, pass_y, 1.5],
        ]
        params_arch = "doors_table_low"
        tags += ["tunnel", "vertical"]

    route += [
        [wall_2 - 1.25, gap_2, 1.5],
        [wall_2 - 0.45, gap_2, 1.5],
        [wall_2 + 0.45, gap_2, 1.5],
        [wall_2 + 1.25, gap_2, 1.5],
        [gx, gy, gz],
    ]

    return {"goal": [gx, gy, gz], "obstacles": obstacles, "route": route,
            "parameters": {"archetype": params_arch, "door_1_x_m": wall_1, "door_2_x_m": wall_2,
                           "door_gap_centers_y_m": [gap_1, gap_2],
                           "door_clear_width_m": mul(gap_half, 2.0),
                           "table_center_m": [table_x, table_y, table_top],
                           "table_half_extent_xy_m": [table_half_x, table_half_y],
                           "table_side": "south" if side > 0 else "north"},
            "tags": tags}


def build_family16(rng: WorldRng, difficulty: float, archetype: int) -> dict[str, Any]:
    g = goal_from_rng(rng)
    gx, gy, gz = g["gx"], g["gy"], g["gz"]
    obstacles: list[dict[str, Any]] = []
    tags = ["long", "vertical"]

    low_x = gx * (0.14 + 0.03 * rng.uniform())
    overhead_x = gx * (0.46 + 0.04 * rng.uniform())
    choice_x = gx * (0.76 + 0.04 * rng.uniform())
    low_top = 1.60 + 0.55 * difficulty
    c_over = 0.24 + 0.06 * rng.uniform()
    over_z = low_top + BODY_RADIUS_M + c_over
    under_z = 1.45 - 0.30 * difficulty
    c_under = 0.28 + 0.06 * rng.uniform()
    overhead_bottom = under_z + BODY_RADIUS_M + c_under
    choice_half = 0.10 + 0.06 * difficulty
    c_choice = 0.24 + 0.06 * rng.uniform()
    choose_over = difficulty >= 0.5
    if choose_over:
        choice_z = 2.0 + choice_half + BODY_RADIUS_M + c_choice
    else:
        choice_z = 2.0 - choice_half - BODY_RADIUS_M - c_choice

    add_box(obstacles, (low_x, 0.0, low_top / 2.0), (0.12, 5.0, low_top / 2.0))
    overhead_half = (5.0 - overhead_bottom) / 2.0
    add_box(obstacles, (overhead_x, 0.0, overhead_bottom + overhead_half), (0.12, 5.0, overhead_half))
    add_box(obstacles, (choice_x, 0.0, 2.0), (0.10, 5.0, choice_half))

    pole_gap = 0.64 - 0.08 * difficulty
    pole_r, pole_h = 0.14, 1.10
    # gate sits between the low barrier and the overhang; poles are offset in y
    # so the east-west witness crosses the gap at y = 0
    x_col = low_x + 1.00
    y_off = pole_gap / 2.0 + pole_r
    for sy in (-1.0, 1.0):
        obstacles.append(cylinder((x_col, sy * y_off, pole_h), pole_r, pole_h))

    if archetype == 1:
        c_squeeze = 0.12 + 0.18 * rng.uniform()
        ceil_bottom = over_z + BODY_RADIUS_M + c_squeeze
        add_box(obstacles, ((low_x - 0.6 + low_x + 0.6) / 2.0, 0.0, (ceil_bottom + 5.0) / 2.0),
                (0.6, 5.0, (5.0 - ceil_bottom) / 2.0))
        tags += ["squeeze"]

    # west semicircle loop for path length: starts at the goal origin, sweeps
    # clockwise 323 deg around (-0.6, -0.8) R=1.0 (west extreme -1.6, clear of
    # the -2 wall), exits east at (-0.6, 0.2); every interior turn stays <= 30 deg
    loop_cx, loop_cy, loop_r = -0.6, -0.8, 1.0
    theta0 = math.degrees(math.atan2(0.0 - loop_cy, 0.0 - loop_cx))
    loop_steps = 11
    arc = [[0.0, 0.0, 1.5]]
    for k in range(1, loop_steps + 1):
        th = math.radians(theta0 - (theta0 + 270.0) * k / loop_steps)
        arc.append([loop_cx + loop_r * math.cos(th), loop_cy + loop_r * math.sin(th), 1.5])

    route = arc + [
        [low_x - 1.1, 0.0, 1.5],
        [low_x - 0.9, 0.0, 1.5],
        [low_x - 0.9, 0.0, over_z],
        [low_x + 0.9, 0.0, over_z],
        [x_col, 0.0, over_z],
        [overhead_x - 1.0, 0.0, over_z],
        [overhead_x - 1.0, 0.0, under_z],
        [overhead_x + 1.0, 0.0, under_z],
        [choice_x - 1.0, 0.0, under_z],
        [choice_x - 1.0, 0.0, choice_z],
        [choice_x + 1.0, 0.0, choice_z],
        [choice_x + 1.6, -1.9, 1.5],
        [gx - 0.8, -2.7, 1.5],
        [gx, gy, gz],
    ]

    return {"goal": [gx, gy, gz], "obstacles": obstacles, "route": route,
            "parameters": {"archetype": "vertical_gate" if archetype == 0 else "vertical_squeeze",
                           "low_barrier_x_m": low_x, "low_barrier_top_m": low_top,
                           "overhang_x_m": overhead_x, "overhang_bottom_m": overhead_bottom,
                           "choice_slab_x_m": choice_x, "choice_slab_half_height_m": choice_half,
                           "witness_choice": "over" if choose_over else "under",
                           "pole_gate_gap_m": pole_gap},
            "tags": tags}


BUILDERS = {14: build_family14, 15: build_family15, 16: build_family16}


# ---------------------------------------------------------- mover placement --


def _mover_in_room(center: list[float], size: float | tuple[float, float, float],
                   until: float, velocity: list[float]) -> float:
    """Earliest time (<=until) at which the body leaves the room, or `until`."""
    for t in [k * 0.25 for k in range(int(until * 4) + 1)]:
        for axis, (lo, hi) in zip(range(3), ((-2.0, 14.0), (-5.0, 5.0), (0.0, 5.0))):
            extent = size[axis] if isinstance(size, tuple) else size
            c = center[axis] + t * velocity[axis]
            if c - extent < lo - 1e-4 or c + extent > hi + 1e-4:
                return t
    return until


def place_movers(world: dict[str, Any], route: list[list[float]], rng: WorldRng,
                 count: int, difficulty: float, timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Phase movers relative to the declared nominal traversal of `route`.

    For a crossing at arc s with nominal passage t_m = s/v_nom and safety
    (seconds of nominal clearance) = error_bound(s):
        t_cross = t_m - safety - t_b ,  t_b = (r + 0.18)/v
        static clearance at t = 0 = v*(t_m - safety) - 2*(r + 0.18)
    so any passage later than nominal keeps >= v*safety clearance, while a
    passage inside [t_cross - t_b, t_cross + t_b] collides.  Speed is nudged up
    from the analytic value (8 steps x 1.25, capped at min(0.45, 1.6/safety))
    until the sampled static clearance (route folds make the analytic value
    optimistic) clears MIN_STATIC_MOVER_CLEARANCE_M.  Candidate arcs/axes/signs
    are tried until one fits inside the room for the whole HOLD_S budget.
    """
    placed: list[dict[str, Any]] = []
    v_nom = timing["nominal_speed_mps"]
    total_len = route_polyline_length(route)
    early = (0.08, 0.14, 0.20, 0.27, 0.34, 0.42, 0.50, 0.58)
    late = (0.64, 0.71, 0.78, 0.85, 0.91)
    groups = [early, late] if count == 2 else [early + late]

    samples = list(route_samples(route, 0.10))
    for group in groups:
        if len(placed) >= count:
            break
        target = count if group is groups[-1] else len(placed) + 1
        for frac in group:
            if len(placed) >= target:
                break
            placed_one = None
            for _attempt in range(4):
                placed_one = _place_one_mover(world, route, rng, frac, timing,
                                              placed, total_len, samples)
                if placed_one:
                    break
            if placed_one:
                placed.append(placed_one)
    return placed


def _place_one_mover(world: dict[str, Any], route: list[list[float]], rng: WorldRng,
                     frac: float, timing: dict[str, Any],
                     placed: list[dict[str, Any]], total_len: float,
                     samples: list[tuple[float, list[float]]]) -> dict[str, Any] | None:
    v_nom = timing["nominal_speed_mps"]
    arc = total_len * frac
    radius = 0.24 + 0.10 * rng.uniform()
    kind = 1 if rng.uniform() >= 0.5 else 2
    cyl_half = radius * (1.0 + 0.6 * rng.uniform()) if kind == 2 else radius
    extents = (radius, radius, cyl_half)
    safety = error_bound_s(arc, timing)
    t_model = arc / v_nom
    if t_model <= safety + 0.5:
        return None
    speed_cap = min(0.45, 1.6 / max(safety, 1e-6))
    base_speed = (2.0 * (radius + BODY_RADIUS_M) + 0.14) / (t_model - safety)
    if base_speed > speed_cap:
        return None

    p = point_at_arc(route, arc)
    direction = route_direction(route, arc)
    horiz = (abs(direction[0]), abs(direction[1]), abs(direction[2]))
    if horiz[2] >= max(horiz[0], horiz[1]):
        axes = (0, 1)
    elif horiz[0] <= horiz[1]:
        axes = (0, 1)
    else:
        axes = (1, 0)

    # per-sample clearance to the world without the candidate mover: computed
    # once per call so the per-candidate test below is O(1) per sample.
    base_cs = [point_clearance(world, q, 0.0) for _, q in samples]
    chosen = None
    speed = base_speed
    for _ in range(8):
        t_b = (radius + BODY_RADIUS_M) / speed
        t_cross = t_model - safety - t_b
        if t_cross <= 0.05:
            break
        for axis in axes:
            for sign in (1.0, -1.0):
                center = list(p)
                velocity = [0.0, 0.0, 0.0]
                center[axis] = f32(p[axis] - sign * speed * t_cross)
                velocity[axis] = f32(sign * speed)
                if _mover_in_room(center, extents, HOLD_S, velocity) < HOLD_S:
                    continue
                # conservative probe: cylinder is never better than the
                # eventual sphere (kind 1) or cylinder (kind 2) body
                mover = {"kind": 2, "center": [f32(v) for v in center],
                         "half_extent": [f32(radius), f32(radius),
                                         f32(radius * 1.6)], "velocity": velocity}
                probe = {"obstacles": world["obstacles"] + [mover],
                         "goal": world["goal"]}
                single = {"obstacles": [mover], "goal": world["goal"]}
                start_clearance = point_clearance(probe, [0.0, 0.0, 1.5], 0.0)
                if start_clearance < 0.10:
                    continue
                # the witness dwells around the goal after its earliest
                # arrival, so probe a time window, not a single instant
                goal_time = total_len / v_nom
                goal_window = error_bound_s(total_len, timing)
                if any(point_clearance(probe, world["goal"],
                                       goal_time + k * goal_window / 5.0) < 0.10
                       for k in range(6)):
                    continue
                static_clearance = min(
                    min(base_cs[i], point_clearance(single, q, 0.0))
                    for i, (_, q) in enumerate(samples))
                if static_clearance < MIN_STATIC_MOVER_CLEARANCE_M:
                    continue
                # the nominal-time route/mover clearance must stay safe too
                # (the sweep can re-cross a folded corner leg later on)
                if route_nominal_timing(probe, route, v_nom)["clearance"] < 0.10:
                    continue
                if any(math.dist(center, other["center"]) <
                       radius + other["radius_m"] + 0.30 for other in placed):
                    continue
                chosen = (center, velocity, static_clearance, start_clearance, speed, t_cross, axis)
                break
            if chosen:
                break
        if chosen:
            break
        speed = min(speed * 1.25, speed_cap)
    if chosen is None:
        return None

    center, velocity, static_clearance, start_clearance, speed, t_cross, axis = chosen
    if kind == 2:
        body = cylinder(tuple(center), radius, cyl_half)
    else:
        body = sphere(tuple(center), radius)
    body["velocity"] = velocity
    world["obstacles"].append(body)
    spec = {
        "obstacle_index": len(world["obstacles"]) - 1,
        "kind": kind,
        "axis": "xyz"[axis],
        "center": [f32(v) for v in center],
        "radius_m": f32(radius),
        "speed_mps": f32(speed),
        "crossing_arc_m": f32(arc),
        "nominal_passage_s": f32(t_model),
        "crossing_time_s": f32(t_cross),
        "blocking_window_s": [f32(t_cross - (radius + BODY_RADIUS_M) / speed),
                              f32(t_cross + (radius + BODY_RADIUS_M) / speed)],
        "nominal_clearance_m": f32(speed * safety),
        "timing_safety_s": f32(safety),
        "error_bound_s": f32(error_bound_s(arc, timing)),
        "static_route_clearance_m": f32(static_clearance),
        "start_clearance_m": f32(start_clearance),
        "room_exit_time_s": f32(_mover_in_room(center, extents, 60.0, velocity)),
    }
    return spec


# ------------------------------------------------------------- validation ----


class ValidationError(Exception):
    pass


def validate(world: dict[str, Any], movers: list[dict[str, Any]], timing: dict[str, Any],
             path_target: float) -> dict[str, Any]:
    if len(world["obstacles"]) > MAX_OBSTACLES:
        raise ValidationError(f"{len(world['obstacles'])} obstacles exceeds 16")

    start = [0.0, 0.0, 1.5]
    goal = world["goal"]
    route = world["witness_route"]
    if any(abs(start[i] - route[0][i]) > 1e-6 for i in range(3)):
        raise ValidationError("witness route has wrong start")
    if any(abs(goal[i] - route[-1][i]) > 1e-6 for i in range(3)):
        raise ValidationError("witness route has wrong goal")

    for obstacle in world["obstacles"]:
        if obstacle["kind"] not in (0, 1, 2):
            raise ValidationError(f"bad kind {obstacle['kind']}")
        size = obstacle["half_extent"]
        kind = obstacle["kind"]
        if (kind == 0 and min(size) <= 0) or (kind == 1 and size[0] <= 0) or \
           (kind == 2 and (size[0] <= 0 or size[2] <= 0)):
            raise ValidationError("non-positive collision dimension")
        for axis, (lo, hi) in zip(range(3), ((-2.0, 14.0), (-5.0, 5.0), (0.0, 5.0))):
            c = obstacle["center"][axis]
            extent = size[0] if kind == 1 else (size[2] if (kind == 2 and axis == 2) else size[axis])
            if c - extent < lo - 1e-4 or c + extent > hi + 1e-4:
                raise ValidationError(f"obstacle outside room on axis {axis} at t=0")
        if obstacle["velocity"] != [0.0, 0.0, 0.0]:
            if _mover_in_room(obstacle["center"], max(obstacle["half_extent"][:2]),
                              HOLD_S, obstacle["velocity"]) < HOLD_S:
                raise ValidationError("mover leaves the room inside the eval budget")

    static = route_metrics(world, route)
    if static["clearance"] <= MARGIN_M:
        raise ValidationError(f"unsafe witness clearance {static['clearance']}")
    if static["clearance"] < MIN_WITNESS_CLEARANCE_M - 1e-6:
        raise ValidationError(f"witness clearance {static['clearance']} below generator target")
    if point_clearance(world, start, 0.0) <= MARGIN_M:
        raise ValidationError("unsafe start")
    if point_clearance(world, goal, 0.0) <= MARGIN_M:
        raise ValidationError("unsafe goal")
    for i in range(1, len(route) - 1):
        v0 = [route[i - 1][k] - route[i][k] for k in range(3)]
        v2 = [route[i + 1][k] - route[i][k] for k in range(3)]
        n0 = math.sqrt(sum(c * c for c in v0))
        n2 = math.sqrt(sum(c * c for c in v2))
        if n0 < 1e-9 or n2 < 1e-9:
            continue
        dot = max(-1.0, min(1.0, sum(v0[k] * v2[k] for k in range(3)) / (n0 * n2)))
        turn = 180.0 - math.degrees(math.acos(dot))
        if turn > MAX_SHARP_TURN_DEG:
            clearance = point_clearance(world, route[i], 0.0)
            if clearance < MIN_TURN_CLEARANCE_M:
                raise ValidationError(
                    f"sharp {turn:.0f}deg turn at waypoint {i} is only "
                    f"{clearance:.3f} m clear (needs {MIN_TURN_CLEARANCE_M} m open)")
    direct = route_metrics(world, [start, goal])
    if direct["clearance"] >= -0.02:
        raise ValidationError(f"direct route not obstructed: {direct['clearance']}")

    timing_metrics = route_nominal_timing(world, route, timing["nominal_speed_mps"])
    if timing_metrics["clearance"] <= MARGIN_M:
        raise ValidationError(f"nominal timing clearance {timing_metrics['clearance']} unsafe")

    for mover in movers:
        if mover["timing_safety_s"] + 1e-6 < mover["error_bound_s"]:
            raise ValidationError("mover timing safety below error bound")
        if mover["crossing_time_s"] <= mover["blocking_window_s"][1]:
            pass  # crossing time is the window centre by construction
        if point_clearance(world, start, 0.0) < 0.04:
            raise ValidationError("mover threatens start")
        goal_time = static["length"] / timing["nominal_speed_mps"]
        probe_goal = point_clearance(world, goal, goal_time)
        if probe_goal <= MARGIN_M:
            raise ValidationError(f"mover threatens goal at nominal arrival ({probe_goal})")

    if static["length"] < path_target:
        raise ValidationError(f"route too short: {static['length']:.2f} m < {path_target} m")

    return {
        "static_clearance_m": static["clearance"],
        "static_samples": static["samples"],
        "path_length_m": static["length"],
        "direct_clearance_m": direct["clearance"],
        "timing_clearance_m": timing_metrics["clearance"],
        "timing_worst_arc_m": timing_metrics["arc_m"],
    }


# ------------------------------------------------------------ record build ---


def mirror_y(world: dict[str, Any]) -> None:
    world["goal"][1] = -world["goal"][1]
    world["wind"][1] = -world["wind"][1]
    for obstacle in world["obstacles"]:
        obstacle["center"][1] = -obstacle["center"][1]
        obstacle["velocity"][1] = -obstacle["velocity"][1]
    for point in world["witness_route"]:
        point[1] = -point[1]


def build_records(master_seed: int, per_split: int, distance: float, families: list[int],
                  world_hash: str, splits: tuple[str, ...], movers: bool,
                  timing: dict[str, Any], path_target: float,
                  collect_failures: bool = False,
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    used_ids: set[str] = set()

    for split in splits:
        for family in families:
            base_seed = stable_base_seed(master_seed, split, family, per_split, distance)
            for index in range(per_split):
                seed = sim_world_seed(base_seed, index)
                rng = WorldRng(seed)
                difficulty = difficulty_of(base_seed, index, per_split)
                band = band_of(difficulty)
                archetype = (index // 6) % 2
                mirrored = index % 2 == 1

                built = BUILDERS[family](rng, difficulty, archetype)
                route = fillet_route(built["route"])
                world: dict[str, Any] = {
                    "family": family,
                    "family_name": FAMILY_NAMES[family],
                    "geometry_contract": FAMILY_CONTRACTS[family],
                    "seed": seed,
                    "difficulty": f32(difficulty),
                    "difficulty_band": band,
                    "goal": [f32(v) for v in built["goal"]],
                    "wind": [0.0, 0.0, 0.0],
                    "obstacles": built["obstacles"],
                    "parameters": built["parameters"],
                    "witness_route": [[f32(v) for v in p] for p in route],
                }

                mover_spec: list[dict[str, Any]] = []
                static_control = index % 3 == 0
                if movers and not static_control:
                    count = 2 if difficulty >= 2.0 / 3.0 else 1
                    mover_spec = place_movers(world, world["witness_route"], rng, count,
                                              difficulty, timing)

                if mirrored:
                    mirror_y(world)
                    world["coordinate_transform"] = "mirror_y"
                    if family == 14:
                        world["geometry_contract"] = FAMILY_CONTRACT_MIRROR[14]
                else:
                    world["coordinate_transform"] = "identity"

                try:
                    metrics = validate(world, mover_spec, timing, path_target)
                except ValidationError as error:
                    detail = (f"{FAMILY_NAMES[family]} {split}[{index}] arch={archetype} "
                              f"d={difficulty:.3f}: {error}")
                    if not collect_failures:
                        raise ValidationError(detail) from error
                    failures.append(detail)
                    continue

                tags = list(built["tags"])
                if mirrored:
                    tags.append("mirror-y")
                if mover_spec:
                    tags.append("moving")
                else:
                    tags.append("static-control")
                tags.append("long" if metrics["path_length_m"] >= 16.0 else "medium-route")

                world["record_type"] = "challenge"
                world["schema_version"] = 1
                world["split"] = split
                world["environment_index"] = index
                world["sim_config"] = {"family": family, "seed": base_seed,
                                       "distance": f32(distance),
                                       "environment_index": index}
                world["room_bounds"] = ROOM
                world["body_radius_m"] = BODY_RADIUS_M
                world["witness_min_clearance_m"] = metrics["static_clearance_m"]
                world["witness_sample_count"] = metrics["static_samples"]
                world["direct_route_clearance_m"] = metrics["direct_clearance_m"]
                world["witness_path_length_m"] = f32(metrics["path_length_m"])
                world["direct_distance_m"] = f32(math.sqrt(
                    sum((world["goal"][i] - (0.0 if i < 2 else 1.5)) ** 2 for i in range(3))))
                world["detour_ratio"] = f32(metrics["path_length_m"] /
                                            max(world["direct_distance_m"], 1e-6))
                world["witness_nominal_timing_clearance_m"] = metrics["timing_clearance_m"]
                world["timing_model"] = dict(timing)
                world["movers"] = mover_spec
                world["mover_count"] = len(mover_spec)
                world["archetype"] = built["parameters"]["archetype"]
                world["course_tags"] = tags
                world["selection_policy"] = (
                    "final split is held out from training, checkpoint selection, and curriculum design"
                    if split == "final" else
                    "development split may be used for training or checkpoint selection"
                    if split == "dev" else "training split")

                scene = scene_hash(world)
                world["failure_id"] = (
                    f"mc{family}-{split}-{index:04d}-s{seed:08x}-w{scene[:8]}"
                    + ("-mirror-y" if mirrored else ""))
                if world["failure_id"] in used_ids:
                    raise ValidationError(f"duplicate failure id {world['failure_id']}")
                used_ids.add(world["failure_id"])
                world["scene_sha256"] = scene
                world["world_source_sha256"] = world_hash

                # loader-equivalent seed check
                derived = xorshift32((base_seed + index * SIM_ENV_STRIDE + SIM_ENV_OFFSET) & MASK32)
                if derived != seed:
                    raise ValidationError("seed derivation mismatch")

                records.append(world)
                diagnostics.append({
                    "failure_id": world["failure_id"], "split": split, "family": family,
                    "archetype": world["archetype"], "band": band,
                    "difficulty": world["difficulty"], "mirrored": mirrored,
                    "movers": len(mover_spec), "obstacles": len(world["obstacles"]),
                    "path_length_m": world["witness_path_length_m"],
                    "detour_ratio": world["detour_ratio"],
                    "static_clearance_m": world["witness_min_clearance_m"],
                    "timing_clearance_m": world["witness_nominal_timing_clearance_m"],
                    "direct_clearance_m": world["direct_route_clearance_m"],
                    "tags": ",".join(tags),
                })
    return records, diagnostics, failures


# ----------------------------------------------------------------- outputs ---


def write_preview_svg(records: list[dict[str, Any]], path: pathlib.Path, split: str) -> None:
    """One top-down plan per family: obstacles, witness route, mover sweep."""
    scale, pad, cols = 10.0, 16.0, 6
    w_room, h_room = 16.0 * scale, 10.0 * scale
    panels = [r for r in records if r["split"] == split]
    families = sorted({r["family"] for r in panels})
    rows_total = sum(math.ceil(len([p for p in panels if p["family"] == f]) / cols)
                     for f in families)
    width = pad * 2 + cols * (w_room + pad)
    height = pad * 2 + rows_total * (h_room + pad + 14)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
           f'font-family="monospace" font-size="9">',
           f'<rect width="{width:.0f}" height="{height:.0f}" fill="#111"/>']
    row = 0
    for family in families:
        group = [p for p in panels if p["family"] == family]
        for n, rec in enumerate(group):
            col, r = n % cols, n // cols
            ox = pad + col * (w_room + pad)
            oy = pad + (row + r) * (h_room + pad + 14)
            out.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{w_room:.1f}" height="{h_room:.1f}" '
                       f'fill="#1a1a1a" stroke="#444"/>')

            def tx(x: float) -> float:
                return ox + (x + 2.0) * scale

            def ty(y: float) -> float:
                return oy + (y + 5.0) * scale

            for obstacle in rec["obstacles"]:
                c, s = obstacle["center"], obstacle["half_extent"]
                if obstacle["kind"] == 0:
                    out.append(f'<rect x="{tx(c[0]-s[0]):.1f}" y="{ty(c[1]-s[1]):.1f}" '
                               f'width="{2*s[0]*scale:.1f}" height="{2*s[1]*scale:.1f}" '
                               f'fill="#2f6f3e"/>')
                else:
                    colour = "#c98b2e" if obstacle["velocity"] != [0.0, 0.0, 0.0] else "#6a6a6a"
                    out.append(f'<ellipse cx="{tx(c[0]):.1f}" cy="{ty(c[1]):.1f}" '
                               f'rx="{s[0]*scale:.1f}" ry="{s[0]*scale:.1f}" fill="{colour}"/>')
                if obstacle["velocity"] != [0.0, 0.0, 0.0]:
                    v = obstacle["velocity"]
                    out.append(f'<line x1="{tx(c[0]):.1f}" y1="{ty(c[1]):.1f}" '
                               f'x2="{tx(c[0]+v[0]*EVAL_BUDGET_S):.1f}" '
                               f'y2="{ty(c[1]+v[1]*EVAL_BUDGET_S):.1f}" '
                               f'stroke="#c98b2e" stroke-dasharray="3,3"/>')
            pts = " ".join(f"{tx(p[0]):.1f},{ty(p[1]):.1f}" for p in rec["witness_route"])
            out.append(f'<polyline points="{pts}" fill="none" stroke="#4ea1ff" stroke-width="1.4"/>')
            out.append(f'<circle cx="{tx(0.0):.1f}" cy="{ty(0.0):.1f}" r="3" fill="#fff"/>')
            out.append(f'<circle cx="{tx(rec["goal"][0]):.1f}" cy="{ty(rec["goal"][1]):.1f}" '
                       f'r="3" fill="#ff5a5a"/>')
            out.append(f'<text x="{ox:.1f}" y="{oy + h_room + 11:.1f}" fill="#aaa">'
                       f'{rec["failure_id"]} L={rec["witness_path_length_m"]:.1f}m '
                       f'c={rec["witness_min_clearance_m"]:.2f} t={rec["mover_count"]}mvr</text>')
        row += math.ceil(len(group) / cols)
    out.append("</svg>")
    path.write_text("\n".join(out))


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {"min": round(min(values), 4), "median": round(ordered[len(ordered) // 2], 4),
                "max": round(max(values), 4), "mean": round(sum(values) / len(values), 4)}

    per_group: dict[str, Any] = {}
    for split in SPLITS:
        for family in (14, 15, 16):
            group = [r for r in records if r["split"] == split and r["family"] == family]
            if not group:
                continue
            key = f"{split}/f{family}"
            per_group[key] = {
                "levels": len(group),
                "bands": dict(Counter(r["difficulty_band"] for r in group)),
                "archetypes": dict(Counter(r["archetype"] for r in group)),
                "mirrored": sum(1 for r in group if r["coordinate_transform"] == "mirror_y"),
                "with_movers": sum(1 for r in group if r["mover_count"] > 0),
                "obstacles_max": max(len(r["obstacles"]) for r in group),
                "path_length_m": stats([r["witness_path_length_m"] for r in group]),
                "detour_ratio": stats([r["detour_ratio"] for r in group]),
                "static_clearance_m": stats([r["witness_min_clearance_m"] for r in group]),
                "timing_clearance_m": stats([r["witness_nominal_timing_clearance_m"] for r in group]),
            }
    kinds = Counter(o["kind"] for r in records for o in r["obstacles"])
    speeds = [m["speed_mps"] for r in records for m in r["movers"]]
    return {
        "levels": len(records),
        "splits": dict(Counter(r["split"] for r in records)),
        "families": dict(Counter(r["family"] for r in records)),
        "obstacle_kinds": {str(k): v for k, v in sorted(kinds.items())},
        "movers": len(speeds),
        "mover_speed_mps": stats(speeds),
        "per_group": per_group,
    }


# ------------------------------------------------------------------ CLI -----


def cmd_build(args: argparse.Namespace) -> int:
    repo = pathlib.Path(__file__).resolve().parent
    world_hash = source_hash(repo / "world.hpp")
    timing = load_timing_model(args.timing_model)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for split in splits:
        if split not in SPLITS:
            raise SystemExit(f"--splits must be subset of {SPLITS}")
    families = [int(v) for v in args.families.split(",") if v]
    if args.per_split % 36 != 0:
        raise SystemExit("--per-split must be a multiple of 36 for band/archetype balance")

    records, diagnostics, failures = build_records(args.seed, args.per_split, args.distance,
                                                   families, world_hash, splits,
                                                   not args.no_movers, timing,
                                                   args.path_target,
                                                   collect_failures=args.debug)

    if failures:
        grouped: dict[str, int] = {}
        for line in failures:
            reason = line.split(": ", 1)[-1]
            key = " ".join(reason.split()[:3])
            grouped[key] = grouped.get(key, 0) + 1
        print(f"{len(failures)} validation failures (grouped):")
        for key, count in sorted(grouped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {key}")
        for line in failures[: args.debug_lines]:
            print(f"  - {line}")
        if len(failures) > args.debug_lines:
            print(f"  ... {len(failures) - args.debug_lines} more")
        return 1
    if not records:
        raise SystemExit("no records generated")

    out: pathlib.Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    bank_sha = hashlib.sha256(out.read_bytes()).hexdigest()

    checks_path = out.with_name(args.checks_name)
    summary = summarise(records)
    checks = {
        "bank": str(out), "bank_sha256": bank_sha,
        "world_source_sha256": world_hash,
        "generator": str(pathlib.Path(__file__).name),
        "generator_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "master_seed": args.seed, "per_split": args.per_split, "distance_m": args.distance,
        "splits": list(splits), "families": families,
        "no_movers": args.no_movers, "path_target_m": args.path_target,
        "timing_model": timing,
        "contract": {
            "record_type": "challenge", "schema_version": 1,
            "max_obstacles": MAX_OBSTACLES,
            "min_witness_clearance_m": MIN_WITNESS_CLEARANCE_M,
            "contract_floor_clearance_m": MARGIN_M,
            "hold_s": HOLD_S, "eval_budget_s": EVAL_BUDGET_S,
        },
        "summary": summary,
        "diagnostics": diagnostics,
    }
    checks_path.write_text(json.dumps(checks, indent=2, sort_keys=True))

    manifest = {
        "name": "mimo-courses", "version": 1, "bank_sha256": bank_sha,
        "world_source_sha256": world_hash, "generator_sha256": checks["generator_sha256"],
        "master_seed": args.seed, "distance_m": args.distance,
        "splits": list(splits), "families": families,
        "selection_policy": {
            "train": "training split", "dev": "development / checkpoint selection",
            "final": "held out: never train, select, or evaluate on final",
        },
        "records": [
            {"failure_id": r["failure_id"], "split": r["split"], "family": r["family"],
             "environment_index": r["environment_index"], "seed": r["seed"],
             "base_seed": r["sim_config"]["seed"], "scene_sha256": r["scene_sha256"],
             "archetype": r["archetype"], "difficulty_band": r["difficulty_band"],
             "difficulty": r["difficulty"], "mover_count": r["mover_count"],
             "obstacle_count": len(r["obstacles"]),
             "path_length_m": r["witness_path_length_m"],
             "detour_ratio": r["detour_ratio"],
             "static_clearance_m": r["witness_min_clearance_m"],
             "nominal_timing_clearance_m": r["witness_nominal_timing_clearance_m"],
             "course_tags": r["course_tags"]}
            for r in records
        ],
        "unknowns": [
            "witness_min_clearance_m is static (t=0), as in challenge_bank.py; "
            "mover-aware clearance is witness_nominal_timing_clearance_m",
            "mover velocity is not in any policy observation (depth history only)",
            "temporal feasibility is validated separately by bank-witness, not by this file",
        ],
    }
    manifest_path = out.with_name(args.manifest_name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if args.preview_dir:
        preview_dir = pathlib.Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            write_preview_svg(records, preview_dir / f"courses-{split}.svg", split)

    checks_rows = out.with_name(args.checks_csv)
    with checks_rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"wrote {len(records)} levels -> {out}")
    print(f"bank_sha256={bank_sha}")
    print(f"world_source_sha256={world_hash}")
    for key, group in summary["per_group"].items():
        print(f"  {key}: n={group['levels']} bands={group['bands']} "
              f"path={group['path_length_m']['median']}m "
              f"(min {group['path_length_m']['min']}) "
              f"static_c={group['static_clearance_m']['min']} "
              f"timing_c={group['timing_clearance_m']['min']} "
              f"movers={group['with_movers']} obs_max={group['obstacles_max']}")
    print(f"manifest={manifest_path} checks={checks_path} contact={checks_rows}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(args.witness.open()))
    if not rows:
        raise SystemExit("witness csv is empty")
    speeds = []
    for row in rows:
        if row.get("success") not in ("1", "true"):
            continue
        path_m, elapsed = float(row["path_m"]), float(row["elapsed_s"])
        if elapsed > 0 and path_m > 0:
            speeds.append(path_m / elapsed)
    if len(speeds) < 10:
        raise SystemExit(f"only {len(speeds)} successful witness rows; cannot calibrate")
    speeds.sort()
    v_median = speeds[len(speeds) // 2]
    # Nominal is the *speed cap*, i.e. the earliest possible arrival on any
    # arc: blocking windows must then end before the witness can possibly
    # reach the crossing (corner stops only make it later, which is safe).
    # The spread bound covers both the shortcut slack and the slowdown from
    # cap speed down to the slowest measured average.
    v_cap = 1.0
    inv_spread = max(abs(1.0 / v - 1.0 / v_cap) for v in speeds)
    model = dict(DEFAULT_TIMING)
    model.update({
        "nominal_speed_mps": v_cap,
        "inv_spread_s_per_m": round(min(max(inv_spread, 0.02), 0.40), 4),
        "source": f"calibrated from {args.witness} "
                  f"({len(speeds)} successful TRAIN pilot witness rows); "
                  f"nominal = witness speed cap (earliest-arrival bound)",
        "pilot_witness": str(args.witness),
        "pilot_rows": len(speeds),
        "pilot_speed_median_mps": round(v_median, 4),
        "pilot_speed_min_mps": round(speeds[0], 4),
        "pilot_speed_max_mps": round(speeds[-1], 4),
        "measured_inv_spread_s_per_m": round(inv_spread, 4),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2, sort_keys=True))
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="generate and validate a course bank")
    build.add_argument("--out", type=pathlib.Path,
                       default=pathlib.Path("results/mimo-courses/course-bank-v1.jsonl"))
    build.add_argument("--manifest-name", default="manifest.json")
    build.add_argument("--checks-name", default="checks.json")
    build.add_argument("--checks-csv", default="contact-checks.csv")
    build.add_argument("--preview-dir", default="results/mimo-courses/preview")
    build.add_argument("--seed", type=int, default=20261002)
    build.add_argument("--per-split", type=int, default=36)
    build.add_argument("--distance", type=float, default=10.0)
    build.add_argument("--families", default="14,15,16")
    build.add_argument("--splits", default="train,dev,final")
    build.add_argument("--path-target", type=float, default=14.5)
    build.add_argument("--no-movers", action="store_true")
    build.add_argument("--timing-model", type=pathlib.Path, default=None)
    build.add_argument("--debug", action="store_true",
                       help="collect every validation failure instead of raising on the first")
    build.add_argument("--debug-lines", type=int, default=25)
    build.set_defaults(func=cmd_build)

    calibrate = subparsers.add_parser("calibrate", help="fit timing model from pilot witness")
    calibrate.add_argument("--witness", type=pathlib.Path, required=True)
    calibrate.add_argument("--out", type=pathlib.Path,
                           default=pathlib.Path("results/mimo-courses/timing-model.json"))
    calibrate.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
