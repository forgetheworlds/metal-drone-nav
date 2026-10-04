#!/usr/bin/env python3
"""Independent analysis of the native Webots RangeFinder frames.

Owner: results/omp-sensor-transfer. Reads only:
  results/omp-sensor-transfer/sensor-geometry/native-frames.jsonl   (fixture output)
and the frozen scene constants below (mirrored from webots/worlds/sensor-geometry-audit.wbt,
whose sha256 is recorded in the summary).

It compares, per pixel and per pose:
  * measured native range values against the analytic camera projection of the scene
    (documented R2025a geometry: fieldOfView = horizontal, square pixels, pixel centres
     (i+0.5)/w, rows top-to-bottom, range-finder optical axis = node +X);
  * native axial depth vs ray range (the encode_depth shader stores -view.z);
  * an exact replication of the production adapter normalize_ranges()/pool_ranges();
  * the source training contract (world.hpp::wcamera ray, ray range, ray origin = body
    centre) against both the adapter output and the native measurement.

No correction is fitted and no navigation outcome is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "omp-sensor-transfer" / "sensor-geometry"
FRAMES = OUT_DIR / "native-frames.jsonl"
WORLD = ROOT / "webots" / "worlds" / "sensor-geometry-audit.wbt"
CONTROLLER = ROOT / "webots" / "controllers" / "sensor_geometry_audit" / "sensor_geometry_audit.c"

WIDTH, HEIGHT = 20, 16
PIXELS = WIDTH * HEIGHT
SOURCE_TAN_V = 0.75                    # world.hpp::wcamera vertical coefficient
SOURCE_RAY = lambda x, y, tan_v: (     # body-frame ray, FLU, image left=+Y, up=+Z
    (2.0 * (x + 0.5) / WIDTH - 1.0) * 1.0,
    (1.0 - 2.0 * (y + 0.5) / HEIGHT) * tan_v,
)

# Frozen scene: (name, centre, half_extent) axis-aligned boxes; cylinders are world-Z.
BOXES = [
    ("GROUND", (0.0, 0.0, -0.05), (20.0, 20.0, 0.05)),
    ("CEILING", (0.0, 0.0, 5.05), (20.0, 20.0, 0.05)),
    ("FRONT_WALL", (3.05, 0.0, 1.3), (0.05, 1.5, 1.3)),
    ("BACK_WALL", (-3.05, 0.0, 2.5), (0.05, 8.0, 2.5)),
    ("LEFT_WALL", (0.0, 3.05, 2.5), (8.0, 0.05, 2.5)),
    ("RIGHT_WALL", (0.0, -3.05, 2.5), (8.0, 0.05, 2.5)),
    ("OCCLUDER", (1.2, 0.9, 1.1), (0.03, 0.3, 0.45)),
    ("MARKER_LEFT_UP", (2.7, 0.9, 2.3), (0.1, 0.15, 0.15)),
    ("MARKER_RIGHT_DOWN", (2.7, -0.9, 0.9), (0.1, 0.15, 0.15)),
]
CYLINDERS = [("POLE", (2.2, -0.5), 0.15, 0.0, 5.0)]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def mat_vec(m, v):
    """Row-major 3x3 matrix (local->world) times vector."""
    return tuple(m[3 * r] * v[0] + m[3 * r + 1] * v[1] + m[3 * r + 2] * v[2] for r in range(3))


def ray_box(o, d, c, h):
    tmin, tmax = -1e30, 1e30
    for j in range(3):
        off = o[j] - c[j]
        if abs(d[j]) < 1e-12:
            if abs(off) > h[j]:
                return None
        else:
            a = (-h[j] - off) / d[j]
            b = (h[j] - off) / d[j]
            if a > b:
                a, b = b, a
            tmin = max(tmin, a)
            tmax = min(tmax, b)
    if tmax < max(tmin, 0.0):
        return None
    return tmin if tmin >= 0.0 else tmax


def ray_cylinder_z(o, d, cx, cy, cz, r, h):
    q = (o[0] - cx, o[1] - cy, o[2] - cz)
    a = d[0] * d[0] + d[1] * d[1]
    b = q[0] * d[0] + q[1] * d[1]
    k = q[0] * q[0] + q[1] * q[1] - r * r
    best = None
    disc = b * b - a * k
    if a > 1e-12 and disc >= 0.0:
        for sign in (-1.0, 1.0):
            t = (-b + sign * math.sqrt(disc)) / a
            if t >= 0.0 and abs(q[2] + t * d[2]) <= h:
                best = t if best is None else min(best, t)
    if abs(d[2]) > 1e-12:
        for sign in (-1.0, 1.0):
            t = (sign * h - q[2]) / d[2]
            if t < 0.0:
                continue
            x, y = q[0] + t * d[0], q[1] + t * d[1]
            if x * x + y * y <= r * r:
                best = t if best is None else min(best, t)
    return best


def scene_hit(o, d):
    """Nearest scene surface along unit ray d from o; returns (t, primitive_name)."""
    best_t, best_name = None, None
    for name, c, h in BOXES:
        t = ray_box(o, d, c, h)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_name = t, name
    for name, (cx, cy), r, z0, z1 in CYLINDERS:
        t = ray_cylinder_z(o, d, cx, cy, 0.5 * (z0 + z1), r, 0.5 * (z1 - z0))
        if t is not None and (best_t is None or t < best_t):
            best_t, best_name = t, name
    return best_t, best_name


NO_HIT = {}


def hit_or_max(o, d, key=None):
    """Explicit 12 m contract: world.hpp::wray clamps every ray to 12 m, so a ray that
    hits nothing is 12 m for the source and +inf (mapped to 12 m) natively."""
    t, name = scene_hit(o, d)
    if t is None:
        if key is not None:
            NO_HIT[key] = NO_HIT.get(key, 0) + 1
        return 12.0, "NONE"
    return t, name


def native_ray(x, y, tan_v, sy, sz):
    """Native documented pinhole: horizontal tangent from tan(fov/2), square pixels."""
    u = (2.0 * (x + 0.5) / WIDTH - 1.0) * 1.0
    v = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * tan_v
    d_local = unit((1.0, sy * u, sz * v))
    return d_local, u, v


def adapter_resample(raw, hfov):
    """Exact port of raptor_webots.cpp normalize_ranges()."""
    tan_h = math.tan(0.5 * hfov)
    tan_v_native = tan_h * HEIGHT / WIDTH
    tan_v_model = SOURCE_TAN_V
    out = [0.0] * PIXELS
    for y in range(HEIGHT):
        for x in range(WIDTH):
            u = (2.0 * (x + 0.5) / WIDTH - 1.0) * tan_h
            v = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * tan_v_model
            source_y = (1.0 - v / tan_v_native) * 0.5 * HEIGHT - 0.5
            y0 = max(0, min(HEIGHT - 1, int(math.floor(source_y))))
            y1 = max(0, min(HEIGHT - 1, int(math.ceil(source_y))))
            a = raw[y0 * WIDTH + x]
            b = raw[y1 * WIDTH + x]
            if a is None or b is None:
                axial = None
            elif a is None:
                axial = b
            elif b is None:
                axial = a
            else:
                axial = min(a, b)
            if axial is None or not math.isfinite(axial):
                rng = 12.0
            else:
                rng = axial * math.sqrt(1.0 + u * u + v * v)
            out[y * WIDTH + x] = max(0.03, min(12.0, rng))
    return out


def pool(ranges):
    pooled = []
    for r in range(8):
        for c in range(10):
            i = (2 * r) * WIDTH + 2 * c
            pooled.append(min(ranges[i], ranges[i + 1], ranges[i + WIDTH], ranges[i + WIDTH + 1]))
    return pooled


def med(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def load():
    rows = []
    for line in FRAMES.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    if not FRAMES.exists():
        print(f"missing {FRAMES}; run the fixture first", file=sys.stderr)
        return 2
    records = load()
    settings = records[0]
    captures = [r for r in records if r.get("record") == "capture"]
    hfov = settings["field_of_view_rad"]
    tan_v_native = math.tan(0.5 * hfov) * HEIGHT / WIDTH
    hyp_tan_v = {
        "docs_aspect_0.8": tan_v_native,
        "source_model_0.75": SOURCE_TAN_V,
    }
    conventions = [("left_+Y_up_+Z", -1, 1), ("right_+Y_up_+Z", 1, 1),
                   ("left_+Y_down_+Z", -1, -1), ("right_+Y_down_+Z", 1, -1)]

    # --- convention + FOV hypothesis table over all pixels of all static captures ---
    convention_table = []
    for name, sy, sz in conventions:
        residuals = []
        for cap in captures:
            if cap["kind"] != 0:
                continue
            R = cap["depth_orientation"]
            origin = cap["depth_position"]
            raw = cap["raw"]
            for i in range(PIXELS):
                m = raw[i]
                if m is None:
                    continue
                d_local, _, _ = native_ray(i % WIDTH, i // WIDTH, tan_v_native, sy, sz)
                t, _ = hit_or_max(origin, mat_vec(R, d_local), ("convention", cap["case"], cap["capture"]))
                residuals.append(abs(m - t * d_local[0]))
        convention_table.append({
            "mapping": name,
            "median_abs_residual_m": med(residuals),
            "p95_abs_residual_m": sorted(residuals)[int(0.95 * len(residuals))] if residuals else None,
            "n": len(residuals),
        })
    best = min(convention_table, key=lambda e: e["median_abs_residual_m"])
    best_name = best["mapping"]
    _, sy, sz = next(c for c in conventions if c[0] == best_name)

    # --- per-row / per-column native tangent measurements (hypothesis-free magnitudes) ---
    row_probe, col_probe = [], []
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        row_acc, col_acc = {}, {}
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            m = raw[i]
            if m is None:
                continue
            d_local, u, v = native_ray(x, y, tan_v_native, sy, sz)
            t, name = hit_or_max(origin, mat_vec(R, d_local), ("probe", cap["case"], cap["capture"]))
            if name == "NONE":
                continue
            if name == "GROUND":
                row_acc.setdefault(y, []).append(origin[2] / m)
            elif name == "CEILING":
                row_acc.setdefault(y, []).append((5.0 - origin[2]) / m)
            elif name == "LEFT_WALL":
                col_acc.setdefault(x, []).append(abs(origin[1] - 3.0) / m)
            elif name == "RIGHT_WALL":
                col_acc.setdefault(x, []).append(abs(origin[1] + 3.0) / m)
        for y, values in sorted(row_acc.items()):
            if len(values) >= 4:
                row_probe.append({
                    "case": cap["case"], "row": y, "n": len(values),
                    "tan_v_magnitude_measured": med(values),
                    "hyp_0.8": tan_v_native * abs(1 - 2 * (y + 0.5) / HEIGHT),
                    "hyp_0.75": SOURCE_TAN_V * abs(1 - 2 * (y + 0.5) / HEIGHT),
                })
        for x, values in sorted(col_acc.items()):
            if len(values) >= 4 and abs(cap["requested_yaw"]) < 0.02:
                col_probe.append({
                    "case": cap["case"], "column": x, "n": len(values),
                    "tan_u_magnitude_measured": med(values),
                    "hyp_1.0": abs(2 * (x + 0.5) / WIDTH - 1.0),
                })

    tan_v_fit = None
    basis = []
    for entry in row_probe:
        g = abs(1 - 2 * (entry["row"] + 0.5) / HEIGHT)
        basis.append((g, entry["tan_v_magnitude_measured"]))
    if basis:
        num = sum(g * v for g, v in basis)
        den = sum(g * g for g, _ in basis)
        tan_v_fit = num / den if den else None

    # --- axial vs ray-range discrimination on perpendicular walls ---
    axial_test = []
    for cap in captures:
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        axial_res, ray_res = [], []
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            m = raw[i]
            if m is None:
                continue
            d_local, _, _ = native_ray(x, y, tan_v_native, sy, sz)
            t, name = hit_or_max(origin, mat_vec(R, d_local), ("wall", cap["case"], cap["capture"]))
            if name not in ("FRONT_WALL", "BACK_WALL", "LEFT_WALL", "RIGHT_WALL"):
                continue
            axial_res.append(abs(m - t * d_local[0]))
            ray_res.append(abs(m - t))
        if axial_res:
            axial_test.append({
                "case": cap["case"], "capture": cap["capture"], "n": len(axial_res),
                "median_axial_residual_m": med(axial_res),
                "median_ray_range_residual_m": med(ray_res),
                "max_axial_residual_m": max(axial_res),
            })

    # --- adapter replication and contract deltas ---
    adapter_rows = []
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        body = cap["rig_position"]
        adapter = adapter_resample(raw, hfov)
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            u_model, v_model = SOURCE_RAY(x, y, SOURCE_TAN_V)
            d_model = unit((1.0, -u_model, v_model))
            t_cam, name = hit_or_max(origin, mat_vec(R, d_model), ("adapter", cap["case"], cap["capture"]))
            t_body, _ = hit_or_max(body, mat_vec(R, d_model), ("adapter", cap["case"], cap["capture"]))
            adapter_rows.append({
                "case": cap["case"], "capture": cap["capture"], "pixel": i, "x": x, "y": y,
                "u_model": u_model, "v_model": v_model, "hit": name,
                "adapter_model_range_m": adapter[i],
                "analytic_model_range_from_camera_m": t_cam,
                "analytic_model_range_from_body_m": t_body,
                "source_body_ray_range_m": t_body,
                "native_axial_m": raw[i],
            })

    def summarize_adapter(rows):
        def errs(key_a, key_b, only_hit=None):
            out = []
            for r in rows:
                if r[key_a] is None or r[key_b] is None:
                    continue
                if only_hit and r["hit"] != only_hit:
                    continue
                out.append(abs(r[key_a] - r[key_b]))
            return out
        pairs = {
            "adapter_vs_model_ray_from_camera": errs("adapter_model_range_m", "analytic_model_range_from_camera_m"),
            "adapter_vs_model_ray_from_body": errs("adapter_model_range_m", "analytic_model_range_from_body_m"),
            "model_ray_camera_vs_body(mount_only)": errs("analytic_model_range_from_camera_m", "analytic_model_range_from_body_m"),
            "adapter_vs_model_ray_from_camera_plane_only":
                errs("adapter_model_range_m", "analytic_model_range_from_camera_m", "FRONT_WALL"),
            "adapter_vs_model_ray_from_camera_occlusion_edges":
                errs("adapter_model_range_m", "analytic_model_range_from_camera_m", "OCCLUDER"),
        }
        result = {}
        for key, values in pairs.items():
            if values:
                values_sorted = sorted(values)
                result[key] = {
                    "n": len(values),
                    "median_m": statistics.median(values),
                    "p95_m": values_sorted[int(0.95 * (len(values_sorted) - 1))],
                    "max_m": max(values),
                    "mean_m": sum(values) / len(values),
                }
        return result

    adapter_summary = summarize_adapter(adapter_rows)

    # boundary pixels: any 8-neighbour in the native grid sees a different primitive
    boundary_rows = []
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin = cap["depth_orientation"], cap["depth_position"]
        names = []
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            d_local, _, _ = native_ray(x, y, tan_v_native, sy, sz)
            _, name = hit_or_max(origin, mat_vec(R, d_local), ("boundary", cap["case"], cap["capture"]))
            names.append(name)
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            boundary = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and names[ny * WIDTH + nx] != names[i]:
                        boundary = True
            boundary_rows.append((cap["case"], cap["capture"], i, boundary))

    boundary_flags = {(c, k, i): b for c, k, i, b in boundary_rows}

    def error_set(rows, want_boundary):
        vals, signed = [], []
        for r in rows:
            key = (r["case"], r["capture"], r["pixel"])
            if boundary_flags.get(key, True) != want_boundary:
                continue
            d = r["adapter_model_range_m"] - r["analytic_model_range_from_camera_m"]
            vals.append(abs(d))
            signed.append(d)
        if not vals:
            return None
        vs = sorted(vals)
        ss = sorted(signed)
        return {"n": len(vals), "median_abs_m": statistics.median(vals),
                "p95_abs_m": vs[int(0.95 * (len(vs) - 1))], "max_abs_m": vs[-1],
                "median_signed_m": statistics.median(signed),
                "positive_fraction": sum(1 for v in signed if v > 1e-9) / len(signed),
                "negative_fraction": sum(1 for v in signed if v < -1e-9) / len(signed)}

    boundary_split = {
        "interior_pixels": error_set(adapter_rows, False),
        "boundary_pixels": error_set(adapter_rows, True),
    }

    worst_pixels = {}
    for cap in captures:
        if cap["kind"] != 0:
            continue
        rows = [r for r in adapter_rows if r["case"] == cap["case"] and r["capture"] == cap["capture"]]
        top = sorted(rows, key=lambda r: -abs(r["adapter_model_range_m"] - r["analytic_model_range_from_camera_m"]))[:5]
        worst_pixels[f"{cap['case']}|{cap['capture']}"] = [
            {"pixel": r["pixel"], "x": r["x"], "y": r["y"], "model_hit": r["hit"],
             "adapter_m": r["adapter_model_range_m"], "model_ray_m": r["analytic_model_range_from_camera_m"],
             "delta_m": r["adapter_model_range_m"] - r["analytic_model_range_from_camera_m"],
             "boundary": bool(boundary_flags.get((r["case"], r["capture"], r["pixel"]), True))}
            for r in top]

    def signed_stats(rows, key_a, key_b, only_hit=None):
        vals = []
        for r in rows:
            if r[key_a] is None or r[key_b] is None:
                continue
            if only_hit and r["hit"] != only_hit:
                continue
            vals.append(r[key_a] - r[key_b])
        if not vals:
            return None
        vs = sorted(vals)
        return {"n": len(vals), "median_signed_m": statistics.median(vals),
                "mean_signed_m": sum(vals) / len(vals),
                "p05_signed_m": vs[int(0.05 * (len(vs) - 1))],
                "p95_signed_m": vs[int(0.95 * (len(vs) - 1))],
                "min_signed_m": vs[0], "max_signed_m": vs[-1]}

    adapter_signed = {
        "adapter_minus_model_ray_from_camera": signed_stats(adapter_rows, "adapter_model_range_m",
                                                            "analytic_model_range_from_camera_m"),
        "adapter_minus_model_ray_from_body": signed_stats(adapter_rows, "adapter_model_range_m",
                                                          "analytic_model_range_from_body_m"),
        "by_model_hit": {
            hit: signed_stats(adapter_rows, "adapter_model_range_m", "analytic_model_range_from_camera_m", hit)
            for hit in sorted({r["hit"] for r in adapter_rows if r["hit"]})
        },
    }

    adapter_ratio_by_hit = {}
    for hit in sorted({r["hit"] for r in adapter_rows if r["hit"]}):
        ratios = [r["adapter_model_range_m"] / r["analytic_model_range_from_camera_m"]
                  for r in adapter_rows
                  if r["hit"] == hit and r["analytic_model_range_from_camera_m"] > 0.5]
        if ratios:
            adapter_ratio_by_hit[hit] = {"n": len(ratios), "median_ratio": statistics.median(ratios),
                                         "p05_ratio": sorted(ratios)[int(0.05 * (len(ratios) - 1))],
                                         "p95_ratio": sorted(ratios)[int(0.95 * (len(ratios) - 1))]}

    # structural disagreement: does the model-grid ray see the same primitive as the native-grid ray?
    hit_mismatch = []
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin = cap["depth_orientation"], cap["depth_position"]
        n_mismatch, examples = 0, []
        for i in range(PIXELS):
            x, y = i % WIDTH, i // WIDTH
            d_local, _, _ = native_ray(x, y, tan_v_native, sy, sz)
            _, native_hit = hit_or_max(origin, mat_vec(R, d_local), ("mismatch", cap["case"], cap["capture"]))
            u_model, v_model = SOURCE_RAY(x, y, SOURCE_TAN_V)
            _, model_hit = hit_or_max(origin, mat_vec(R, unit((1.0, -u_model, v_model))),
                                      ("mismatch", cap["case"], cap["capture"]))
            if native_hit != model_hit:
                n_mismatch += 1
                if len(examples) < 6:
                    examples.append({"pixel": i, "x": x, "y": y, "native_hit": native_hit, "model_hit": model_hit})
        hit_mismatch.append({"case": cap["case"], "capture": cap["capture"], "n_mismatch": n_mismatch,
                             "of": PIXELS, "examples": examples})

    # the deployed actor consumes the pooled 8x10 (v1) and raw 320 (v2) channels
    pooled_stats = []
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        body = cap["rig_position"]
        adapter = pool(adapter_resample(raw, hfov))
        model_cam, model_body = [], []
        for r in range(8):
            for c in range(10):
                vals_c, vals_b = [], []
                for dy in (0, 1):
                    for dx in (0, 1):
                        x, y = 2 * c + dx, 2 * r + dy
                        u_model, v_model = SOURCE_RAY(x, y, SOURCE_TAN_V)
                        d = mat_vec(R, unit((1.0, -u_model, v_model)))
                        t_cam, _ = hit_or_max(origin, d, ("pooled", cap["case"], cap["capture"]))
                        t_body, _ = hit_or_max(body, d, ("pooled", cap["case"], cap["capture"]))
                        vals_c.append(t_cam)
                        vals_b.append(t_body)
                model_cam.append(min(vals_c))
                model_body.append(min(vals_b))
        pooled_stats.append({
            "case": cap["case"], "capture": cap["capture"],
            "median_adapter_vs_model_camera_m": med([abs(a - b) for a, b in zip(adapter, model_cam)]),
            "max_adapter_vs_model_camera_m": max(abs(a - b) for a, b in zip(adapter, model_cam)),
            "median_adapter_vs_model_body_m": med([abs(a - b) for a, b in zip(adapter, model_body)]),
            "max_adapter_vs_model_body_m": max(abs(a - b) for a, b in zip(adapter, model_body)),
            "n_pooled_cells_changed_by_mount_gt_5mm": sum(1 for a, b in zip(model_cam, model_body) if abs(a - b) > 0.005),
        })

    vertical_sampling = []
    for y in range(HEIGHT):
        src_y = (1.0 - SOURCE_TAN_V * (1 - 2 * (y + 0.5) / HEIGHT) / tan_v_native) * 0.5 * HEIGHT - 0.5
        y0 = max(0, min(HEIGHT - 1, int(math.floor(src_y))))
        y1 = max(0, min(HEIGHT - 1, int(math.ceil(src_y))))
        vertical_sampling.append({
            "model_row": y, "source_row_float": src_y, "sampled_native_rows": [y0, y1],
            "v_model": SOURCE_TAN_V * (1 - 2 * (y + 0.5) / HEIGHT),
            "v_native_row0": tan_v_native * (1 - 2 * (y0 + 0.5) / HEIGHT),
            "v_native_row1": tan_v_native * (1 - 2 * (y1 + 0.5) / HEIGHT),
        })

    # --- discontinuity profile: measured vs both adapter-stage predictions ---
    discontinuity = []
    for cap in captures:
        if cap["case"] != "discontinuity" and not (cap["case"] == "front_near"):
            continue
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        adapter = adapter_resample(raw, hfov)
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                i = y * WIDTH + x
                d_local, _, _ = native_ray(x, y, tan_v_native, sy, sz)
                t_nat, name_nat = hit_or_max(origin, mat_vec(R, d_local),
                                             ("discontinuity", cap["case"], cap["capture"]))
                u_model, v_model = SOURCE_RAY(x, y, SOURCE_TAN_V)
                t_mod, name_mod = hit_or_max(origin, mat_vec(R, unit((1.0, -u_model, v_model))),
                                             ("discontinuity", cap["case"], cap["capture"]))
                row.append({
                    "x": x, "measured_native_axial_m": raw[i], "native_hit": name_nat,
                    "native_axial_pred_m": t_nat * d_local[0],
                    "model_hit": name_mod, "adapter_model_range_m": adapter[i],
                })
            discontinuity.append({"case": cap["case"], "capture": cap["capture"], "row": y, "pixels": row})

    # --- latency: does a first-refresh frame match its own pose or the previous one ---
    latency = []
    for idx, cap in enumerate(captures):
        if cap["kind"] != 1:
            continue
        adapter = adapter_resample(cap["raw"], hfov)
        candidates = [("own", cap)]
        if idx > 0:
            candidates.append(("previous", captures[idx - 1]))
        entry = {"case": cap["case"], "capture": cap["capture"], "step": cap["step"],
                 "requested_position": cap["requested_position"],
                 "measured_position": cap["rig_position"]}
        for label, ref in candidates:
            R, origin = ref["depth_orientation"], ref["depth_position"]
            errs = []
            for i in range(PIXELS):
                x, y = i % WIDTH, i // WIDTH
                d_local, _, _ = native_ray(x, y, tan_v_native, sy, sz)
                t, _ = hit_or_max(origin, mat_vec(R, d_local), ("latency", label, cap["capture"]))
                errs.append(abs(cap["raw"][i] - t * d_local[0]) if cap["raw"][i] is not None else 12.0)
            entry[f"median_residual_vs_{label}_pose_m"] = med(errs)
        own = entry.get("median_residual_vs_own_pose_m")
        prev = entry.get("median_residual_vs_previous_pose_m")
        entry["classification"] = ("no_stale_frame" if (own is not None and (prev is None or own <= prev))
                                   else "stale_frame_suspected")
        latency.append(entry)

    # --- outputs ---
    for cap in captures:
        if cap["kind"] != 0:
            continue
        R, origin, raw = cap["depth_orientation"], cap["depth_position"], cap["raw"]
        adapter = adapter_resample(raw, hfov)
        path = OUT_DIR / f"pixels_{cap['case']}_{cap['capture']:02d}.csv"
        with path.open("w") as f:
            f.write("case,capture,pixel,x,y,u_native,v_native,hit,measured_axial_m,native_axial_pred_m,"
                    "axial_residual_m,native_ray_range_pred_m,adapter_model_range_m,model_ray_from_body_m,"
                    "model_ray_from_camera_m\n")
            for i in range(PIXELS):
                x, y = i % WIDTH, i // WIDTH
                d_local, u, v = native_ray(x, y, tan_v_native, sy, sz)
                t, name = hit_or_max(origin, mat_vec(R, d_local), ("csv", cap["case"], cap["capture"]))
                u_model, v_model = SOURCE_RAY(x, y, SOURCE_TAN_V)
                d_model = unit((1.0, -u_model, v_model))
                t_cam, _ = hit_or_max(origin, mat_vec(R, d_model), ("csv", cap["case"], cap["capture"]))
                t_body, _ = hit_or_max(cap["rig_position"], mat_vec(R, d_model),
                                       ("csv", cap["case"], cap["capture"]))
                m = raw[i]
                pred_axial = t * d_local[0]
                f.write(f"{cap['case']},{cap['capture']},{i},{x},{y},{u:.6f},{v:.6f},{name},"
                        f"{'' if m is None else format(m, '.9g')},"
                        f"{format(pred_axial, '.9g')},"
                        f"{'' if m is None else format(m - pred_axial, '.9g')},"
                        f"{format(t, '.9g')},"
                        f"{format(adapter[i], '.9g')},"
                        f"{format(t_body, '.9g')},"
                        f"{format(t_cam, '.9g')}\n")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary = {
        "schema": "sensor-geometry-audit-summary-v1",
        "fixture": {
            "world": str(WORLD.relative_to(ROOT)),
            "world_sha256": sha256(WORLD),
            "controller": str(CONTROLLER.relative_to(ROOT)),
            "controller_sha256": sha256(CONTROLLER),
            "analyzer": str(Path(__file__).resolve().relative_to(ROOT)),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "raw_frames": str(FRAMES.relative_to(ROOT)),
            "raw_frames_sha256": sha256(FRAMES),
        },
        "declared_settings": {
            "field_of_view_rad": hfov,
            "width": WIDTH, "height": HEIGHT,
            "declared_native_vertical_half_tangent": tan_v_native,
            "source_model_vertical_half_tangent": SOURCE_TAN_V,
            "mount_translation_m": settings["mount_translation_m"],
            "sampling_period_ms": settings["sampling_period_ms"],
        },
        "convention_and_fov_hypotheses": convention_table,
        "selected_convention": best_name,
        "native_vertical_tangent": {
            "docs_aspect_value": tan_v_native,
            "source_model_value": SOURCE_TAN_V,
            "row_probe": row_probe,
            "column_probe": col_probe,
            "least_squares_tan_v_fit": tan_v_fit,
        },
        "axial_vs_ray_range": axial_test,
        "adapter_replication": adapter_summary,
        "boundary_split": boundary_split,
        "worst_adapter_pixels": worst_pixels,
        "adapter_signed_residuals": adapter_signed,
        "adapter_over_model_range_ratio_by_hit": adapter_ratio_by_hit,
        "source_model_vs_native_hit_mismatch": hit_mismatch,
        "pooled_actor_channels": pooled_stats,
        "vertical_sampling_map": vertical_sampling,
        "no_hit_rays_mapped_to_12m": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(NO_HIT.items())},
        "native_finite_pixel_counts": [
            {"case": c["case"], "capture": c["capture"], "finite": sum(1 for v in c["raw"] if v is not None),
             "of": PIXELS} for c in captures],
        "latency": latency,
        "captures": [
            {"case": c["case"], "capture": c["capture"], "kind": c["kind"], "step": c["step"],
             "time_s": c["time_s"], "requested_position": c["requested_position"],
             "requested_yaw": c["requested_yaw"], "depth_position": c["depth_position"],
             "rig_position": c["rig_position"], "gps": c["gps"], "imu_roll_pitch_yaw": c["imu_roll_pitch_yaw"]}
            for c in captures
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT_DIR / "discontinuity.json").write_text(json.dumps(discontinuity, indent=1))
    with (OUT_DIR / "adapter-pixels.csv").open("w") as f:
        keys = ["case", "capture", "pixel", "x", "y", "u_model", "v_model", "hit",
                "adapter_model_range_m", "analytic_model_range_from_camera_m",
                "analytic_model_range_from_body_m", "native_axial_m"]
        f.write(",".join(keys) + "\n")
        for r in adapter_rows:
            f.write(",".join("" if r[k] is None else (format(r[k], ".9g") if isinstance(r[k], float) else str(r[k]))
                             for k in keys) + "\n")

    print("== convention hypotheses ==")
    for entry in convention_table:
        print(f"  {entry['mapping']:>16}  median={entry['median_abs_residual_m']:.6f} m  n={entry['n']}")
    print(f"  selected = {best_name}")
    print("== native vertical tangent ==")
    print(f"  docs aspect tan_v={tan_v_native:.6f}  source model={SOURCE_TAN_V}  least-squares fit={tan_v_fit}")
    for entry in row_probe:
        print(f"  {entry['case']:>12} row={entry['row']:2d} n={entry['n']:3d} "
              f"measured={entry['tan_v_magnitude_measured']:.6f} hyp0.8={entry['hyp_0.8']:.6f} "
              f"hyp0.75={entry['hyp_0.75']:.6f}")
    for entry in col_probe:
        print(f"  {entry['case']:>12} col={entry['column']:2d} n={entry['n']:3d} "
              f"measured={entry['tan_u_magnitude_measured']:.6f} hyp1.0={entry['hyp_1.0']:.6f}")
    print("== axial vs ray range (perpendicular walls) ==")
    for entry in axial_test:
        print(f"  {entry['case']:>14} n={entry['n']:3d} axial_med={entry['median_axial_residual_m']:.6f} "
              f"ray_med={entry['median_ray_range_residual_m']:.6f} axial_max={entry['max_axial_residual_m']:.6f}")
    print("== adapter replication vs analytic model rays ==")
    for key, value in adapter_summary.items():
        print(f"  {key:>52}  n={value['n']:5d} median={value['median_m']:.6f} p95={value['p95_m']:.6f} max={value['max_m']:.6f}")
    print("== adapter signed residuals (adapter - analytic) ==")
    for key, value in adapter_signed.items():
        if key == "by_model_hit":
            for hit, entry in value.items():
                print(f"  by_hit {hit:>18} n={entry['n']:5d} signed_median={entry['median_signed_m']:+.6f} "
                      f"signed_mean={entry['mean_signed_m']:+.6f} p05={entry['p05_signed_m']:+.6f} p95={entry['p95_signed_m']:+.6f}")
            continue
        print(f"  {key:>40} n={value['n']:5d} signed_median={value['median_signed_m']:+.6f} "
              f"signed_mean={value['mean_signed_m']:+.6f} p05={value['p05_signed_m']:+.6f} p95={value['p95_signed_m']:+.6f}")
    print("== source-model vs native primitive disagreement per 20x16 frame ==")
    for entry in hit_mismatch:
        print(f"  {entry['case']:>14} mismatch={entry['n_mismatch']:3d}/{entry['of']}  "
              + "; ".join(f"px({e['x']},{e['y']}) nat={e['native_hit']} model={e['model_hit']}" for e in entry["examples"]))
    print("== pooled 8x10 actor channels (adapter vs analytic model) ==")
    for entry in pooled_stats:
        print(f"  {entry['case']:>14} cam: med={entry['median_adapter_vs_model_camera_m']:.6f} max={entry['max_adapter_vs_model_camera_m']:.6f} | "
              f"body: med={entry['median_adapter_vs_model_body_m']:.6f} max={entry['max_adapter_vs_model_body_m']:.6f} | "
              f"mount>5mm cells={entry['n_pooled_cells_changed_by_mount_gt_5mm']}/80")
    print("== native finiteness / 12 m no-hit mapping ==")
    for entry in summary["native_finite_pixel_counts"]:
        print(f"  {entry['case']:>14} finite={entry['finite']}/{entry['of']}")
    print(f"  no-hit rays mapped to the source 12 m contract: {sum(NO_HIT.values())} "
          f"({dict(sorted(NO_HIT.items())) if NO_HIT else '{}'})")
    print("== adapter error: interior vs boundary pixels ==")
    for key, value in boundary_split.items():
        if value is None:
            print(f"  {key}: none")
            continue
        print(f"  {key:>16} n={value['n']:5d} median_abs={value['median_abs_m']:.6f} p95_abs={value['p95_abs_m']:.6f} "
              f"max_abs={value['max_abs_m']:.6f} signed_median={value['median_signed_m']:+.6f} "
              f"pos={value['positive_fraction']:.3f} neg={value['negative_fraction']:.3f}")
    print("== worst adapter pixels per pose (|adapter - model ray from camera|) ==")
    for key, entries in worst_pixels.items():
        print(f"  {key}: " + "; ".join(
            f"({e['x']},{e['y']}) model={e['model_hit']} adapter={e['adapter_m']:.3f} ray={e['model_ray_m']:.3f} "
            f"d={e['delta_m']:+.3f} boundary={e['boundary']}" for e in entries[:3]))
    print("== latency ==")
    for entry in latency:
        print(f"  {entry['case']:>10} step={entry['step']:3d} own={entry['median_residual_vs_own_pose_m']:.6f} "
              f"previous={entry.get('median_residual_vs_previous_pose_m')} -> {entry['classification']}")
    print(f"wrote {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
