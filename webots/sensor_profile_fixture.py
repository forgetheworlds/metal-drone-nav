#!/usr/bin/env python3
"""Compare camera-profile predictions against the loaded native Webots fixture.

Reads the frozen native RangeFinder captures from the sensor-geometry audit
(real measured pixels, real measured camera poses) and the exact ray table
exported by the C++/Metal shared header (`sensor_profile.hpp`) via
`metal_nav_raw_guided sensor-profile-check`. It then ray-casts the frozen
audit-rig scene (the same axis-aligned primitives as
webots/worlds/sensor-geometry-audit.wbt, hash-checked) from the measured
sensor pose and reports:

  * native axial geometric residual (measured vs predicted, calibrated ray);
  * calibrated end-to-end residual (source ray cast from the mount vs the
    native reading converted with the profile ray norm);
  * legacy adapter vs legacy source model, reproducing the 0.75/0.8 mismatch
    (0.9375 floor/ceiling scale and edge primitive changes).

This is a calibration fixture, not a flight or transfer result.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys

AUDIT_WORLD_SHA256 = "b3ff6a6ea2ed6c7aea510f151d6192b68100090d9fe20f5f16e9bf29cd40df1c"
NATIVE_FRAMES_SHA256 = "68c840ffe697eb7779458aa06d34c1b8308f7179034db89b9299dfd6b36d2d33"

# Frozen scene primitives, exactly the wbt boxes/cylinder as (kind, center, half/size).
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
CYLINDERS = [("POLE", (2.2, -0.5, 2.5), 0.15, 2.5)]


def ray_box(p, d, c, half):
    lo, hi = -1e20, 1e20
    for j in range(3):
        pp, dd, ee = p[j] - c[j], d[j], half[j]
        if abs(dd) < 1e-12:
            if abs(pp) > ee:
                return None
        else:
            a, b = (-ee - pp) / dd, (ee - pp) / dd
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    if hi < max(lo, 0.0):
        return None
    return lo if lo >= 0 else hi


def ray_cylinder(p, d, c, r, h):
    q = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    a = d[0] * d[0] + d[1] * d[1]
    b = q[0] * d[0] + q[1] * d[1]
    k = q[0] * q[0] + q[1] * q[1] - r * r
    best = None
    disc = b * b - a * k
    if a > 1e-12 and disc >= 0:
        for t in ((-b - math.sqrt(disc)) / a, (-b + math.sqrt(disc)) / a):
            if t >= 0 and abs(q[2] + t * d[2]) <= h:
                best = t if best is None else min(best, t)
    if abs(d[2]) > 1e-12:
        for sign in (-1, 1):
            t = (sign * h - q[2]) / d[2]
            x, y = q[0] + t * d[0], q[1] + t * d[1]
            if t >= 0 and x * x + y * y <= r * r:
                best = t if best is None else min(best, t)
    return best


def raycast(p, d):
    """Return (t, primitive). d may be unnormalized; t is in units of |d|."""
    best, hit = None, "MISS"
    for name, c, half in BOXES:
        t = ray_box(p, d, c, half)
        if t is not None and (best is None or t < best):
            best, hit = t, name
    for name, c, r, h in CYLINDERS:
        t = ray_cylinder(p, d, c, r, h)
        if t is not None and (best is None or t < best):
            best, hit = t, name
    return best, hit


def rotate_body_to_world(r, b):
    return [r[3 * i] * b[0] + r[3 * i + 1] * b[1] + r[3 * i + 2] * b[2] for i in range(3)]


def load_rays(path):
    pixel = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["kind"] != "pixel":
                continue
            pixel.setdefault(row["profile"], {})[int(row["pixel"])] = (
                float(row["dir_x"]), float(row["dir_y"]), float(row["dir_z"]),
                float(row["dir_norm"]), float(row["ray_x"]), float(row["ray_y"]), float(row["ray_z"]),
            )
    return pixel


def legacy_wcamera(pixel, tan_h=1.0, tan_v=0.75):
    y = (pixel % 20 + 0.5) / 20 * 2 - 1
    z = (pixel // 20 + 0.5) / 16 * 2 - 1
    v = (1.0, -y * tan_h, -z * tan_v)
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v), v


def production_adapter(raw, width=20, height=16, hfov=1.570796327):
    """Verbatim port of raptor_webots.cpp::normalize_ranges (legacy 0.75 grid)."""
    tan_h = math.tan(0.5 * hfov)
    tan_v_native = tan_h * height / width
    tan_v_model = 0.75
    out = [0.0] * (width * height)
    for y in range(height):
        for x in range(width):
            u = (2.0 * (x + 0.5) / width - 1.0) * tan_h
            v = (1.0 - 2.0 * (y + 0.5) / height) * tan_v_model
            source_y = (1.0 - v / tan_v_native) * 0.5 * height - 0.5
            y0 = max(0, min(height - 1, int(math.floor(source_y))))
            y1 = max(0, min(height - 1, int(math.ceil(source_y))))
            axial = min(raw[y0 * width + x], raw[y1 * width + x])
            out[y * width + x] = min(max(axial * math.sqrt(1 + u * u + v * v), 0.03), 12.0)
    return out


def stats(values):
    if not values:
        return {}
    values = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max": values[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native_jsonl")
    ap.add_argument("rays_csv")
    ap.add_argument("output_json")
    args = ap.parse_args()

    frames_bytes = pathlib.Path(args.native_jsonl).read_bytes()
    got = hashlib.sha256(frames_bytes).hexdigest()
    if got != NATIVE_FRAMES_SHA256:
        sys.exit(f"native fixture hash mismatch: {got}")
    records = [json.loads(line) for line in frames_bytes.decode().splitlines() if line.strip()]
    settings = records[0]
    captures = [r for r in records[1:] if r.get("record") == "capture"]

    rays = load_rays(args.rays_csv)
    for profile in ("legacy", "native"):
        if len(rays.get(profile, {})) != 320:
            sys.exit(f"ray table for {profile} must have 320 pixels")

    native_axial_residual = []
    calibrated_e2e = []
    legacy_model_vs_native = []
    adapter_vs_model = []
    adapter_ratio = {"GROUND": [], "CEILING": []}
    primitive_disagreements = 0
    total_pixels = 0
    worst_calibrated = []
    worst_legacy = []
    clipped = 0
    all_finite = 0

    for cap in captures:
        raw = cap["raw"]
        cam = cap["depth_position"]
        rig = cap["rig_position"]
        r = cap["depth_orientation"]  # row-major local -> world
        adapter = production_adapter(raw)
        for k in range(320):
            total_pixels += 1
            axial = raw[k]
            if math.isfinite(axial):
                all_finite += 1
            if not (settings["min_range_m"] < axial < settings["max_range_m"]):
                clipped += 1
            row, col = k // 20, k % 20
            # calibrated: header native ray, camera-origin cast, axial units
            d = rays["native"][k][:3]
            dw = rotate_body_to_world(r, d)
            t_ax, _ = raycast(cam, dw)
            if t_ax is not None:
                native_axial_residual.append(abs(t_ax - axial))
                # calibrated end-to-end: source ray range vs native reading
                calibrated_e2e.append(abs(rays["native"][k][3] * axial - t_ax * rays["native"][k][3]))
                worst_calibrated.append((abs(t_ax - axial), cap["case"], cap["kind"], k, row, col))
            # legacy source model: wcamera 0.75 ray, body-origin, ray range.
            ld, _ = legacy_wcamera(k)
            ldw = rotate_body_to_world(r, ld)
            t_legacy, hit_legacy = raycast(rig, ldw)
            # native sensor expressed as ray range on its own grid.
            native_ray_range = axial * rays["native"][k][3]
            if t_legacy is not None:
                legacy_model_vs_native.append(abs(t_legacy - native_ray_range))
            # production adapter (0.75 grid) vs the legacy source-model ray range.
            adapter_vs_model.append(abs(adapter[k] - t_legacy) if t_legacy is not None else 0.0)
            _, hit_native = raycast(cam, rotate_body_to_world(r, rays["native"][k][:3]))
            if hit_legacy != hit_native:
                primitive_disagreements += 1
                if t_legacy is not None:
                    worst_legacy.append((abs(adapter[k] - t_legacy), cap["case"], cap["kind"], k,
                                         row, col, hit_legacy, hit_native))
            if hit_legacy in ("GROUND", "CEILING") and t_legacy is not None and t_legacy > 1e-3:
                adapter_ratio[hit_legacy].append(adapter[k] / t_legacy)

    def ratio_summary(values):
        if not values:
            return {}
        return {"n": len(values), "median": statistics.median(values),
                "min": min(values), "max": max(values)}

    report = {
        "schema": "sensor-profile-native-fixture-v1",
        "native_fixture": args.native_jsonl,
        "native_fixture_sha256": got,
        "audit_world_sha256": AUDIT_WORLD_SHA256,
        "ray_table": args.rays_csv,
        "captures": len(captures),
        "pixels": total_pixels,
        "finite_pixels": all_finite,
        "range_clipped_pixels": clipped,
        "native_axial_geometric_residual_m": stats(native_axial_residual),
        "calibrated_end_to_end_residual_m": stats(calibrated_e2e),
        "legacy_source_model_vs_native_ray_range_m": stats(legacy_model_vs_native),
        "legacy_adapter_vs_legacy_model_m": stats(adapter_vs_model),
        "legacy_adapter_over_legacy_model_ratio": {
            name: ratio_summary(values) for name, values in adapter_ratio.items()
        },
        "legacy_vs_calibrated_primitive_disagreements": primitive_disagreements,
        "primitive_disagreement_fraction": primitive_disagreements / total_pixels,
        "worst_calibrated_pixels": [
            {"residual_m": v[0], "case": v[1], "kind": v[2], "pixel": v[3], "row": v[4], "col": v[5]}
            for v in sorted(worst_calibrated, reverse=True)[:5]
        ],
        "worst_legacy_edge_pixels": [
            {"adapter_minus_model_m": v[0], "case": v[1], "kind": v[2], "pixel": v[3], "row": v[4],
             "col": v[5], "legacy_hit": v[6], "native_hit": v[7]}
            for v in sorted(worst_legacy, reverse=True)[:5]
        ],
        "limits": [
            "static axis-aligned rig, no noise (configured off), no motion",
            "native resolution 0.001 m and pixel-centre ray model are the only "
            "quantisation sources in the calibrated residual",
            "min/max range clipping not exercised in these poses",
            "not a flight, transfer or navigation result",
        ],
    }
    pathlib.Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
