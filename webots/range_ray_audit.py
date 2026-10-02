#!/usr/bin/env python3
"""Compare one saved native Webots RangeFinder frame with analytic scene rays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mat_vec(matrix, value):
    return [dot(matrix[row * 3 : row * 3 + 3], value) for row in range(3)]


def ray_box(origin, direction, center, half):
    lo, hi = -math.inf, math.inf
    for axis in range(3):
        p = origin[axis] - center[axis]
        d = direction[axis]
        if abs(d) < 1e-10:
            if abs(p) > half[axis]:
                return math.inf
            continue
        a, b = (-half[axis] - p) / d, (half[axis] - p) / d
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo if lo >= 0 else hi) if hi >= max(lo, 0.0) else math.inf


def ray_sphere(origin, direction, center, radius):
    q = [origin[i] - center[i] for i in range(3)]
    b = dot(q, direction)
    disc = b * b - (dot(q, q) - radius * radius)
    if disc < 0:
        return math.inf
    near, far = -b - math.sqrt(disc), -b + math.sqrt(disc)
    return near if near >= 0 else far if far >= 0 else math.inf


def ray_cylinder_z(origin, direction, center, radius, half_height):
    q = [origin[i] - center[i] for i in range(3)]
    best = math.inf
    a = direction[0] ** 2 + direction[1] ** 2
    b = q[0] * direction[0] + q[1] * direction[1]
    c = q[0] ** 2 + q[1] ** 2 - radius**2
    disc = b * b - a * c
    if a > 1e-12 and disc >= 0:
        for t in ((-b - math.sqrt(disc)) / a, (-b + math.sqrt(disc)) / a):
            if t >= 0 and abs(q[2] + t * direction[2]) <= half_height:
                best = min(best, t)
    if abs(direction[2]) > 1e-12:
        for cap in (-half_height, half_height):
            t = (cap - q[2]) / direction[2]
            if t >= 0:
                x, y = q[0] + t * direction[0], q[1] + t * direction[1]
                if x * x + y * y <= radius * radius:
                    best = min(best, t)
    return best


def expected_range(origin, direction, scene, max_range):
    nearest = max_range
    for obstacle in scene["obstacles"]:
        center = obstacle["center"]
        if obstacle["kind"] == "box":
            half = [0.5 * side for side in obstacle["size"]]
            hit = ray_box(origin, direction, center, half)
        elif obstacle["kind"] == "sphere":
            hit = ray_sphere(origin, direction, center, obstacle["radius"])
        elif obstacle["kind"] == "cylinder":
            hit = ray_cylinder_z(origin, direction, center, obstacle["radius"], obstacle["height"] / 2)
        else:
            raise ValueError(f"unsupported obstacle kind: {obstacle['kind']}")
        nearest = min(nearest, hit)

    if scene["bounded_room"]:
        lo, hi = (-2.0, -5.0, 0.0), (14.0, 5.0, 5.0)
        for axis in range(3):
            if abs(direction[axis]) < 1e-10:
                continue
            t = ((hi[axis] if direction[axis] > 0 else lo[axis]) - origin[axis]) / direction[axis]
            if t >= 0:
                nearest = min(nearest, t)
    else:
        # The open benchmark has a finite 18 x 12 m ground slab with its top at z=0.
        if direction[2] < -1e-10:
            t = -origin[2] / direction[2]
            hit = [origin[i] + t * direction[i] for i in range(3)]
            if t >= 0 and 2.0 - 9.0 <= hit[0] <= 2.0 + 9.0 and -6.0 <= hit[1] <= 6.0:
                nearest = min(nearest, t)
    return min(nearest, max_range)


def near_cylinder_tangent(origin, direction, scene, tolerance=0.002):
    for obstacle in scene["obstacles"]:
        if obstacle["kind"] != "cylinder":
            continue
        q = [origin[i] - obstacle["center"][i] for i in range(3)]
        a = direction[0] ** 2 + direction[1] ** 2
        if a < 1e-12:
            continue
        t = -(q[0] * direction[0] + q[1] * direction[1]) / a
        z = q[2] + t * direction[2]
        if t < 0 or abs(z) > obstacle["height"] / 2:
            continue
        radial = math.hypot(q[0] + t * direction[0], q[1] + t * direction[1])
        if abs(radial - obstacle["radius"]) <= tolerance:
            return True
    return False


def audit(frame_path: Path, scene_path: Path, max_range: float = 12.0):
    frame = json.loads(frame_path.read_text())
    scene = json.loads(scene_path.read_text())
    width, height = int(frame["width"]), int(frame["height"])
    if (width, height) != (20, 16):
        raise ValueError(f"expected native 20x16 frame, got {width}x{height}")
    raw = frame["native_axial_depth_m"]
    if len(raw) != width * height:
        raise ValueError("native image size does not match its dimensions")
    rotation = frame.get("camera_rotation_supervisor_row_major", frame["body_rotation_sensor_row_major"])
    body_rotation = frame["body_rotation_sensor_row_major"]
    origin = frame["camera_position_supervisor_xyz_m"]
    tan_h = math.tan(float(frame["horizontal_fov_rad"]) / 2)
    tan_v = float(frame["native_vertical_tangent"])
    errors = []
    non_grazing_errors = []
    grazing_count = 0
    resample_errors = []
    worst = None
    model_vertical_tangent = 0.75
    for y in range(height):
        v = (1 - 2 * (y + 0.5) / height) * tan_v
        for x in range(width):
            u = (2 * (x + 0.5) / width - 1) * tan_h
            norm = math.sqrt(1 + u * u + v * v)
            local_ray = [1 / norm, -u / norm, v / norm]
            direction = mat_vec(rotation, local_ray)
            predicted = expected_range(origin, direction, scene, max_range)
            index = y * width + x
            measured_slant = float(raw[index]) * norm
            if predicted < max_range - 1e-4 and measured_slant < max_range - 0.01:
                error = measured_slant - predicted
                errors.append(error)
                if near_cylinder_tangent(origin, direction, scene):
                    grazing_count += 1
                else:
                    non_grazing_errors.append(error)
            # Reproduce the controller's vertical ray resampling to separate
            # implementation error from the intentional .8-to-.75 FOV change.
            model_v = (1 - 2 * (y + 0.5) / height) * model_vertical_tangent
            source_y = (1 - model_v / tan_v) * 0.5 * height - 0.5
            y0 = max(0, min(height - 1, math.floor(source_y)))
            y1 = max(0, min(height - 1, math.ceil(source_y)))
            u = (2 * (x + 0.5) / width - 1) * tan_h
            axial = min(float(raw[y0 * width + x]), float(raw[y1 * width + x]))
            expected_normalized = min(max(axial * math.sqrt(1 + u * u + model_v * model_v), 0.03), max_range)
            resample_errors.append(float(frame["normalized_ray_range_m"][index]) - expected_normalized)
            if predicted < max_range - 1e-4 and measured_slant < max_range - 0.01:
                sample = (abs(measured_slant - predicted), x, y, measured_slant, predicted)
                if worst is None or sample[0] > worst[0]:
                    worst = sample
    if not errors:
        raise ValueError("no pixels saw a modeled surface below max range")
    absolute = sorted(abs(error) for error in errors)
    signed = sorted(errors)
    p95_index = min(len(absolute) - 1, math.ceil(0.95 * len(absolute)) - 1)
    record = {
        "schema": "webots-native-ray-comparison-v1",
        "scene": str(scene_path),
        "frame": str(frame_path),
        "seed": scene["seed"],
        "family": scene["family"],
        "bounded_room": scene["bounded_room"],
        "capture_step": frame["step"],
        "capture_time_s": frame["time_s"],
        "tested_pixels": len(errors),
        "absolute_error_median_m": absolute[len(absolute) // 2],
        "absolute_error_p95_m": absolute[p95_index],
        "absolute_error_max_m": absolute[-1],
        "near_tangent_pixel_count_within_2mm": grazing_count,
        "absolute_error_max_excluding_near_tangent_m": max(abs(error) for error in non_grazing_errors),
        "signed_error_mean_m": sum(errors) / len(errors),
        "worst_pixel": {"x": worst[1], "y": worst[2], "measured_slant_m": worst[3], "analytic_range_m": worst[4]},
        "normalization_reimplementation_error_max_m": max(abs(error) for error in resample_errors),
        "camera_body_rotation_max_abs_delta": max(abs(a - b) for a, b in zip(rotation, body_rotation)),
        "actor_input_or_ground_truth_used": False,
        "ray_model": "native Webots FOV pixel-center rays; body sensor rotation; measured camera-origin translation; exact metadata boxes/spheres/local-Z cylinders; open floor or bounded room shell",
    }
    output = frame_path.with_name("range-ray-comparison.json")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("scene", type=Path, help="matching challenge metadata JSON")
    args = parser.parse_args()
    audit(args.frame, args.scene)


if __name__ == "__main__":
    main()
