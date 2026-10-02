#!/usr/bin/env python3
"""Run reproducible closed-loop Webots A-to-B episodes and preserve raw traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORLDS = ROOT / "worlds"
RESULTS = ROOT / "results"
DEFAULT_WEBOTS = Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots")


def with_custom_data(source: str, updates: dict[str, str]) -> str:
    match = re.search(r'customData\s+"([^"]*)"', source)
    if not match:
        raise ValueError("world has no Robot.customData field")
    fields = dict(
        part.split("=", 1)
        for part in match.group(1).split(";")
        if "=" in part
    )
    fields.update(updates)
    replacement = "customData \"" + ";".join(f"{key}={value}" for key, value in fields.items()) + "\""
    return source[: match.start()] + replacement + source[match.end() :]


def recording_viewpoint(source: str, target: tuple[float, float, float],
                        camera: tuple[float, float, float]) -> str:
    """Aim a fixed wide 3D view at the selected route using Webots +X-forward axes."""
    forward = [target[i] - camera[i] for i in range(3)]
    length = math.sqrt(sum(value * value for value in forward))
    forward = [value / length for value in forward]
    world_up = (0.0, 0.0, 1.0)
    left = [world_up[1] * forward[2] - world_up[2] * forward[1],
            world_up[2] * forward[0] - world_up[0] * forward[2],
            world_up[0] * forward[1] - world_up[1] * forward[0]]
    left_norm = math.sqrt(sum(value * value for value in left))
    left = [value / left_norm for value in left]
    up = [forward[1] * left[2] - forward[2] * left[1],
          forward[2] * left[0] - forward[0] * left[2],
          forward[0] * left[1] - forward[1] * left[0]]
    # R2025a WbViewpoint uses local +X forward, +Y left and +Z up.
    matrix = [[forward[0], left[0], up[0]],
              [forward[1], left[1], up[1]],
              [forward[2], left[2], up[2]]]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    angle = 2 * math.acos(max(-1.0, min(1.0, qw)))
    sine = math.sqrt(max(1e-16, 1.0 - qw * qw))
    axis = (qx / sine, qy / sine, qz / sine)
    viewpoint = "DEF RLRecordingViewpoint Viewpoint { position " + " ".join(f"{v:.9f}" for v in camera)
    viewpoint += " orientation " + " ".join(f"{v:.9f}" for v in axis) + f" {angle:.9f} fieldOfView 1.2 }}"
    source, replacements = re.subn(r"Viewpoint\s*\{[^}]*\}", viewpoint, source, count=1)
    if replacements != 1:
        raise ValueError("recording world must contain one Viewpoint node")
    return source


def inspect_native_movie(movie: Path) -> dict:
    """Decode the actual recording and reject sustained blank frames."""
    decoder = shutil.which("ffmpeg")
    if decoder is None:
        raise RuntimeError("Native movie checks require ffmpeg on PATH")
    command = [decoder, "-hide_banner", "-nostdin", "-i", str(movie),
               "-vf", "blackdetect=d=0.04:pix_th=0.01:pic_th=0.98",
               "-an", "-f", "null", "-"]
    decoded = subprocess.run(command, capture_output=True, text=True, timeout=30)
    black_seconds = sum(float(value) for value in
                        re.findall(r"black_duration:([0-9.]+)", decoded.stderr))
    return {"passed": decoded.returncode == 0 and black_seconds <= 0.2,
            "decoder_exit_code": decoded.returncode,
            "near_black_duration_s": black_seconds, "maximum_allowed_s": 0.2,
            "decoder": "ffmpeg blackdetect; all frames decoded",
            "scope": "blank-frame check; inspect drone visibility separately"}


def run_one(webots: Path, world_name: str, seed: int, policy: str, steps: int, port: int = 23456,
            physics_step_ms: int = 1, motor_sampling: str = "average", physics_profile: str = "hover",
            sensor_audit: bool = False, goal_objective: str = "entry", record_movie: bool = False,
            capture_trajectory: bool = False, scene_metadata: Path | None = None) -> dict:
    if goal_objective not in ("entry", "hold"):
        raise ValueError("goal_objective must be entry or hold")
    base = WORLDS / f"{world_name}.wbt"
    if not base.is_file():
        raise FileNotFoundError(base)
    world = WORLDS / f".run-{world_name}-{Path(policy).stem}-{seed}.wbt"
    suffix = ("-hold" if goal_objective == "hold" else "") + ("-recording" if record_movie else "")
    if capture_trajectory:
        suffix += "-trajectory"
    output_dir = RESULTS / Path(policy).stem / f"{world_name}-seed-{seed}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    movie_path = (output_dir / "flight.mp4").resolve()
    view_snapshot_path = (output_dir / "view-at-1s.jpg").resolve()
    updates = {
        "phase": "navigation",
        "seed": str(seed),
        "policy": policy,
        "max_steps": str(steps),
        "profile": physics_profile,
        "motor_sampling": motor_sampling,
        "sensor_audit": "1" if sensor_audit else "0",
        "goal_objective": goal_objective,
        "capture_trajectory": "1" if capture_trajectory else "0",
    }
    if record_movie:
        updates["movie_file"] = str(movie_path)
        updates["view_snapshot_file"] = str(view_snapshot_path)
    source = base.read_text()
    if record_movie:
        target = (2.0, 0.0, 1.6)
        camera = (-2.0, -1.0, 3.3)
        challenge_metadata = RESULTS / "challenges" / f"{world_name}.json"
        if challenge_metadata.is_file():
            metadata = json.loads(challenge_metadata.read_text())
            if metadata.get("family") == "doorway":
                parameters = metadata["generator_parameters"]
                target = (parameters["wall_x_m"], parameters["opening_center_y_m"], parameters["opening_center_z_m"])
        elif world_name.startswith("recording_metal_"):
            scene_name = world_name.removeprefix("recording_")
            metal_metadata = RESULTS / "metal_scenes" / f"{scene_name}.json"
            if metal_metadata.is_file():
                metadata = json.loads(metal_metadata.read_text())
                start, goal = metadata["start_xyz_m"], metadata["goal_xyz_m"]
                target = ((float(start[0]) + float(goal[0])) / 2,
                          (float(start[1]) + float(goal[1])) / 2,
                          (float(start[2]) + float(goal[2])) / 2 + 0.5)
                camera = (-2.0, -4.0, 4.5)
        source = recording_viewpoint(source, target, camera)
    source = source.replace("basicTimeStep 10", f"basicTimeStep {physics_step_ms}", 1)
    world_parameters = {}
    if world_name == "a_to_b_offset_box":
        rng = random.Random(seed)
        offset_y = rng.uniform(-0.55, 0.55)
        world_parameters["box_y_offset_m"] = round(offset_y, 5)
        source = source.replace(
            "translation 2.0 0.0 1.5",
            f"translation 2.0 {offset_y:.5f} 1.5",
            1,
        )
    world.write_text(with_custom_data(source, updates))
    shutil.copy2(world, output_dir / "route.wbt")
    try:
        if record_movie:
            # A stale per-world project file can leave the RangeFinder panel in
            # the main view. Start this isolated recording world on its 3D view.
            project_file = WORLDS / f".{world.stem}.wbproj"
            project_file.write_text("Webots Project File version R2025a\nrenderingDevicePerspectives: \n")
        for stale in (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker", RESULTS / "last-run-trace.csv",
                      RESULTS / "range-ray-audit.json"):
            stale.unlink(missing_ok=True)
        command = [str(webots), f"--port={port}", "--batch", "--mode=fast"]
        if not record_movie:
            command.append("--minimize")
            command.append("--no-rendering")
        else:
            command[-1] = "--mode=realtime"
        command += ["--stdout", "--stderr", str(world)]
        if record_movie:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            owned_command = subprocess.run(["ps","-p",str(process.pid),"-o","command="],capture_output=True,text=True,check=False).stdout
            if str(world) not in owned_command:
                process.terminate()
                raise RuntimeError(f"refusing movie run: spawned process does not own {world}")
            # A minimized or covered Webots window can export empty frames.
            # Bring only this RL-owned Webots process forward for native rendering.
            activation = subprocess.run(
                ["osascript","-e",f'tell application "System Events" to tell (first application process whose unix id is {process.pid}) to set frontmost to true'],
                capture_output=True,text=True,timeout=10,check=False)
            time.sleep(0.35)
            stdout,stderr=process.communicate(timeout=120)
            returncode=process.returncode
            (output_dir/"webots.log").write_text(stdout+stderr+"\nVIEW_ACTIVATION_EXIT="+str(activation.returncode)+"\n"+activation.stderr)
        else:
            process = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90,check=False)
            returncode=process.returncode
            (output_dir/"webots.log").write_text(process.stdout+process.stderr)
        if returncode != 0:
            raise RuntimeError(f"Webots exited {returncode}; see {output_dir / 'webots.log'}")
        result_path = RESULTS / "last-run.json"
        marker_path = RESULTS / "last-run-exit.marker"
        trace_path = RESULTS / "last-run-trace.csv"
        if not result_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"controller did not write its result marker; see {output_dir / 'webots.log'}")
        result = json.loads(result_path.read_text())
        shutil.copy2(marker_path, output_dir / "exit.marker")
        if trace_path.is_file():
            shutil.copy2(trace_path, output_dir / "trace.csv")
        if capture_trajectory:
            with (output_dir / "trace.csv").open(newline="") as source:
                trace_rows = list(csv.DictReader(source))
            required = {"x", "y", "z", "q_w", "q_x", "q_y", "q_z"}
            if not trace_rows or not required.issubset(trace_rows[0]):
                raise RuntimeError("trajectory capture did not write dense world position and quaternion samples")
            result.update(trajectory_trace=True, trajectory_samples=len(trace_rows),
                          trajectory_sample_rate_hz=100, trajectory_frame="Webots ENU world", quaternion_order="wxyz")
        audit_path = RESULTS / "range-ray-audit.json"
        if sensor_audit and audit_path.is_file():
            shutil.copy2(audit_path, output_dir / "range-ray-audit.json")
        result.update(world=world_name, policy=Path(policy).name, seed=seed,
                      physics_step_ms=physics_step_ms, motor_sampling=motor_sampling,
                      physics_profile=physics_profile, goal_objective=goal_objective, **world_parameters)
        if record_movie:
            if not movie_path.is_file() or movie_path.stat().st_size == 0:
                raise RuntimeError(f"Webots did not create the rendered movie: {movie_path}")
            if not view_snapshot_path.is_file() or view_snapshot_path.stat().st_size == 0:
                raise RuntimeError(f"Webots did not export its 1 s main-view snapshot: {view_snapshot_path}")
            result.update(movie_recording=True, movie_file=str(movie_path), movie_bytes=movie_path.stat().st_size,
                          view_snapshot=str(view_snapshot_path), view_snapshot_bytes=view_snapshot_path.stat().st_size)
        selected_metadata = scene_metadata
        if selected_metadata is None:
            for candidate in (RESULTS / "challenges" / f"{world_name}.json",
                              RESULTS / "metal_scenes" / f"{world_name}.json"):
                if candidate.is_file():
                    selected_metadata = candidate
                    break
        if selected_metadata is not None:
            selected_metadata = selected_metadata.resolve()
            shutil.copy2(selected_metadata, output_dir / "scene-metadata.json")
            result["scene_metadata"] = str((output_dir / "scene-metadata.json").resolve())
        if record_movie:
            result["movie_visual_check"] = inspect_native_movie(movie_path)
        (output_dir / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
        if record_movie and not result["movie_visual_check"]["passed"]:
            raise RuntimeError(f"Native recording failed decoded-frame checks; physical receipt is preserved in {output_dir}")
        return result
    finally:
        world.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--worlds", default="a_to_b,a_to_b_offset_box")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--policies", default="../assets/navigation.bin")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--port", type=int, default=23456, help="isolated Webots controller port")
    parser.add_argument("--physics-step-ms", type=int, default=1)
    parser.add_argument("--motor-sampling", choices=("end", "average"), default="average")
    parser.add_argument("--physics-profile", default="hover")
    parser.add_argument("--sensor-audit", action="store_true", help="save one native RangeFinder frame and capture pose for raycast audit")
    parser.add_argument("--goal-objective", choices=("entry", "hold"), default="entry",
                        help="stop on first goal entry or after a stable low-speed hold")
    parser.add_argument("--record-movie", action="store_true", help="render and save a native Webots MP4")
    parser.add_argument("--capture-trajectory", action="store_true", help="save every 100 Hz Webots pose and quaternion as flight evidence")
    args = parser.parse_args()
    if not args.webots.is_file():
        parser.error(f"Webots executable not found: {args.webots}")
    rows = []
    for policy in args.policies.split(","):
        for world in args.worlds.split(","):
            for seed_text in args.seeds.split(","):
                seed = int(seed_text)
                print(f"run policy={policy} world={world} seed={seed}", flush=True)
                rows.append(run_one(args.webots, world, seed, policy, args.steps, args.port,
                                    args.physics_step_ms, args.motor_sampling, args.physics_profile,
                                    args.sensor_audit, args.goal_objective, args.record_movie,
                                    args.capture_trajectory))
    summary = RESULTS / ("benchmark-hold.csv" if args.goal_objective == "hold" else "benchmark.csv")
    fields = [
        "policy", "world", "seed", "success", "collision", "timeout", "steps", "time_s",
        "path_m", "final_error_m", "peak_speed_mps", "min_sensor_range_m", "altitude_min_m",
        "altitude_max_m", "tracking_rms_mps", "sensor_updates", "navigation_loaded", "box_y_offset_m",
        "physics_step_ms", "motor_sampling", "physics_profile",
        "goal_objective", "goal_radius_entry_count", "goal_radius_entry_first_time_s",
        "goal_radius_entry_last_time_s", "goal_dwell_s", "final_world_speed_mps",
        "movie_recording", "movie_file", "movie_bytes",
        "trajectory_trace", "trajectory_samples", "trajectory_sample_rate_hz", "trajectory_frame", "quaternion_order",
        "scene_metadata",
    ]
    accumulated: dict[tuple[str, str, str], dict] = {}
    if summary.exists():
        with summary.open(newline="") as previous:
            for old in csv.DictReader(previous):
                accumulated[(old["policy"], old["world"], old["seed"])] = old
    for row in rows:
        accumulated[(row["policy"], row["world"], str(row["seed"]))] = row
    with summary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(accumulated[key] for key in sorted(accumulated))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
