#!/usr/bin/env python3
"""Cold Webots transfer check for the frozen 30-level family-15 DEV bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
RESULTS = WEBOTS_DIR / "results"
OUT = RESULTS / "room-webots-transfer"
BANK = ROOT / "evidence/inputs/challenge-bank-mirrored-v1.jsonl"
FOCUSED = ROOT / "evidence/inputs/room-training/focused-selected-dev.csv"
EXPORTER = WEBOTS_DIR / "metal_scene.py"
WEBOTS = Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots")
POLICY_RELATIVE = "../assets/navigation-rooms-experimental.bin"
POLICY = ROOT / "assets/navigation-rooms-experimental.bin"
MAX_STEPS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    bank_bytes = BANK.read_bytes()
    bank_records = {}
    source_lines = {}
    for line in bank_bytes.splitlines(keepends=True):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == "dev" and record.get("family") == 15:
            if record["failure_id"] in bank_records:
                raise ValueError(f"duplicate bank id: {record['failure_id']}")
            bank_records[record["failure_id"]] = record
            source_lines[record["failure_id"]] = line
    with FOCUSED.open(newline="") as source:
        focused_rows = [row for row in csv.DictReader(source)
                        if row["split"] == "dev" and int(row["family"]) == 15]
    focused_by_id = {row["failure_id"]: row for row in focused_rows}
    if len(bank_records) != 30 or len(focused_by_id) != 30:
        raise ValueError(f"expected 30 family-15 DEV records, got bank={len(bank_records)} CSV={len(focused_by_id)}")
    if set(bank_records) != set(focused_by_id):
        raise ValueError("bank and focused evaluation IDs do not match")
    return bank_bytes, bank_records, source_lines, focused_by_id


def export_scene(failure_id: str, generated: Path, metadata_dir: Path) -> tuple[Path, Path, str]:
    generated.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    command = ["python3", str(EXPORTER), failure_id, "--bank", str(BANK),
               "--policy", POLICY_RELATIVE, "--max-steps", str(MAX_STEPS),
               "--world-dir", str(generated), "--metadata-dir", str(metadata_dir)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"metal_scene.py failed for {failure_id}: {log}")
    scene_name = f"metal_{failure_id}"
    return generated / f"{scene_name}.wbt", metadata_dir / f"{scene_name}.json", log


def trace_measurements(trace_path: Path) -> dict:
    with trace_path.open(newline="") as source:
        samples = list(csv.DictReader(source))
    if not samples:
        return {"webots_start_z_at_0_05m": "", "webots_start_vz_at_0_05mps": "",
                "webots_start_z_at_0_20m": "", "webots_start_vz_at_0_20mps": "",
                "last_trace_time_s": "", "last_trace_x_m": "", "last_trace_y_m": "",
                "last_trace_z_m": "", "failure_zone_from_last_trace": "missing_trace"}
    def nearest(time_s: float) -> dict:
        return min(samples, key=lambda row: abs(float(row["time_s"]) - time_s))
    at_005, at_020, last = nearest(0.05), nearest(0.20), samples[-1]
    x, z = float(last["x"]), float(last["z"])
    zone = "none"
    if 0.55 <= x <= 1.65:
        zone = "near_first_partition"
    elif 5.5 <= x <= 6.9:
        zone = "near_second_partition"
    elif z <= 0.55 or x < 0.0:
        zone = "near_start_floor_or_other"
    else:
        zone = "other"
    return {"webots_start_z_at_0_05m": float(at_005["z"]),
            "webots_start_vz_at_0_05mps": float(at_005["vz"]),
            "webots_start_z_at_0_20m": float(at_020["z"]),
            "webots_start_vz_at_0_20mps": float(at_020["vz"]),
            "last_trace_time_s": float(last["time_s"]),
            "last_trace_x_m": x,"last_trace_y_m": float(last["y"]),"last_trace_z_m": z,
            "failure_zone_from_last_trace": zone}


def run_episode(failure_id: str, record: dict, focused: dict, source_line: bytes,
                generated: Path, metadata_dir: Path, bank_sha: str, webots: Path) -> dict:
    case_dir = OUT / "cases" / failure_id
    case_dir.mkdir(parents=True, exist_ok=True)
    scene_name = f"metal_{failure_id}"
    base_world, base_meta, export_log = export_scene(failure_id, generated, metadata_dir)
    metadata = json.loads(base_meta.read_text())
    if metadata["source_record_sha256"] != sha256_bytes(source_line):
        raise ValueError(f"exported record hash mismatch for {failure_id}")
    if metadata["family"] != 15 or metadata["split"] != "dev":
        raise ValueError(f"exporter returned non-DEV/non-room scene for {failure_id}")
    route_world = case_dir / "route.wbt"
    shutil.copy2(base_world, route_world)
    shutil.copy2(base_meta, case_dir / "scene-metadata.json")
    (case_dir / "bank-record.json").write_bytes(source_line)
    with (case_dir / "focused-selected-dev.csv").open("w", newline="") as focused_file:
        focused_writer = csv.DictWriter(focused_file, fieldnames=list(focused.keys()), lineterminator="\n")
        focused_writer.writeheader()
        focused_writer.writerow(focused)
    (case_dir / "export.log").write_text(export_log)
    base_world.unlink(missing_ok=True)
    base_meta.unlink(missing_ok=True)

    # Controller writes shared completion files inside this RL project's own results directory.
    for stale in (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker", RESULTS / "last-run-trace.csv"):
        stale.unlink(missing_ok=True)
    temporary_world = WEBOTS_DIR / "worlds" / f".run-room-transfer-{failure_id}.wbt"
    shutil.copy2(route_world, temporary_world)
    try:
        for stale in (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker", RESULTS / "last-run-trace.csv"):
            stale.unlink(missing_ok=True)
        command = [str(webots), "--port=23456", "--minimize", "--batch", "--mode=fast", "--no-rendering",
                   "--stdout", "--stderr", str(temporary_world)]
        process = subprocess.run(command, cwd=WEBOTS_DIR, capture_output=True, text=True,
                                 timeout=90, check=False)
        (case_dir / "webots.log").write_text(process.stdout + process.stderr)
        if process.returncode != 0:
            raise RuntimeError(f"Webots exit={process.returncode} for {failure_id}; see {case_dir/'webots.log'}")
        episode_path, marker_path, trace_path = (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker",
                                                 RESULTS / "last-run-trace.csv")
        if not episode_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"Webots did not write episode receipt for {failure_id}")
        episode = json.loads(episode_path.read_text())
        shutil.copy2(episode_path, case_dir / "episode.json")
        shutil.copy2(marker_path, case_dir / "exit.marker")
        if trace_path.is_file():
            shutil.copy2(trace_path, case_dir / "trace.csv")
    finally:
        temporary_world.unlink(missing_ok=True)
        (WEBOTS_DIR / "worlds" / f".{temporary_world.stem}.wbproj").unlink(missing_ok=True)

    run_manifest = {
        "schema": "webots-room-transfer-run-v1",
        "webots_release": "R2025a",
        "failure_id": failure_id,
        "split": "dev",
        "family": 15,
        "policy": POLICY.name,
        "objective": "first goal-radius entry <0.35 m",
        "protocol": {"physics_step_ms": 1, "raptor_control_period_ms": 10,
                     "navigation_period_ms": 50, "max_steps": MAX_STEPS,
                     "budget_s": 20.0, "motor_sampling": "average", "physics_profile": "hover",
                     "speed_cap_mps": 1.5, "collision_radius_m": 0.18,
                     "depth_noise": 0.0, "dropout": 0.0,
                     "ego_sensors": "ideal GPS, InertialUnit and Gyro"},
        "source_hashes": {"bank_sha256": bank_sha,
                          "source_record_sha256": metadata["source_record_sha256"],
                          "policy_sha256": sha256(POLICY),
                          "raptor_sha256": sha256(ROOT / "assets/raptor.bin"),
                          "controller_source_sha256": sha256(WEBOTS_DIR / "controllers/raptor_webots/raptor_webots.cpp"),
                          "controller_binary_sha256": sha256(WEBOTS_DIR / "controllers/raptor_webots/raptor_webots"),
                          "vehicle_proto_sha256": sha256(WEBOTS_DIR / "protos/RaptorCrazyflie.proto"),
                          "physics_header_sha256": sha256(ROOT / "physics.hpp"),
                          "generated_world_source_sha256": metadata["world_sha256"],
                          "generated_world_sha256": sha256(route_world),
                          "metadata_sha256": sha256(case_dir / "scene-metadata.json")},
        "world_bounds_m": metadata["room_bounds_m"],
        "start_xyz_m": metadata["start_xyz_m"],
        "goal_xyz_m": metadata["goal_xyz_m"],
        "obstacle_count": len(metadata["obstacles"]),
        "actor_inputs_exclude_scene_metadata_and_witness": True,
        "files": {name: str(case_dir / name) for name in
                  ("route.wbt", "scene-metadata.json", "bank-record.json", "focused-selected-dev.csv",
                   "episode.json", "exit.marker", "trace.csv", "webots.log", "export.log")
                  if (case_dir / name).is_file()},
    }
    (case_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    trace_data=trace_measurements(case_dir/"trace.csv")
    return {"failure_id": failure_id, "family": 15, "split": "dev", "seed": record["seed"],
            "environment_index": record["environment_index"], "difficulty": record["difficulty"],
            "difficulty_band": record["difficulty_band"], "goal_xyz_m": json.dumps(record["goal"], separators=(",", ":")),
            "coordinate_transform": record.get("coordinate_transform","identity"),
            "scene_sha256": record["scene_sha256"], "focused_metal_success": focused["success"],
            "focused_metal_collision": focused["collision"], "focused_metal_timeout": focused["timeout"],
            "focused_metal_goal_time_s": focused["goal_time_s"],
            "webots_success": int(bool(episode["success"])), "webots_collision": int(bool(episode["collision"])),
            "webots_timeout": int(bool(episode["timeout"])), "webots_steps": episode["steps"],
            "webots_time_s": episode["time_s"], "webots_final_error_m": episode["final_error_m"],
            "webots_path_m": episode["path_m"], "webots_mean_speed_mps": episode["path_m"] / max(episode["time_s"], 1e-6),
            "webots_peak_speed_mps": episode["peak_speed_mps"], "webots_min_sensor_range_m": episode["min_sensor_range_m"],
            "webots_tracking_rms_mps": episode["tracking_rms_mps"],
            "webots_altitude_min_m": episode["altitude_min_m"], "webots_altitude_max_m": episode["altitude_max_m"],
            "webots_goal_entry_count": episode.get("goal_radius_entry_count", 0),
            "webots_goal_entry_first_time_s": episode.get("goal_radius_entry_first_time_s", -1),
            "world_source_sha256": metadata["world_sha256"],
            **trace_data,
            "episode_json": str(case_dir / "episode.json"), "raw_trace_csv": str(case_dir / "trace.csv"),
            "route_world": str(case_dir / "route.wbt"), "run_manifest": str(case_dir / "run_manifest.json")}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--focused-csv", type=Path, default=FOCUSED)
    parser.add_argument("--webots", type=Path, default=WEBOTS)
    parser.add_argument("--resume", action="store_true", help="skip cases with a validated saved receipt")
    args = parser.parse_args()
    bank_bytes, bank_records, source_lines, focused_by_id = load_inputs()
    bank_sha = sha256_bytes(bank_bytes)
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    generated = OUT / "_generated_worlds"
    metadata_dir = OUT / "_generated_metadata"
    rows = []
    failures = []
    ordered_ids = sorted(bank_records, key=lambda failure_id: int(bank_records[failure_id]["environment_index"]))
    for index, failure_id in enumerate(ordered_ids, 1):
        case_dir = OUT / "cases" / failure_id
        previous = case_dir / "episode.json"
        manifest_path = case_dir / "run_manifest.json"
        if args.resume and previous.is_file() and manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text())
            episode = json.loads(previous.read_text())
            if (existing_manifest.get("source_hashes", {}).get("bank_sha256") == bank_sha and
                existing_manifest.get("source_hashes", {}).get("policy_sha256") == sha256(POLICY) and
                episode.get("success") is not None):
                print(f"reuse {index}/30 {failure_id}", flush=True)
                # Keep aggregation explicit; regenerate the result row from the retained receipts below.
                # The normal fast path is used only after a previous complete matrix.
                continue
        print(f"Webots room transfer {index}/30 {failure_id}", flush=True)
        try:
            rows.append(run_episode(failure_id, bank_records[failure_id], focused_by_id[failure_id],
                                    source_lines[failure_id], generated, metadata_dir, bank_sha, args.webots))
        except Exception as error:
            failures.append({"failure_id": failure_id, "error": str(error),
                             "retained_case_dir": str(case_dir)})
            print(f"ERROR {failure_id}: {error}", flush=True)
    # Rebuild every completed row from durable episode and focused-bank receipts, including resumed cases.
    if len(rows) < 30:
        completed_ids = {row["failure_id"] for row in rows}
        for failure_id in ordered_ids:
            if failure_id in completed_ids:
                continue
            case_dir = OUT / "cases" / failure_id
            episode_path, manifest_path = case_dir / "episode.json", case_dir / "run_manifest.json"
            if not episode_path.is_file() or not manifest_path.is_file():
                continue
            record, focused = bank_records[failure_id], focused_by_id[failure_id]
            episode, metadata = json.loads(episode_path.read_text()), json.loads((case_dir / "scene-metadata.json").read_text())
            rows.append({"failure_id":failure_id,"family":15,"split":"dev","seed":record["seed"],
                "environment_index":record["environment_index"],"difficulty":record["difficulty"],
                "difficulty_band":record["difficulty_band"],"goal_xyz_m":json.dumps(record["goal"],separators=(",",":")),
                "coordinate_transform":record.get("coordinate_transform","identity"),
                "scene_sha256":record["scene_sha256"],"focused_metal_success":focused["success"],
                "focused_metal_collision":focused["collision"],"focused_metal_timeout":focused["timeout"],
                "focused_metal_goal_time_s":focused["goal_time_s"],"webots_success":int(bool(episode["success"])),
                "webots_collision":int(bool(episode["collision"])),"webots_timeout":int(bool(episode["timeout"])),
                "webots_steps":episode["steps"],"webots_time_s":episode["time_s"],"webots_final_error_m":episode["final_error_m"],
                "webots_path_m":episode["path_m"],"webots_mean_speed_mps":episode["path_m"]/max(episode["time_s"],1e-6),
                "webots_peak_speed_mps":episode["peak_speed_mps"],"webots_min_sensor_range_m":episode["min_sensor_range_m"],
                "webots_tracking_rms_mps":episode["tracking_rms_mps"],"webots_altitude_min_m":episode["altitude_min_m"],
                "webots_altitude_max_m":episode["altitude_max_m"],"webots_goal_entry_count":episode.get("goal_radius_entry_count",0),
                "webots_goal_entry_first_time_s":episode.get("goal_radius_entry_first_time_s",-1),
                **trace_measurements(case_dir/"trace.csv"),
                "world_source_sha256":metadata["world_sha256"],"episode_json":str(episode_path),
                "raw_trace_csv":str(case_dir/"trace.csv"),"route_world":str(case_dir/"route.wbt"),
                "run_manifest":str(manifest_path)})
    rows.sort(key=lambda row: int(row["environment_index"]))
    fields = list(rows[0].keys()) if rows else ["failure_id","family","split","webots_success","error"]
    summary_csv = OUT / "cases.csv"
    with summary_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    success=sum(int(row.get("webots_success",0)) for row in rows)
    collision=sum(int(row.get("webots_collision",0)) for row in rows)
    timeout=sum(int(row.get("webots_timeout",0)) for row in rows)
    metal_success=sum(int(row["focused_metal_success"]) for row in rows)
    failure_zones={}
    for row in rows:
        if int(row.get("webots_collision",0)):
            zone=row.get("failure_zone_from_last_trace","unclassified")
            failure_zones[zone]=failure_zones.get(zone,0)+1
    shared=[row for row in rows if int(row["focused_metal_success"]) and int(row["webots_success"])]
    startup_z=[row["webots_start_z_at_0_05m"] for row in rows if row.get("webots_start_z_at_0_05m")!=""]
    startup_vz=[row["webots_start_vz_at_0_05mps"] for row in rows if row.get("webots_start_vz_at_0_05mps")!=""]
    startup_z_020=[row["webots_start_z_at_0_20m"] for row in rows if row.get("webots_start_z_at_0_20m")!=""]
    startup_vz_020=[row["webots_start_vz_at_0_20mps"] for row in rows if row.get("webots_start_vz_at_0_20mps")!=""]
    def group_counts(subset):
        return {"episodes":len(subset),"webots_success":sum(int(row["webots_success"]) for row in subset),
                "webots_contacts":sum(int(row["webots_collision"]) for row in subset),
                "webots_timeouts":sum(int(row["webots_timeout"]) for row in subset),
                "focused_metal_success":sum(int(row["focused_metal_success"]) for row in subset)}
    webots_failure_ids=[row["failure_id"] for row in rows if not int(row["webots_success"])]
    contact_ids=[row["failure_id"] for row in rows if int(row["webots_collision"])]
    timeout_ids=[row["failure_id"] for row in rows if int(row["webots_timeout"])]
    transfer_fail_ids=[row["failure_id"] for row in rows if int(row["focused_metal_success"]) and not int(row["webots_success"])]
    transfer_recovery_ids=[row["failure_id"] for row in rows if not int(row["focused_metal_success"]) and int(row["webots_success"])]
    difficulty_counts={band:group_counts([row for row in rows if row["difficulty_band"]==band]) for band in ("easy","medium","hard")}
    transform_counts={label:group_counts([row for row in rows if row["coordinate_transform"]==transform])
                      for label,transform in (("identity","identity"),("mirror_y","mirror_y"))}
    summary={"schema":"webots-room-transfer-summary-v1","family":15,"split":"dev",
             "episodes_completed":len(rows),"episodes_expected":30,
             "webots_successes":success,"webots_contacts":collision,"webots_timeouts":timeout,
             "focused_metal_successes_joined":metal_success,
             "webots_transfer_success_over_metal_success":sum(int(row["webots_success"]) for row in rows if int(row["focused_metal_success"])==1),
             "metal_success_webots_failure":sum(int(row["webots_success"])==0 for row in rows if int(row["focused_metal_success"])==1),
             "webots_success_metal_failure":sum(int(row["webots_success"])==1 for row in rows if int(row["focused_metal_success"])==0),
             "webots_failure_ids":webots_failure_ids,"contact_failure_ids":contact_ids,"timeout_failure_ids":timeout_ids,
             "metal_success_webots_failure_ids":transfer_fail_ids,"webots_success_metal_failure_ids":transfer_recovery_ids,
             "difficulty_distribution":difficulty_counts,"coordinate_transform_distribution":transform_counts,
             "webots_median_success_time_s":float(statistics.median(float(row["webots_time_s"]) for row in rows if int(row["webots_success"]))) if any(int(row["webots_success"]) for row in rows) else None,
             "matched_metal_webots_success_count":len(shared),
             "matched_success_metal_mean_time_s":sum(float(row["focused_metal_goal_time_s"]) for row in shared)/len(shared) if shared else None,
             "matched_success_webots_mean_time_s":sum(float(row["webots_time_s"]) for row in shared)/len(shared) if shared else None,
             "failure_zones_from_last_sparse_trace_sample_only":failure_zones,
             "last_trace_failure_zone_limit":"Trace cadence is 100Hz for the first 0.8s and 1Hz afterward; zone labels are approximate last-sample regions, not exact contact-point identities.",
             "measured_webots_start_transient":{
                 "mean_z_at_0_05s_m":sum(float(v) for v in startup_z)/len(startup_z) if startup_z else None,
                 "mean_vz_at_0_05s_mps":sum(float(v) for v in startup_vz)/len(startup_vz) if startup_vz else None,
                 "mean_z_at_0_20s_m":sum(float(v) for v in startup_z_020)/len(startup_z_020) if startup_z_020 else None,
                 "mean_vz_at_0_20s_mps":sum(float(v) for v in startup_vz_020)/len(startup_vz_020) if startup_vz_020 else None,
                 "actuator_initialization":"Webots propeller speed starts at zero; Metal reset starts motor state at hover RPM"},
             "sensor_contract_difference":"Metal casts 16x20 analytic rays from body center at vertical slope 0.75. Webots uses a 20x16, 90-degree RangeFinder at +0.08 m body-forward, converts axial depth to ray range, resamples to slope 0.75 then 2x2-min pools; ideal GPS/IMU/Gyro and zero depth noise/dropout.",
             "matrix_failures":failures,
             "commands":{"scene_export":"python3 webots/metal_scene.py FAILURE_ID --bank evidence/inputs/challenge-bank-mirrored-v1.jsonl --policy ../assets/navigation-rooms-experimental.bin --max-steps 2000",
                         "webots":"Webots R2025a --port=23456 --minimize --batch --mode=fast --no-rendering; 1ms ODE; 20s episode budget"},
             "hashes":{"bank_sha256":bank_sha,"focused_csv_sha256":sha256(args.focused_csv),
                       "policy_sha256":sha256(POLICY),"webots_runner_sha256":sha256(Path(__file__)),
                       "scene_exporter_sha256":sha256(EXPORTER),
                       "controller_source_sha256":sha256(WEBOTS_DIR/"controllers/raptor_webots/raptor_webots.cpp"),
                       "controller_binary_sha256":sha256(WEBOTS_DIR/"controllers/raptor_webots/raptor_webots")},
             "final_split_evaluated":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("episodes_completed","webots_successes","webots_contacts","webots_timeouts","metal_success_webots_failure")},indent=2))
    print(summary_csv)
    return 0 if len(rows)==30 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
