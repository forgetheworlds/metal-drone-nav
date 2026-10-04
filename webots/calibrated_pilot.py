#!/usr/bin/env python3
"""Small paired Webots DEV pilot for the calibrated sensor profile.

Reuses the working exporter (`webots/metal_scene.py`): worlds are written into
`webots/worlds/` so the template's `../protos/RaptorCrazyflie.proto` resolves
against the project, `basicTimeStep` is 1 ms, and `max_steps` is the controller's
100 Hz RAPTOR step count (2000 = the 20 s DEV budget).

Four declared DEV cases are run twice each with the SAME frozen 824-input actor
weights deployed as two explicit assets:

  legacy    : NAVRAW2, policy_version=2, sensor_profile=legacy (0.75, body origin)
  calibrated: NAVCAL3, policy_version=3, sensor_profile=native (0.8, mount 0.08)

No target-side training or tuning. The shared GPU lock is an absolute-path
fcntl lock whose descriptor is non-inheritable; each Webots process runs in its
own session and is stopped only through its owned process group.
"""
import argparse
import fcntl
import json
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
import metal_scene  # noqa: E402  (reuse the working exporter)

WORLDS_DIR = WEBOTS_DIR / "worlds"
RESULTS_DIR = WEBOTS_DIR / "results"
WEBOTS = pathlib.Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots")
GPU_LOCK = pathlib.Path("/Users/muadhsambul/RL/results/metal-training.lock")
PORT = "23456"
MAX_STEPS = 2000          # 100 Hz RAPTOR steps = 20 s, matching the source DEV budget
RUN_TIMEOUT_S = 240
LOCK_WAIT_S = 300
INVALID_MARKERS = (
    "EXTERNPROTO", "could not be found", "cannot be found", "No such file",
    "does not exist", "Unable to load", "unknown controller",
    "controller not found", "segmentation fault",
    "controller exited with status", "policy declares sensor profile",
)

# Declared DEV cases: first two connected-rooms levels, one vertical-choice,
# one hallway-corner. Selected by environment_index before any result is seen.
CASES = [("15", 0), ("15", 1), ("16", 0), ("14", 0)]


def load_case(bank_path, family, environment_index):
    for line in bank_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec.get("split") == "dev" and int(rec["family"]) == int(family)
                and int(rec["environment_index"]) == environment_index):
            return rec
    raise SystemExit(f"no dev level family={family} env={environment_index}")


def read_asset_header(path):
    data = pathlib.Path(path).read_bytes()
    magic = data[:8]
    if magic not in (b"NAVRAW2\0", b"NAVCAL3\0"):
        raise SystemExit(f"{path}: unexpected asset magic {magic!r}")
    version, obs, hidden, action, mode, weights = struct.unpack_from("<6I", data, 8)
    rows, cols = struct.unpack_from("<2I", data, 36)
    if obs != 824 or weights != 53064 or hidden != 64 or action != 4 or mode != 17:
        raise SystemExit(f"{path}: unexpected deployed actor dimensions")
    if rows != 16 or cols != 20:
        raise SystemExit(f"{path}: unexpected sensor grid {rows}x{cols}")
    profile = None
    if magic == b"NAVCAL3\0":
        pid, tan_h, tan_v, mount = struct.unpack_from("<I3f", data, 64)
        if pid != 2 or abs(tan_h - 1.0) > 1e-6 or abs(tan_v - 0.8) > 1e-6 or abs(mount - 0.08) > 1e-6:
            raise SystemExit(f"{path}: calibrated contract is not the native profile")
        profile = {"profile_id": pid, "tan_h": tan_h, "tan_v": tan_v, "mount_x_m": mount}
    return {"magic": magic.rstrip(b"\0").decode(), "version": version,
            "observation_count": obs, "weight_count": weights, "contract": profile}


def write_world(record, obstacles, policy_abs, policy_version, sensor_profile):
    if ";" in policy_abs or '"' in policy_abs:
        raise SystemExit("policy path contains customData delimiters")
    text = metal_scene.scene_world(record, obstacles, policy_abs, MAX_STEPS)
    extra = f"policy_version={policy_version};sensor_profile={sensor_profile};profile=hover"
    text, n = re.subn(r'(customData "[^"]*)(")', lambda m: m.group(1) + ";" + extra + m.group(2),
                      text, count=1)
    if n != 1:
        raise SystemExit("failed to extend customData")
    slug = record["failure_id"].replace("/", "_")
    path = WORLDS_DIR / f"calibrated-pilot-{slug}.{sensor_profile}.wbt"
    path.write_text(text)
    return path


def acquire_lock(fd, deadline):
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.5)


RECEIPT = RESULTS_DIR / "last-run.json"
EXIT_MARKER = RESULTS_DIR / "last-run-exit.marker"


def run_world(world, log_path, receipt_path):
    # Clear stale receipts so a fresh one proves this run produced a result.
    for stale in (RECEIPT, EXIT_MARKER):
        stale.unlink(missing_ok=True)
    cmd = [str(WEBOTS), "--minimize", "--batch", "--mode=fast", "--no-rendering",
           f"--port={PORT}", "--stdout", "--stderr", str(world)]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True, cwd=str(WORKTREE))
    status = {"invalid": False, "reason": ""}
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            break
        if time.monotonic() > deadline:
            status.update(invalid=True, reason="wall-clock timeout")
            break
        # Stop an owned run immediately if the controller aborts or a world
        # path fails, instead of waiting for the wall-clock bound.
        try:
            seen = pathlib.Path(log_path).read_text(errors="replace")
        except OSError:
            seen = ""
        for marker in INVALID_MARKERS:
            if marker in seen:
                status.update(invalid=True, reason=f"log marker: {marker}")
                break
        if status["invalid"]:
            break
        time.sleep(0.25)
    log.flush()
    text = pathlib.Path(log_path).read_text(errors="replace")
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
    log.close()
    if not status["invalid"]:
        for marker in INVALID_MARKERS:
            if marker in text:
                status.update(invalid=True, reason=f"log marker: {marker}")
                break
    receipt = {}
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        pathlib.Path(receipt_path).write_text(json.dumps(receipt, indent=2))
    if not status["invalid"] and not receipt:
        status.update(invalid=True, reason="no controller receipt (last-run.json)")
    if not status["invalid"] and not receipt.get("navigation_loaded", False):
        status.update(invalid=True, reason="receipt says the navigation policy did not load")
    if not status["invalid"] and "WEBOTS_SENSOR_PROFILE " not in text:
        status.update(invalid=True, reason="controller did not report its sensor profile")
    return proc.returncode, text, receipt, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", type=pathlib.Path)
    ap.add_argument("output_dir", type=pathlib.Path)
    ap.add_argument("--legacy-asset", required=True)
    ap.add_argument("--native-asset", required=True)
    args = ap.parse_args()

    legacy_asset = pathlib.Path(args.legacy_asset).resolve()
    native_asset = pathlib.Path(args.native_asset).resolve()
    legacy_header = read_asset_header(legacy_asset)
    native_header = read_asset_header(native_asset)
    if legacy_header["magic"] != "NAVRAW2" or native_header["magic"] != "NAVCAL3":
        raise SystemExit("assets must be NAVRAW2 (legacy) and NAVCAL3 (native)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"legacy_asset={legacy_asset} {legacy_header}", flush=True)
    print(f"native_asset={native_asset} {native_header}", flush=True)

    lock_fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    os.set_inheritable(lock_fd, False)
    if not acquire_lock(lock_fd, time.monotonic() + LOCK_WAIT_S):
        os.close(lock_fd)
        raise SystemExit("could not acquire the shared GPU lock within the bounded wait")
    rows = []
    try:
        for family, env_index in CASES:
            record = load_case(args.bank, family, env_index)
            obstacles = metal_scene.validate_record(record)
            for profile, asset, version in (("legacy", legacy_asset, 2), ("native", native_asset, 3)):
                world = write_world(record, obstacles, str(asset), version, profile)
                slug = f"{record['failure_id']}.{profile}"
                log_path = args.output_dir / f"{slug}.log"
                receipt_path = args.output_dir / f"{slug}.last-run.json"
                code, text, receipt, status = run_world(world, log_path, receipt_path)
                if not status["invalid"]:
                    terminal = sum(int(receipt.get(k, 0)) for k in ("success", "collision", "timeout"))
                    if terminal != 1:
                        status.update(invalid=True, reason="receipt is not a single terminal outcome")
                    elif receipt.get("sensor_profile") != profile:
                        status.update(invalid=True, reason="receipt sensor_profile does not match the run")
                row = {
                    "failure_id": record["failure_id"], "family": int(family),
                    "family_name": record["family_name"], "environment_index": env_index,
                    "profile": profile, "policy_version": version,
                    "asset": str(asset), "asset_sha256": metal_scene.sha256(asset.read_bytes()),
                    "world": str(world), "world_sha256": metal_scene.sha256(world.read_bytes()),
                    "log": str(log_path), "receipt": str(receipt_path), "exit_code": code,
                    "valid": not status["invalid"], "invalid_reason": status["reason"],
                    "success": int(receipt.get("success", 0)), "collision": int(receipt.get("collision", 0)),
                    "timeout": int(receipt.get("timeout", 0)), "steps": int(receipt.get("steps", 0)),
                    "time_s": float(receipt.get("time_s", 0.0)), "path_m": float(receipt.get("path_m", 0.0)),
                    "final_error_m": float(receipt.get("final_error_m", 0.0)),
                    "min_sensor_range_m": float(receipt.get("min_sensor_range_m", 0.0)),
                    "peak_speed_mps": float(receipt.get("peak_speed_mps", 0.0)),
                    "tracking_rms_mps": float(receipt.get("tracking_rms_mps", 0.0)),
                    "sensor_profile_reported": receipt.get("sensor_profile", ""),
                    "goal_objective": receipt.get("goal_objective", ""),
                    "raptor_loaded": bool(receipt.get("raptor_loaded", False)),
                    "motor_sampling": "average",
                    "physics_profile": "hover",
                    "max_steps": MAX_STEPS,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        os.close(lock_fd)
    out_path = args.output_dir / "pilot-results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    valid = [r for r in rows if r["valid"]]
    print(f"pilot rows={len(rows)} valid={len(valid)} output={out_path}")


if __name__ == "__main__":
    main()
