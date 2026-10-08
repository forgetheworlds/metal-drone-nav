"""Replay the frozen48-flight wide-policy Webots diagnostic without Webots."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import tarfile
import tempfile

from obstacle_motion import position

ORIGINAL_ROOT = Path("/Users/muadhsambul/RL")


def review(root):
    folder = root / "results/root-wide-native"
    request = json.loads((folder / "paired-request.json").read_text())
    rows = json.loads((folder / "paired-progress.json").read_text())

    def resolve(path):
        return root / Path(path).relative_to(ORIGINAL_ROOT)

    for path, expected in request["freeze"].items():
        if hashlib.sha256(resolve(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input hash differs: {path}")
    expected = {(c["panel"], c["index"], role)
                for c in request["cases"] for role in request["actors"]}
    if len(rows) != 48 or {(r["panel"], r["index"], r["role"]) for r in rows} != expected:
        raise ValueError("The48 predeclared cases are incomplete")
    totals, mover_checks = {}, []
    for row in rows:
        directory = resolve(row["directory"])
        process = json.loads((directory / "process.json").read_text())
        receipt = json.loads((directory / "episode.json").read_text())
        run = json.loads((directory / "run.json").read_text())
        if process["state"] != "exited" or process["exit_code"] != 0 or not run["valid"]:
            raise ValueError("Invalid actual native execution")
        if receipt != row["receipt"] or receipt["run_slug"] != directory.name:
            raise ValueError("Episode receipt identity differs")
        if process["argv"][-1] != row["world"] or not all(flag in process["argv"] for flag in ["--minimize", "--batch", "--port=23456"]):
            raise ValueError("Native project/launch contract differs")
        if not receipt["navigation_loaded"] or not receipt["raptor_loaded"] or receipt["policy_version"] != 4:
            raise ValueError("Native model did not load")
        if receipt["grading"] != "navigation_task_step" or sum(int(receipt[k]) for k in ["success", "collision", "timeout"]) != 1:
            raise ValueError("Native outcome is not exhaustive/exclusive")
        if receipt["success"] and (receipt["stable_hold_s"] < .1999 or receipt["final_error_m"] > .3501 or receipt["final_world_speed_mps"] > .5001):
            raise ValueError("Successful episode lacks stable arrival")
        nav = resolve(request["actors"][row["role"]]).read_bytes()
        if nav[:8] != b"NAVWID4\0" or struct.unpack_from("<I", nav, 16)[0] != 5120:
            raise ValueError("Wide actor architecture differs")
        source_hash = struct.unpack_from("<Q", nav, len(nav) - 8)[0]
        log = (directory / "webots.log").read_text()
        if f"source_hash={source_hash:016x}" not in log:
            raise ValueError("Loaded actor provenance differs")
        world = resolve(row["world"])
        if hashlib.sha256(world.read_bytes()).hexdigest() != row["world_sha256"] or f"policy={request['actors'][row['role']]}" not in world.read_text():
            raise ValueError("World or actor path differs")
        if row["panel"] == "moving":
            manifest = json.loads(world.with_suffix(".json").read_text())
            schedules = {s["def"]: s for s in manifest["schedules"]}
            samples = re.findall(r"BOUNDED_MOVER time_s=([\d.]+) def=(\S+) actual=([-\d.eE+,]+)", log)
            if len(samples) < 3:
                raise ValueError("No actual mover telemetry")
            seen, max_error = set(), 0.0
            for tick, name, xyz in samples:
                s = schedules[name]
                predicted = position(s["kind"], s["center"], s["size"], s["velocity"], float(tick))
                actual = [float(v) for v in xyz.split(",")]
                max_error = max(max_error, max(abs(a-b) for a, b in zip(actual, predicted)))
                seen.add(name)
            if seen != set(schedules) or max_error > .01:
                raise ValueError("Actual mover geometry/time differs")
            mover_checks.append(max_error)
        key = row["panel"] + "-" + row["role"]
        total = totals.setdefault(key, {k: 0 for k in ["tasks", "success", "collision", "timeout"]})
        total["tasks"] += 1
        for metric in ["success", "collision", "timeout"]:
            total[metric] += int(receipt[metric])
    return {"actual_flights": 48, "new_flights": 47, "reused_smoke": 1,
            "totals": totals, "max_mover_error_m": max(mover_checks),
            "scope": request["scope"], "camera_note": request["camera_note"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(args.archive, "r:gz") as archive:
            manifest = json.load(archive.extractfile("SHA256.json"))
            for name, expected in manifest.items():
                member = archive.getmember(name)
                if not member.isfile() or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe archive input")
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Input hash differs: {name}")
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        result = review(root)
        result["verified_inputs"] = len(manifest)
        print(json.dumps(result, indent=2))
