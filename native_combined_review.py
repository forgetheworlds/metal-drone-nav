"""Audit the exact 96-flight frozen combined-navigation Webots comparison."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

from obstacle_motion import position


def review(path):
    with tarfile.open(path) as archive:
        hashes = json.load(archive.extractfile("SHA256.json"))
        data = {}
        for name, digest in hashes.items():
            contents = archive.extractfile(name).read()
            assert hashlib.sha256(contents).hexdigest() == digest, name
            data[name] = contents
    rows = json.loads(data["receipts.json"])
    actors = json.loads(data["actors.json"])
    assert len(rows) == 96 and len({(r["index"], r["actor"]) for r in rows}) == 96
    audits = []
    for row in rows:
        slug = row["slug"];prefix = f"flights/{slug}"
        receipt = json.loads(data[prefix + "/episode.json"])
        manifest = json.loads(data[prefix + "/manifest.json"])
        assert row["run"]["valid"] and row["run"]["exit_code"] == 0, slug
        assert receipt == row["run"]["receipt"], slug
        assert receipt["navigation_loaded"] and receipt["raptor_loaded"], slug
        assert receipt["grading"] == "navigation_task_step" and receipt["goal_objective"] == "hold", slug
        assert sum(bool(receipt[x]) for x in ["success", "collision", "timeout"]) == 1, slug
        assert manifest["world_sha256"] == hashes[prefix + "/world.wbt"] == row["world_sha256"], slug
        assert manifest["policy_sha256"] == actors[row["actor"]]["nav_sha256"] == row["nav_sha256"], slug
        assert row["bank_sha256"] == hashes["bank.bin"] == manifest["bank_sha256"], slug
        assert manifest["motion_module_sha256"] == hashes["source/obstacle_motion.py"], slug
        assert hashes[f"nav/{row['actor']}.nav"] == row["nav_sha256"], slug
        log = data[prefix + "/webots.log"].decode(errors="replace")
        assert "WEBOTS_MODELS raptor=loaded" in log and "WEBOTS_START phase=navigation" in log, slug
        assert "wrong type or length" not in log and "Traceback" not in log, slug
        marker = data[prefix + "/exit.marker"].decode()
        assert "completed=1" in marker, slug
        trace = list(csv.DictReader(io.StringIO(data[prefix + "/trace.csv"].decode())))
        assert trace and len(trace) >= 2, slug
        if receipt["success"]:
            assert receipt["goal_dwell_s"] >= .199 and receipt["final_error_m"] <= .35001, slug
            assert receipt["final_world_speed_mps"] <= .50001, slug
        schedules = {m["def"]: m for m in manifest["schedules"]}
        samples, error = 0, 0.0
        for clock, name, xyz in re.findall(r"BOUNDED_MOVER time_s=([0-9.]+) def=(\S+) actual=([-0-9.eE+,]+)", log):
            schedule = schedules[name]
            expected = position(schedule["kind"], schedule["center"], schedule["size"], schedule["velocity"], float(clock))
            actual = list(map(float, xyz.split(",")))
            error = max(error, math.dist(expected, actual));samples += 1
        if schedules:
            assert samples >= 3 and error <= .01, (slug, samples, error)
        audits.append({"slug": slug, "index": row["index"], "actor": row["actor"],
                       "mover_samples": samples, "max_mover_error_m": error,
                       "success": receipt["success"], "contact": receipt["collision"],
                       "timeout": receipt["timeout"], "time_s": receipt["time_s"],
                       "path_m": receipt["path_m"], "initial_distance_m": receipt["initial_distance_m"]})
    result = {"valid_flights": 96, "input_files": len(hashes), "actors": {}, "paired": {}, "audits": audits}
    for actor in actors:
        subset = [r for r in audits if r["actor"] == actor];assert len(subset) == 24
        good = [r for r in subset if r["success"]]
        result["actors"][actor] = {"n": 24, "successes": len(good), "contacts": sum(r["contact"] for r in subset),
                                    "timeouts": sum(r["timeout"] for r in subset),
                                    "successful_time_s": statistics.mean(r["time_s"] for r in good),
                                    "successful_path_ratio": statistics.mean(r["path_m"] / r["initial_distance_m"] for r in good)}
    for seed in [1, 2]:
        parent = {r["index"]: r for r in audits if r["actor"] == f"parent-s{seed}"}
        candidate = {r["index"]: r for r in audits if r["actor"] == f"combined-s{seed}"}
        assert parent.keys() == candidate.keys()
        common = [i for i in parent if parent[i]["success"] and candidate[i]["success"]]
        result["paired"][str(seed)] = {
            "wins": sum(candidate[i]["success"] and not parent[i]["success"] for i in parent),
            "losses": sum(parent[i]["success"] and not candidate[i]["success"] for i in parent),
            "common_success_time_delta_s": statistics.mean(candidate[i]["time_s"] - parent[i]["time_s"] for i in common),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path);parser.add_argument("--output", type=Path)
    args = parser.parse_args();text = json.dumps(review(args.archive), indent=2) + "\n"
    if args.output:args.output.write_text(text)
    print(text)
