"""Independently replay the frozen mean-versus-sampling diagnosis."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile


def review(archive_path):
    prefix = "results/root-open-mean-diagnostic/"
    root = "/Users/muadhsambul/RL/"
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest = json.load(archive.extractfile("SHA256.json"))
        for name, expected in manifest.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError("Diagnostic input hash differs")
        request = json.load(archive.extractfile(prefix + "request.json"))
        jobs = json.load(archive.extractfile(prefix + "jobs.json"))
        if len(jobs) != 10 or any(j.get("exit") != 0 for j in jobs):
            raise ValueError("All ten actual executions are required")
        tables, totals = {}, {}
        for job in jobs:
            path = job["argv"][6].removeprefix(root)
            data = archive.extractfile(path).read()
            if hashlib.sha256(data).hexdigest() != job["csv_sha256"]:
                raise ValueError("Scored flight hash differs")
            rows = list(csv.DictReader(io.StringIO(data.decode())))
            if len(rows) != 128 or {int(r["env"]) for r in rows} != set(range(128)):
                raise ValueError("128 complete task identities required")
            for row in rows:
                if sum(int(row[k]) for k in ["success", "collision", "timeout"]) != 1:
                    raise ValueError("Nonexclusive episode outcome")
                if int(row["success"]) and (float(row["stable_hold_s"]) < .19999 or float(row["final_speed_mps"]) > .50001 or float(row["final_distance_m"]) > .35001):
                    raise ValueError("Missing stable arrival")
            tables[job["tag"]] = {int(r["env"]): r for r in rows}
            totals[job["tag"]] = {k: sum(int(r[k]) for r in rows) for k in ["success", "collision", "timeout"]}
        reference = request["freeze"]
        for path, expected in reference.items():
            if manifest[path.removeprefix(root)] != expected:
                raise ValueError("Frozen source input differs")
        baseline = tables["mean17"]
        for env, row in tables["mean22-off"].items():
            for key, value in row.items():
                if key not in ["mode", "split"] and value != baseline[env][key]:
                    raise ValueError("Noise-off mode mapping differs")
        failed = [env for env, row in baseline.items() if not int(row["success"])]
        if failed != request["failed_mean_tasks"]:
            raise ValueError("Mechanically selected failure set differs")
        seeds = [f"stoch-{seed}" for seed in request["seeds"]]
        failures = [{"env": env, "successes": sum(int(tables[s][env]["success"]) for s in seeds),
                     "contacts": sum(int(tables[s][env]["collision"]) for s in seeds), "samples": 8}
                    for env in failed]
        return {"actual_flights": 1280, "verified_inputs": len(manifest),
                "totals": totals, "failed_mean_tasks": failures,
                "support_count": sum(r["successes"] >= 4 for r in failures),
                "scope": request["scope"], "no_new_learning": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.archive), indent=2))
