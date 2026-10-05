"""Recompute the completed parameter-anchor comparison from hashed flight records."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

from joint_critic_review import summarize

SPLITS = ["long-open", "long-hallway", "dev-a", "dev-b", "dev-c", "open", "clutter", "composite-dev"]


def review(archive):
    with tarfile.open(archive) as package:
        hashes = json.load(package.extractfile("SHA256.json"))
        content = {}
        for name, expected in hashes.items():
            data = package.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == expected, name
            content[name] = data
    jobs = json.loads(content["provenance/jobs.json"])
    assert len(jobs) == 4
    for job in jobs:
        assert job["exit"] == 0 and job["rollouts"] == 10000 and job["optimizer_step"] == 320000
        checkpoint = content[f"runs/{job['arm']}.bin"]
        assert struct.unpack_from("<I", checkpoint, 36)[0] == 10000
        assert struct.unpack_from("<Q", checkpoint, 40)[0] == 320000
    receipts = json.loads(content["provenance/eval-receipts.json"])
    assert len(receipts) == 64 and all(x["exit"] == 0 for x in receipts)
    result = {"input_files": len(hashes), "stages": {}, "individual_seeds": {}}
    for best in [False, True]:
        stage = "selected_best" if best else "final10000"
        result["stages"][stage] = {}
        result["individual_seeds"][stage] = {}
        for split in SPLITS:
            summary, paired = {}, {}
            individual = result["individual_seeds"][stage][split] = {}
            for arm in ["C", "T"]:
                rows = [];paired[arm] = {}
                for seed in [1, 2]:
                    label = f"anchor-{arm}-s{seed}" + ("-best" if best else "")
                    batch = list(csv.DictReader(io.StringIO(content[f"evals/{label}-{split}.csv"].decode())))
                    assert len(batch) == 128 and len({int(r["env"]) for r in batch}) == 128
                    for row in batch:
                        assert sum(int(row[k]) for k in ["success", "collision", "timeout"]) == 1
                        if int(row["success"]):
                            assert float(row["stable_hold_s"]) >= .199 and float(row["final_distance_m"]) <= .35001
                            assert float(row["final_speed_mps"]) <= .50001
                    individual[f"{arm}-s{seed}"] = summarize(batch)
                    paired[arm].update({(seed, int(r["env"])): r for r in batch})
                    rows.extend(batch)
                summary[arm] = summarize(rows)
            delays = [float(t["time_s"]) - float(paired["C"][key]["time_s"])
                      for key, t in paired["T"].items() if int(t["success"]) and int(paired["C"][key]["success"])]
            summary["paired"] = {
                "wins": sum(int(t["success"]) and not int(paired["C"][key]["success"]) for key, t in paired["T"].items()),
                "losses": sum(int(c["success"]) and not int(paired["T"][key]["success"]) for key, c in paired["C"].items()),
                "common_success_delay_s": sum(delays) / len(delays),
            }
            result["stages"][stage][split] = summary
    final = result["stages"]["final10000"]
    seeds = result["individual_seeds"]["final10000"]
    result["gates"] = {
        "long_open_floor": final["long-open"]["T"]["S"] >= 253,
        "long_hallway_floor": final["long-hallway"]["T"]["S"] >= 253,
        "short_open_floor": final["open"]["T"]["S"] >= 253,
        "static_retention": all(final[s]["T"]["S"] >= final[s]["C"]["S"] - 3 for s in ["dev-a", "dev-b", "dev-c", "clutter"]),
        "absolute_static_B": final["dev-b"]["T"]["S"] >= 223,
        "absolute_clutter": final["clutter"]["T"]["S"] >= 219,
        "contacts": all(final[s]["T"]["C"] <= final[s]["C"]["C"] + 3 for s in SPLITS if s != "composite-dev"),
        "timeouts": all(final[s]["T"]["T"] <= final[s]["C"]["T"] + 3 for s in SPLITS if s != "composite-dev"),
        "speed": all(final[s]["paired"]["common_success_delay_s"] <= .5 for s in SPLITS if s != "composite-dev"),
        "per_seed_long_retention": all(sum(seeds[s][f"T-s{i}"]["S"] - seeds[s][f"C-s{i}"]["S"]
                                             for s in ["long-open", "long-hallway"]) >= -1 for i in [1, 2]),
    }
    result["adopted"] = all(result["gates"].values())
    result["gradient_diagnostics"] = {}
    for seed in [1, 2]:
        for arm in ["C", "T"]:
            label = f"anchor-{arm}-s{seed}"
            rows = list(csv.DictReader(io.StringIO(content[f"runs/{label}.bin.diag.csv"].decode())))[-100:]
            assert len(rows) == 100
            result["gradient_diagnostics"][label] = {key: sum(float(r[key]) for r in rows) / 100
                                                     for key in ["ppo_actor_grad_norm_mean", "anchor_grad_norm_mean", "actor_grad_norm_mean", "actor_drift_l2"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(args.archive), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
