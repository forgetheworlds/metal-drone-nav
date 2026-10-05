"""Recompute the completed long-goal coverage experiment and its frozen gates."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

from joint_critic_review import summarize

SPLITS = ["long-open", "long-hallway", "dev-a", "dev-b", "dev-c", "open", "clutter"]


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
        saved = content["runs/" + job["arm"] + ".bin"]
        assert struct.unpack_from("<I", saved, 36)[0] == 10000
        assert struct.unpack_from("<Q", saved, 40)[0] == 320000
    receipts = json.loads(content["provenance/eval-receipts.json"])
    assert len(receipts) == 61 and all(x["exit"] == 0 for x in receipts)

    def rows(name):
        values = list(csv.DictReader(io.StringIO(content[name].decode())))
        assert len(values) == 128 and len({int(r["env"]) for r in values}) == 128
        for r in values:
            assert sum(int(r[k]) for k in ["success", "collision", "timeout"]) == 1
            if int(r["success"]):
                assert float(r["stable_hold_s"]) >= .199
                assert float(r["final_distance_m"]) <= .35001 and float(r["final_speed_mps"]) <= .50001
        return values

    result = {"input_files": len(hashes), "trials_per_arm_per_panel": 256,
              "unique_task_instances_per_panel": 128, "stages": {}}
    for best in [False, True]:
        stage = "selected_best" if best else "final10000"
        result["stages"][stage] = {}
        for split in SPLITS:
            arms, paired = {}, {}
            for arm in ["C", "T"]:
                batch = []
                paired[arm] = {}
                for seed in [1, 2]:
                    label = f"distance-{arm}-s{seed}" + ("-best" if best else "")
                    flight = rows(f"evals/{label}-{split}.csv")
                    batch.extend(flight)
                    paired[arm].update({(seed, int(r["env"])): r for r in flight})
                arms[arm] = summarize(batch)
            arms["paired_wins"] = sum(int(t["success"]) and not int(paired["C"][key]["success"])
                                      for key, t in paired["T"].items())
            arms["paired_losses"] = sum(int(c["success"]) and not int(paired["T"][key]["success"])
                                        for key, c in paired["C"].items())
            result["stages"][stage][split] = arms
    final = result["stages"]["final10000"]
    long_checks, seed_long = {}, {i: {arm: 0 for arm in ["C", "T"]} for i in [1, 2]}
    for split in ["long-open", "long-hallway"]:
        kind = split.removeprefix("long-")
        bc = rows(f"preflight/dev-{kind}-bc.csv")
        timing = []
        for seed in [1, 2]:
            for arm in ["C", "T"]:
                flight = rows(f"evals/distance-{arm}-s{seed}-{split}.csv")
                seed_long[seed][arm] += sum(int(r["success"]) for r in flight)
                if arm == "T":
                    timing.extend(float(t["time_s"]) - float(b["time_s"])
                                  for t, b in zip(flight, bc) if int(t["success"]) and int(b["success"]))
        long_checks[split] = {"bc_per128": summarize(bc), "common_success_arrival_delta_vs_bc_s": sum(timing) / len(timing)}
    bc_contacts = sum(v["bc_per128"]["C"] * 2 for v in long_checks.values())
    bc_timeouts = sum(v["bc_per128"]["T"] * 2 for v in long_checks.values())
    result["long_reference"] = long_checks
    result["seed_long_success"] = seed_long
    result["gates"] = {
        "long_gain": sum(final[s]["T"]["S"] - final[s]["C"]["S"] for s in long_checks) >= 12,
        "both_seeds_retain_long": all(v["T"] >= v["C"] for v in seed_long.values()),
        "long_bc_retention": all(final[s]["T"]["S"] >= v["bc_per128"]["S"] * 2 - 3 for s, v in long_checks.items()),
        "long_contacts": sum(final[s]["T"]["C"] for s in long_checks) <= bc_contacts + 4,
        "long_timeouts": sum(final[s]["T"]["T"] for s in long_checks) <= bc_timeouts + 4,
        "static_retention": all(final[s]["T"]["S"] >= final[s]["C"]["S"] - 3 for s in ["dev-a", "dev-b", "dev-c", "clutter"]),
        "open_floor": final["open"]["T"]["S"] >= 253,
        "long_arrival": all(v["common_success_arrival_delta_vs_bc_s"] <= .5 for v in long_checks.values()),
    }
    result["adopted"] = all(result["gates"].values())
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
