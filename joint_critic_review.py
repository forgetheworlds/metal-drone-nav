"""Recompute the mixed static/moving critic comparison from its evidence archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

SPLITS = ["dev-dyn", "dev-dyn2", "dev-a", "dev-b", "dev-c", "open", "clutter"]
STATIC_BLOCKED = ["dev-a", "dev-b", "dev-c", "clutter"]


def summarize(rows):
    success = [row for row in rows if int(row["success"])]
    return {"n": len(rows), "S": len(success),
            "C": sum(int(row["collision"]) for row in rows),
            "T": sum(int(row["timeout"]) for row in rows),
            "success_time": sum(float(row["time_s"]) for row in success) / len(success) if success else None}


def review(archive):
    with tarfile.open(archive) as package:
        hashes = json.load(package.extractfile("SHA256.json"))
        contents = {}
        for name, digest in hashes.items():
            data = package.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == digest, name
            contents[name] = data
    jobs = json.loads(contents["provenance/jobs.json"])
    assert len(jobs) == 4
    for job in jobs:
        assert job["exit"] == 0 and job["rollouts"] == 5000 and job["optimizer_step"] == 160000
        header = contents["runs/" + job["arm"] + ".bin"]
        assert struct.unpack_from("<I", header, 36)[0] == 5000
        assert struct.unpack_from("<Q", header, 40)[0] == 160000
    receipts = json.loads(contents["provenance/eval-receipts.json"])
    assert len(receipts) == 56 and all(receipt["exit"] == 0 for receipt in receipts)
    result = {"input_files": len(hashes)}
    blocked = {arm: {"n": 0, "S": 0} for arm in ["C", "T"]}
    for best in [False, True]:
        stage = "selected_best" if best else "final5000"
        result[stage] = {}
        for split in SPLITS:
            result[stage][split] = {}
            for arm in ["C", "T"]:
                rows = []
                for seed in [1, 2]:
                    label = f"joint-{arm}-s{seed}" + ("-best" if best else "")
                    batch = list(csv.DictReader(io.StringIO(contents[f"evals/{label}-{split}.csv"].decode())))
                    assert len(batch) == 128 and len({int(row["env"]) for row in batch}) == 128
                    for row in batch:
                        assert sum(int(row[field]) for field in ["success", "collision", "timeout"]) == 1
                    rows.extend(batch)
                    if not best and split in STATIC_BLOCKED:
                        bank = contents[f"banks/{split}-static.bin"]
                        period = struct.unpack_from("<I", bank, 12)[0]
                        for row in batch:
                            offset = 88 + int(row["env"]) * period * 756 + 748
                            if struct.unpack_from("<I", bank, offset)[0] == 1:
                                blocked[arm]["n"] += 1
                                blocked[arm]["S"] += int(row["success"])
                result[stage][split][arm] = summarize(rows)
    final = result["final5000"]
    result["blocked"] = blocked
    result["gates"] = {
        "fresh_dynamic_retention": final["dev-dyn2"]["T"]["S"] >= final["dev-dyn2"]["C"]["S"] - 3,
        "fresh_dynamic_safety": final["dev-dyn2"]["T"]["C"] <= final["dev-dyn2"]["C"]["C"] + 3,
        "static_retention": all(final[split]["T"]["S"] >= final[split]["C"]["S"] - 3 for split in STATIC_BLOCKED),
        "open_floor": final["open"]["T"]["S"] >= 254,
        "blocked_gain": blocked["T"]["S"] >= blocked["C"]["S"] + 6,
        "selector_retention": final["dev-dyn"]["T"]["S"] >= final["dev-dyn"]["C"]["S"] - 3,
        "arrival": all(final[split]["T"]["success_time"] <= final[split]["C"]["success_time"] + .5 for split in SPLITS),
    }
    result["adopted"] = all(result["gates"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review(args.archive)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
