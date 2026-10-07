"""Recompute matched PPO scaling curves from process receipts and raw grades."""
import argparse
import csv
import json
import math
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ["long-open", "long-hallway", "dev-a", "dev-b", "dev-c", "open", "clutter", "composite"]
PROFILES = ["nominal", "sensor-delay", "command-delay", "both-delay", "combined"]
SAMPLE_INTERVAL = 4_194_304
FINAL_SAMPLES = 41_943_040


def read_grades(path):
    with path.open() as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 128:
        raise ValueError("Expected 128 task grades: " + str(path))
    for row in rows:
        if sum(int(row[key]) for key in ["success", "collision", "timeout"]) != 1:
            raise ValueError("Task must have exactly one terminal result")
        if not all(math.isfinite(float(row[key])) for key in ["time_s", "path_m"]):
            raise ValueError("Nonfinite flight grade")
        if int(row["success"]) and not (
            float(row["stable_hold_s"]) >= .199
            and float(row["final_distance_m"]) <= .35001
            and float(row["final_speed_mps"]) <= .50001
        ):
            raise ValueError("Success does not meet stable-arrival contract")
    return rows


def pool_grades(batches):
    rows = [row for batch in batches for row in batch]
    successes = [row for row in rows if int(row["success"])]
    return {
        "success": len(successes),
        "contacts": sum(int(row["collision"]) for row in rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "individual_success": [sum(int(row["success"]) for row in batch) for batch in batches],
        "successful_time_s": sum(float(row["time_s"]) for row in successes) / len(successes) if successes else None,
    }


def review(folder):
    jobs = json.loads((folder / "jobs.json").read_text())
    receipts = json.loads((folder / "eval-receipts.json").read_text())
    if len(jobs) != 4 or any(job["exit"] != 0 or job["optimizer_step"] != 327680 for job in jobs):
        raise ValueError("Incomplete producer batch")
    if len(receipts) != 480 or any(receipt["exit"] != 0 for receipt in receipts):
        raise ValueError("Incomplete evaluation batch")
    for job in jobs:
        path = folder / (job["arm"] + ".bin")
        # Public bundles retain the original full checkpoint header separately.
        header_path = path if path.exists() else path.with_suffix(".header.bin")
        with header_path.open("rb") as file:
            header = file.read(136)
        if len(header) != 136 or header[:7] != b"PPOFIX1":
            raise ValueError("Invalid checkpoint header")
        count = struct.unpack_from("<I", header, 24)[0]
        rollouts = struct.unpack_from("<I", header, 36)[0]
        steps = struct.unpack_from("<Q", header, 40)[0]
        if count != job["n"] or rollouts * count * 32 != FINAL_SAMPLES or steps != 327680:
            raise ValueError("Saved header disagrees with sample budget")
    curves = {}
    for count in [512, 8192]:
        curves[str(count)] = {}
        for samples in range(SAMPLE_INTERVAL, FINAL_SAMPLES + 1, SAMPLE_INTERVAL):
            pooled = {}
            for panel in PANELS:
                for profile in PROFILES if panel == "composite" else ["nominal"]:
                    batches = [read_grades(folder / "evals" / f"n{count}-s{seed}-samples{samples}-{panel}-{profile}.csv") for seed in [1, 2]]
                    pooled[panel + "/" + profile] = pool_grades(batches)
            curves[str(count)][str(samples)] = pooled
    return {"scope": "Two matched source parents, practical PPO batch regimes. Final endpoints primary; earlier peaks diagnostic, not new selected policies.",
            "curves": curves, "jobs": jobs,
            "evaluation_wall_s": sum(receipt["evaluation_wall_s"] for receipt in receipts)}


def plot(report, path):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    keys = ["open/nominal", "dev-c/nominal", "clutter/nominal", "long-open/nominal", "composite/nominal", "composite/combined"]
    titles = ["Short open", "Static C", "Clutter", "Long open", "Combined course", "Combined stress"]
    for axis, key, title in zip(axes.flat, keys, titles):
        for count, color in [("512", "#227b68"), ("8192", "#8363a2")]:
            curve = report["curves"][count]
            x = [int(samples) / 1e6 for samples in curve]
            y = [curve[samples][key]["success"] / 256 * 100 for samples in curve]
            axis.plot(x, y, marker="o", color=color, label=f"N={count}")
        axis.set(title=title, xlabel="Training samples (millions)", ylabel="Stable success (%)", ylim=(0, 100))
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    figure.suptitle("More experience: capability and retention across matched PPO regimes")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def review_fresh(folder):
    receipts = json.loads((folder / "receipts.json").read_text())
    if len(receipts) != 90 or any(receipt["exit"] != 0 for receipt in receipts):
        raise ValueError("Incomplete fresh evaluation batch")
    pooled = {}
    for group in ["warm", "n512", "n8192"]:
        pooled[group] = {}
        for panel in ["static", "clutter", "open", "long-open", "long-hallway", "course", "course-reflected"]:
            for profile in PROFILES if panel.startswith("course") else ["nominal"]:
                batches = [read_grades(folder / f"{group}-s{seed}-{panel}-{profile}.csv") for seed in [1, 2]]
                pooled[group][panel + "/" + profile] = pool_grades(batches)
    return {"scope": "Fresh seeded source DEV after final checkpoint freeze; no policy selection or training on these tasks. Same procedural families, not independent or sealed FINAL.",
            "pooled": pooled, "receipts": len(receipts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/plots/experience-scale.png"))
    parser.add_argument("--fresh", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    (args.folder / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.out)
    if args.fresh:
        fresh = review_fresh(args.fresh)
        (args.fresh / "review.json").write_text(json.dumps(fresh, indent=2) + "\n")
        print("Verified 90 fresh panels, 11,520 source flights")
    print("Verified four saved budgets and 480 evaluations")
    for count, curve in result["curves"].items():
        print(count, {key: (value["success"], value["contacts"], value["timeouts"]) for key, value in curve[str(FINAL_SAMPLES)].items()})
