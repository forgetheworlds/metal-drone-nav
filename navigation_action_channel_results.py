"""Recompute the frozen output-channel intervention from retained flight tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

PANELS = [
    ("composite", "nominal"),
    ("open", "nominal"),
    ("long-open", "nominal"),
    ("dev-b", "nominal"),
    ("fresh-course-reflected", "both-delay"),
    ("fresh-course-reflected", "combined"),
]


def flights(path):
    with path.open() as file:
        records = list(csv.DictReader(file))
    assert len(records) == 128 and {int(r["env"]) for r in records} == set(range(128))
    result = {int(r["env"]): r for r in records}
    for r in result.values():
        assert sum(int(r[k]) for k in ["success", "collision", "timeout"]) == 1
        if int(r["success"]):
            assert float(r["final_distance_m"]) <= 0.35001
            assert float(r["final_speed_mps"]) <= 0.50001
            assert float(r["stable_hold_s"]) >= 0.19999
    return result


def totals(rows):
    return {
        k: sum(int(r[k]) for r in rows.values())
        for k in ["success", "collision", "timeout"]
    }


def review(folder):
    hashes = json.loads((folder / "SHA256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, (
            name
        )
    result = {
        "verified_records": len(hashes),
        "panels": {},
        "course_interventions": {},
        "scope": "Frozen exposed development intervention and exact single-MLP composition; no new learning or independent transfer.",
    }
    for panel, profile in PANELS:
        sums = {
            stage: {k: 0 for k in ["success", "collision", "timeout"]}
            for stage in ["early", "full", "composed"]
        }
        rescues = new_losses = 0
        for seed in [1, 2]:
            e = flights(folder / f"early-s{seed}-{panel}-{profile}.csv")
            f = flights(folder / f"full-s{seed}-{panel}-{profile}.csv")
            tag = (
                f"s{seed}-early-xyz"
                if panel == "composite"
                else f"s{seed}-early-xyz-{panel}-{profile}"
            )
            c = flights(folder / (tag + "-single.csv"))
            dual = folder / (tag + ".csv")
            assert dual.read_bytes() == (folder / (tag + "-single.csv")).read_bytes()
            for stage, rows in [("early", e), ("full", f), ("composed", c)]:
                for k, v in totals(rows).items():
                    sums[stage][k] += v
            rescues += sum(
                int(f[k]["success"]) == 0 and int(c[k]["success"]) == 1 for k in f
            )
            new_losses += sum(
                int(f[k]["success"]) == 1 and int(c[k]["success"]) == 0 for k in f
            )
        result["panels"][panel + "-" + profile] = {
            "tasks": 256,
            **sums,
            "rescued_full_failures": rescues,
            "new_full_losses": new_losses,
        }
    for seed in [1, 2]:
        e = flights(folder / f"early-s{seed}-composite-nominal.csv")
        f = flights(folder / f"full-s{seed}-composite-nominal.csv")
        lost = [k for k in e if int(e[k]["success"]) and not int(f[k]["success"])]
        entry = {"lost_from_early": len(lost)}
        for component in ["early-yaw", "early-xyz"]:
            c = flights(folder / f"s{seed}-{component}.csv")
            recovered = sum(int(c[k]["success"]) for k in lost)
            new = sum(int(f[k]["success"]) and not int(c[k]["success"]) for k in f)
            entry[component] = {
                "outcomes": totals(c),
                "rescued_early_losses": recovered,
                "new_full_losses": new,
                "gate": recovered >= len(lost) / 2 and new <= 3,
            }
        result["course_interventions"][str(seed)] = entry
    receipts = json.loads((folder / "composed-receipts.json").read_text())
    assert len(receipts) == 12 and all(
        r["exit"] == 0 and r["entire_csv_identical"] for r in receipts
    )
    result["single_vs_dual_flight_parity"] = {
        "panels": 12,
        "flights": 1536,
        "entire_csv_identical": True,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory).resolve()
        with tarfile.open(args.archive, "r:gz") as archive:
            for member in archive.getmembers():
                assert (folder / member.name).resolve().is_relative_to(folder) and (
                    member.isfile() or member.isdir()
                )
            archive.extractall(folder)
        result = review(folder)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [
            "Course",
            "Open",
            "Long open",
            "Static B",
            "Reflected\n100 ms delays",
            "Reflected\ncombined stress",
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors = ["#8094a2", "#d6784c", "#258975"]
        for j, stage in enumerate(["early", "full", "composed"]):
            values = [
                result["panels"][p + "-" + profile][stage]["success"]
                for p, profile in PANELS
            ]
            ax.bar(
                [i + (j - 1) * 0.25 for i in range(6)],
                values,
                0.25,
                label=stage,
                color=colors[j],
            )
        ax.set_xticks(range(6), labels)
        ax.set_ylim(0, 270)
        ax.set_ylabel("Successful goals / 256")
        ax.legend()
        ax.set_title("One composed actor recovers retained navigation behavior")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
