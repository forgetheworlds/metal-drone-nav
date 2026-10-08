"""Independently grade arrival-release endpoints and all retained comparisons."""

import argparse
import tarfile
import tempfile
from pathlib import Path
import hashlib
import importlib.util
import json
import struct

ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path("/Users/muadhsambul/RL")
ARCHIVE_ROOT = None
OUT = ROOT / "results/omp-arrival-retention"
spec = importlib.util.spec_from_file_location(
    "flight_checks", ROOT / "navigation_physical_retention_results.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def resolve(path):
    path = Path(path)
    if ARCHIVE_ROOT is not None and path.is_absolute():
        if path.is_relative_to(ARCHIVE_ROOT):
            return path
        return ARCHIVE_ROOT / path.relative_to(ORIGINAL_ROOT)
    return path


def sha(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def review():
    request = json.loads((OUT / "pilot-request.json").read_text())
    accepted = json.loads((OUT / "root-accepted-request.json").read_text())
    assert accepted["request_sha256"] == sha(OUT / "pilot-request.json")
    jobs = json.loads((OUT / "root-pilot-jobs.json").read_text())
    receipts = json.loads((OUT / "root-eval-receipts.json").read_text())
    assert len(jobs) == 2 and all(j.get("exit") == 0 for j in jobs)
    assert len(receipts) == request["evaluation"]["jobs_count"] == 184
    identities = [(r["label"], r["seed"], r["panel"], r["profile"]) for r in receipts]
    expected = {
        (r["label"], r["seed"], r["panel"], r["profile"])
        for r in request["evaluation"]["jobs"]
    }
    assert len(set(identities)) == len(receipts) and set(identities) == expected
    budgets = {}
    for job in jobs:
        checkpoint = resolve(job["checkpoint"])
        assert sha(checkpoint) == job["checkpoint_sha256"]
        header = checkpoint.read_bytes()[:136]
        assert struct.unpack_from("<4I", header, 12) == (967688, 4225, 32, 512)
        assert struct.unpack_from("<I", header, 36)[0] == 256
        assert struct.unpack_from("<Q", header, 40)[0] == 32768
        exposure = Path(str(checkpoint) + ".entry-transitions.bin").read_bytes()
        assert (
            sum(struct.unpack("<" + str(len(exposure) // 8) + "Q", exposure)) == 4194304
        )
        budgets[str(job["seed"])] = {
            "transitions": 4194304,
            "optimizer_steps": 32768,
            "wall_s": job["wall_s"],
        }
    data = {}
    for receipt in receipts:
        assert receipt["exit"] == 0 and sha(receipt["csv"]) == receipt["csv_sha256"]
        assert sha(receipt["bank"]) == receipt["bank_sha256"]
        data[
            receipt["label"], receipt["seed"], receipt["panel"], receipt["profile"]
        ] = checks.flights(resolve(receipt["csv"]))
    roles = [
        "composed_parent",
        "unmasked_control",
        "unmasked_treatment",
        "masked_treatment",
    ]
    panels, per_seed, gates = {}, {}, []
    identity_fields = [
        "env",
        "scene_seed",
        "family",
        "route_class",
        "start_x",
        "start_y",
        "start_z",
        "start_yaw",
        "goal_x",
        "goal_y",
        "goal_z",
    ]
    for panel, profile in sorted({(r["panel"], r["profile"]) for r in receipts}):
        key = panel + "-" + profile
        pooled = {
            role: {field: 0 for field in ["success", "collision", "timeout"]}
            for role in roles
        }
        seed_results, arrival_delta = {}, []
        for seed in [1, 2]:
            values = {role: data[role, seed, panel, profile] for role in roles}
            for env in range(128):
                for field in identity_fields:
                    assert len({values[role][env][field] for role in roles}) == 1, (
                        key,
                        seed,
                        env,
                        field,
                    )
            summary = {role: checks.totals(rows) for role, rows in values.items()}
            for role in roles:
                for field, count in summary[role].items():
                    pooled[role][field] += count
            old, new = values["unmasked_treatment"], values["masked_treatment"]
            common = [
                e for e in old if int(old[e]["success"]) and int(new[e]["success"])
            ]
            deltas = [float(new[e]["time_s"]) - float(old[e]["time_s"]) for e in common]
            arrival_delta.extend(deltas)
            summary["wins"] = [
                e for e in old if not int(old[e]["success"]) and int(new[e]["success"])
            ]
            summary["losses"] = [
                e for e in old if int(old[e]["success"]) and not int(new[e]["success"])
            ]
            summary["shared_arrival_delta_s"] = (
                sum(deltas) / len(deltas) if deltas else None
            )
            seed_results[str(seed)] = summary
        panels[key] = {
            "tasks": 256,
            **pooled,
            "shared_arrival_delta_s": sum(arrival_delta) / len(arrival_delta)
            if arrival_delta
            else None,
            "shared_successes": len(arrival_delta),
        }
        per_seed[key] = seed_results
        parent, old, new = (
            pooled["composed_parent"],
            pooled["unmasked_treatment"],
            pooled["masked_treatment"],
        )
        if profile == "nominal" and panel in ["open", "fresh-open"]:
            gates.append(
                {
                    "gate": key + " primary arrival floor",
                    "pass": new["success"] >= 253,
                    "actual": new["success"],
                    "required": 253,
                }
            )
        if panel == "composite" and profile == "nominal":
            gates.append(
                {
                    "gate": "course absolute floor",
                    "pass": new["success"] >= 241,
                    "actual": new["success"],
                }
            )
            gates.append(
                {
                    "gate": "course retention",
                    "pass": new["success"] >= old["success"] - 3,
                    "delta": new["success"] - old["success"],
                }
            )
        if panel == "dev-c" and profile == "nominal":
            gates.append(
                {
                    "gate": "static C recovery",
                    "pass": new["success"] >= 219,
                    "actual": new["success"],
                }
            )
        if profile == "nominal" and panel not in [
            "composite",
            "fresh-course",
            "fresh-course-reflected",
        ]:
            for label, reference in [("parent", parent), ("unmasked", old)]:
                gates.append(
                    {
                        "gate": key + " retention vs " + label,
                        "pass": new["success"] >= reference["success"] - 3,
                        "delta": new["success"] - reference["success"],
                    }
                )
        for field in ["collision", "timeout"]:
            gates.append(
                {
                    "gate": key + " " + field + " vs parent",
                    "pass": new[field] <= parent[field] + 3,
                    "delta": new[field] - parent[field],
                }
            )
        if profile == "nominal" and panel in [
            "open",
            "long-open",
            "long-hallway",
            "fresh-open",
            "fresh-long-open",
            "fresh-long-hallway",
        ]:
            gates.append(
                {
                    "gate": key + " absolute floor",
                    "pass": new["success"] >= 253,
                    "actual": new["success"],
                }
            )
        if panel == "fresh-course-reflected" and profile == "combined":
            gates.append(
                {
                    "gate": key + " stress retention",
                    "pass": new["success"] >= old["success"] - 3,
                    "delta": new["success"] - old["success"],
                }
            )
        gates.append(
            {
                "gate": key + " shared arrival",
                "pass": bool(arrival_delta)
                and sum(arrival_delta) / len(arrival_delta) <= 0.5,
                "delta_s": panels[key]["shared_arrival_delta_s"],
            }
        )
    result = {
        "scope": "Exposed source development, endpoint paired masks, retained controls. No FINAL or independent transfer.",
        "budgets": budgets,
        "panels": panels,
        "per_seed": per_seed,
        "gates": gates,
        "failed_gates": [g for g in gates if not g["pass"]],
        "new_training_transitions": 8388608,
        "new_evaluation_panels": sum(not r.get("reused") for r in receipts),
        "reused_evaluation_panels": sum(bool(r.get("reused")) for r in receipts),
    }
    assert (
        result["new_evaluation_panels"] == 46
        and result["reused_evaluation_panels"] == 138
    )
    result["verdict"] = "PASS" if not result["failed_gates"] else "NOT_ADOPTED"
    (OUT / "root-results-review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "ROOT_ARRIVAL_REVIEW",
        result["verdict"],
        "failed gates",
        len(result["failed_gates"]),
    )
    return result


def main():
    global OUT, ARCHIVE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.inputs.is_dir():
        OUT = args.inputs.resolve()
        result = review()
    else:
        with tempfile.TemporaryDirectory() as directory:
            ARCHIVE_ROOT = Path(directory).resolve()
            with tarfile.open(args.inputs, "r:gz") as archive:
                for member in archive.getmembers():
                    destination = (ARCHIVE_ROOT / member.name).resolve()
                    if not destination.is_relative_to(ARCHIVE_ROOT) or not (
                        member.isfile() or member.isdir()
                    ):
                        raise ValueError("Unsafe archive member")
                archive.extractall(ARCHIVE_ROOT)
            hashes = json.loads((ARCHIVE_ROOT / "SHA256.json").read_text())
            for name, expected in hashes.items():
                assert (
                    hashlib.sha256((ARCHIVE_ROOT / name).read_bytes()).hexdigest()
                    == expected
                ), name
            OUT = ARCHIVE_ROOT / "results/omp-arrival-retention"
            result = review()
            result["verified_archive_inputs"] = len(hashes)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [
            "open-nominal",
            "fresh-open-nominal",
            "long-open-nominal",
            "composite-nominal",
            "dev-c-nominal",
            "fresh-course-reflected-combined",
        ]
        labels = [
            "Open",
            "Fresh open",
            "Long open",
            "Course",
            "Static C",
            "Reflected\ncombined stress",
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for index, (role, label) in enumerate(
            [
                ("unmasked_treatment", "Full reference loss"),
                ("masked_treatment", "Goal-distance release"),
            ]
        ):
            values = [result["panels"][key][role]["success"] for key in keys]
            ax.bar(
                [i + (index - 0.5) * 0.32 for i in range(len(keys))],
                values,
                0.32,
                label=label,
            )
        ax.set_xticks(range(len(keys)), labels)
        ax.set_ylim(0, 270)
        ax.set_ylabel("Successful goals / 256")
        ax.set_title("Releasing the teacher near goals did not improve completion")
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
