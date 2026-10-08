"""Grade actual paired pilot flights without selecting checkpoints or dropping failures."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import struct
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/omp-functional-retention"
ARCHIVE_ROOT = None
ORIGINAL_ROOT = Path("/Users/muadhsambul/RL")


def resolve(path):
    p = Path(path)
    if ARCHIVE_ROOT is not None and p.is_absolute():
        if p.is_relative_to(ARCHIVE_ROOT):
            return p
        return ARCHIVE_ROOT / p.relative_to(ORIGINAL_ROOT)
    return p


def sha(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def flights(path):
    with resolve(path).open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 128 and {int(r["env"]) for r in rows} == set(range(128))
    result = {int(r["env"]): r for r in rows}
    for r in result.values():
        assert sum(int(r[k]) for k in ["success", "collision", "timeout"]) == 1
        if int(r["success"]):
            assert (
                float(r["final_distance_m"]) <= 0.35001
                and float(r["final_speed_mps"]) <= 0.50001
            )
            assert float(r["stable_hold_s"]) >= 0.19999
    return result


def totals(rows):
    return {
        k: sum(int(r[k]) for r in rows.values())
        for k in ["success", "collision", "timeout"]
    }


def review():
    request = json.loads((OUT / "pilot-request.json").read_text())
    accepted = json.loads((OUT / "root-accepted-request.json").read_text())
    assert accepted["request_sha256"] == sha(OUT / "pilot-request.json")
    jobs = json.loads((OUT / "root-pilot-jobs.json").read_text())
    receipts = json.loads((OUT / "root-eval-receipts.json").read_text())
    assert len(jobs) == 4 and all(j.get("exit") == 0 for j in jobs)
    assert len(receipts) == request["evaluation"]["jobs_count"] == 138
    expected = {
        (r["label"], r["seed"], r["panel"], r["profile"])
        for r in request["evaluation"]["jobs"]
    }
    observed = {(r["label"], r["seed"], r["panel"], r["profile"]) for r in receipts}
    assert observed == expected and len(observed) == len(receipts)
    data = {}
    budgets = {}
    for job in jobs:
        p = resolve(job["checkpoint"])
        assert sha(p) == job["checkpoint_sha256"]
        b = p.read_bytes()[:136]
        actor, critic, horizon, n = struct.unpack_from("<4I", b, 12)
        assert (actor, critic, horizon, n) == (967688, 4225, 32, 512)
        assert (
            struct.unpack_from("<I", b, 36)[0] == 256
            and struct.unpack_from("<Q", b, 40)[0] == 32768
        )
        exposure = Path(str(p) + ".entry-transitions.bin").read_bytes()
        assert (
            sum(struct.unpack("<" + str(len(exposure) // 8) + "Q", exposure)) == 4194304
        )
        budgets[job["group"] + "-s" + str(job["seed"])] = {
            "transitions": 4194304,
            "optimizer_steps": 32768,
            "training_wall_s": job["wall_s"],
        }
    for receipt in receipts:
        assert receipt["exit"] == 0 and sha(receipt["csv"]) == receipt["csv_sha256"]
        assert sha(receipt["bank"]) == receipt["bank_sha256"]
        data[
            receipt["label"], receipt["seed"], receipt["panel"], receipt["profile"]
        ] = flights(receipt["csv"])
    per_seed = {}
    pooled = {}
    gates = []
    panels = sorted({(r["panel"], r["profile"]) for r in receipts})
    for panel, profile in panels:
        samples = {role: [] for role in ["composed_parent", "control", "treatment"]}
        time_deltas = []
        seed_results = {}
        for seed in [1, 2]:
            roles = {role: data[role, seed, panel, profile] for role in samples}
            identities = [
                "env",
                "scene_seed",
                "route_class",
                "family",
                "start_x",
                "start_y",
                "start_z",
                "start_yaw",
                "goal_x",
                "goal_y",
                "goal_z",
            ]
            for env in range(128):
                for field in identities:
                    assert len({roles[r][env][field] for r in roles}) == 1, (
                        panel,
                        seed,
                        env,
                        field,
                    )
            seed_results[str(seed)] = {
                role: totals(rows) for role, rows in roles.items()
            }
            for role in samples:
                samples[role].extend(roles[role].values())
            common = [
                e
                for e in roles["control"]
                if int(roles["composed_parent"][e]["success"])
                and int(roles["treatment"][e]["success"])
            ]
            deltas = [
                float(roles["treatment"][e]["time_s"])
                - float(roles["composed_parent"][e]["time_s"])
                for e in common
            ]
            time_deltas.extend(deltas)
            seed_results[str(seed)]["shared_parent_arrival_delta_s"] = (
                sum(deltas) / len(deltas) if deltas else None
            )
            seed_results[str(seed)]["wins_vs_control"] = [
                e
                for e in range(128)
                if not int(roles["control"][e]["success"])
                and int(roles["treatment"][e]["success"])
            ]
            seed_results[str(seed)]["losses_vs_control"] = [
                e
                for e in range(128)
                if int(roles["control"][e]["success"])
                and not int(roles["treatment"][e]["success"])
            ]
        counts = {
            role: {
                k: sum(int(r[k]) for r in rows)
                for k in ["success", "collision", "timeout"]
            }
            for role, rows in samples.items()
        }
        key = panel + "-" + profile
        per_seed[key] = seed_results
        pooled[key] = {
            "tasks": 256,
            **counts,
            "shared_parent_arrival_delta_s": sum(time_deltas) / len(time_deltas)
            if time_deltas
            else None,
            "shared_parent_successes": len(time_deltas),
        }
        p, c, t = counts["composed_parent"], counts["control"], counts["treatment"]
        if panel == "composite" and profile == "nominal":
            gates.extend(
                [
                    {
                        "gate": "primary course absolute",
                        "pass": t["success"] >= 241,
                        "actual": t["success"],
                        "required": 241,
                    },
                    {
                        "gate": "primary course gain",
                        "pass": t["success"] >= c["success"] + 10,
                        "actual_delta": t["success"] - c["success"],
                        "required_delta": 10,
                    },
                    {
                        "gate": "primary course contacts",
                        "pass": t["collision"] <= c["collision"] - 10,
                        "actual_delta": t["collision"] - c["collision"],
                        "required_delta": -10,
                    },
                ]
            )
        if profile == "nominal" and panel not in [
            "composite",
            "fresh-course",
            "fresh-course-reflected",
        ]:
            for label, reference in [("parent", p), ("control", c)]:
                gates.append(
                    {
                        "gate": key + " retention vs " + label,
                        "pass": t["success"] >= reference["success"] - 3,
                        "actual_delta": t["success"] - reference["success"],
                    }
                )
        for field in ["collision", "timeout"]:
            gates.append(
                {
                    "gate": key + " " + field + " vs parent",
                    "pass": t[field] <= p[field] + 3,
                    "actual_delta": t[field] - p[field],
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
                    "pass": t["success"] >= 253,
                    "actual": t["success"],
                    "parent": p["success"],
                    "required": 253,
                }
            )
        if panel == "fresh-course-reflected" and profile == "combined":
            gates.append(
                {
                    "gate": key + " parent retention",
                    "pass": t["success"] >= p["success"] - 3,
                    "actual_delta": t["success"] - p["success"],
                }
            )
        gates.append(
            {
                "gate": key + " shared arrival",
                "pass": bool(time_deltas)
                and sum(time_deltas) / len(time_deltas) <= 0.5,
                "actual_delta_s": pooled[key]["shared_parent_arrival_delta_s"],
            }
        )
    result = {
        "scope": "Exposed source development panels, endpoint policies, same composed warmstart. No independent transfer or FINAL.",
        "budgets": budgets,
        "panels": pooled,
        "per_seed": per_seed,
        "gates": gates,
        "failed_gates": [g for g in gates if not g["pass"]],
        "verdict": "PASS" if all(g["pass"] for g in gates) else "NOT_ADOPTED",
        "note": "Missing a gain threshold does not alone prove the loss blocks useful learning; inspect the actual paired losses and scope before another fix.",
    }
    (OUT / "root-results-review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "ROOT_PILOT_REVIEW",
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
        with tempfile.TemporaryDirectory() as temporary:
            ARCHIVE_ROOT = Path(temporary).resolve()
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
            OUT = ARCHIVE_ROOT / "results/omp-functional-retention"
            result = review()
            result["verified_archive_inputs"] = len(hashes)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [
            "composite-nominal",
            "composite-both-delay",
            "long-open-nominal",
            "fresh-course-reflected-combined",
            "dev-c-nominal",
            "open-nominal",
        ]
        labels = [
            "Course",
            "Course\n100ms delays",
            "Long open",
            "Reflected\ncombined stress",
            "Static C",
            "Short open",
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for index, label in enumerate(["composed_parent", "control", "treatment"]):
            ax.bar(
                [i + (index - 1) * 0.25 for i in range(len(keys))],
                [result["panels"][key][label]["success"] for key in keys],
                0.25,
                label={
                    "composed_parent": "Starting policy",
                    "control": "PPO",
                    "treatment": "PPO + retention",
                }[label],
            )
        ax.set_xticks(range(len(keys)), labels)
        ax.set_ylabel("Successful goals / 256")
        ax.set_ylim(0, 270)
        ax.legend()
        ax.set_title("Retaining translation preserves skills during PPO")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
