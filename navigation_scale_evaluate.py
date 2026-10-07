"""Score hash-frozen policies on the fresh scale suite without training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_suite(suite, output, evaluator, policies):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((suite / "manifest.json").read_text())
    root = Path(__file__).resolve().parent
    frozen = {str(evaluator): digest(evaluator)}
    # The executable compiles these shaders at runtime.
    for name in ["sim.metal", "ppo.metal", "memory.metal"]:
        path = root / name
        frozen[str(path)] = digest(path)
    for policy in policies.values():
        frozen[str(policy)] = digest(policy)
    for bank in manifest["banks"].values():
        path = suite / bank["path"]
        if digest(path) != bank["sha256"]:
            raise ValueError("Evaluation bank changed after generation")
        frozen[str(path)] = digest(path)
    (output / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    receipts = []
    for actor, policy in policies.items():
        for panel, bank in manifest["banks"].items():
            profiles = ["nominal", "sensor-delay", "command-delay", "both-delay", "combined"] if panel.startswith("course") else ["nominal"]
            for profile in profiles:
                for path, expected in frozen.items():
                    if digest(Path(path)) != expected:
                        raise ValueError("Frozen input changed: " + path)
                label = f"{actor}-{panel}-{profile}"
                command = ["python3", str(root / "run_locked.py"), "--", str(evaluator),
                           str(policy), str(suite / bank["path"]), str(output / (label + ".csv")), profile]
                started = time.monotonic()
                with (output / (label + ".log")).open("w") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                receipts.append({"actor": actor, "panel": panel, "profile": profile,
                                 "argv": command, "exit": code, "wall_s": time.monotonic() - started})
                (output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
                if code:
                    raise RuntimeError("Evaluation failed; inspect retained log: " + label)
        print(actor + " fresh evaluation complete", flush=True)
    print(f"Completed {len(receipts)} panels", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True, help="LABEL=CHECKPOINT")
    args = parser.parse_args()
    policies = {}
    for value in args.policy:
        label, filename = value.split("=", 1)
        if label in policies or "/" in label:
            raise ValueError("Policy labels must be unique filenames")
        policies[label] = Path(filename).resolve()
    evaluate_suite(args.suite.resolve(), args.output.resolve(), args.evaluator.resolve(), policies)
