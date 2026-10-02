#!/usr/bin/env python3
"""Compare independent ODE and L2F dynamics under common motor commands."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from benchmark import DEFAULT_WEBOTS, ROOT, with_custom_data


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-sampling", choices=("end", "average"), default="end")
    parser.add_argument("--profiles", default="hover,collective,roll,pitch,yaw")
    parser.add_argument("--physics-step-ms", type=int, choices=(1,2,5,10), default=10)
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    args = parser.parse_args()
    base = (ROOT / "worlds/a_to_b.wbt").read_text()
    base = re.sub(r"basicTimeStep\s+10", f"basicTimeStep {args.physics_step_ms}", base, count=1)
    arm = f"{args.motor_sampling}-dt{args.physics_step_ms}"
    summaries = []
    for profile in args.profiles.split(","):
        if profile not in ("hover", "collective", "roll", "pitch", "yaw"):
            parser.error(f"unknown profile {profile}")
        output = ROOT / "results/physics-comparison" / arm / profile
        output.mkdir(parents=True, exist_ok=True)
        world = ROOT / "worlds" / f".run-physics-{arm}-{profile}.wbt"
        world.write_text(with_custom_data(base, {
            "phase": "physics-audit", "profile": profile,
            "motor_sampling": args.motor_sampling, "max_steps": "320",
        }))
        shutil.copy2(world, output / "world.wbt")
        for name in ("physics-comparison.csv", "physics-comparison.json", "last-run.json"):
            (ROOT / "results" / name).unlink(missing_ok=True)
        command = [str(args.webots), "--minimize", "--batch", f"--port={args.port}",
                   "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)]
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=90)
            (output / "run.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"Webots exited {result.returncode}; see {output / 'run.log'}")
            for name in ("physics-comparison.csv", "physics-comparison.json", "last-run.json"):
                shutil.copy2(ROOT / "results" / name, output / name)
            summary = json.loads((output / "physics-comparison.json").read_text())
            if summary["motor_sampling"] != args.motor_sampling or summary["profile"] != profile:
                raise RuntimeError("physics comparison result does not match this run")
            episode = json.loads((output / "last-run.json").read_text())
            if episode["collision"] or episode["phase"] != "physics-audit":
                raise RuntimeError("physics comparison ended with contact or wrong controller phase")
            summary["physics_step_ms"] = args.physics_step_ms
            summary["controller_step_ms"] = 10
            summary["hashes"] = {
                "controller": digest(ROOT / "controllers/raptor_webots/raptor_webots"),
                "controller_source": digest(ROOT / "controllers/raptor_webots/raptor_webots.cpp"),
                "comparison_source": digest(ROOT / "controllers/raptor_webots/physics_comparison.hpp"),
                "physics_model": digest(ROOT.parent / "physics.hpp"),
                "vehicle": digest(ROOT / "protos/RaptorCrazyflie.proto"),
                "world": digest(output / "world.wbt"),
                "trace": digest(output / "physics-comparison.csv"),
            }
            (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
            summaries.append(summary)
            print(profile, "max_position_m", summary["max_position_m"],
                  "max_body_rate_rps", summary["max_body_rate_rps"], flush=True)
        finally:
            world.unlink(missing_ok=True)
    (ROOT / "results/physics-comparison" / arm / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
