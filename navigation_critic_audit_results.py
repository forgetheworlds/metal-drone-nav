#!/usr/bin/env python3
"""Replay the frozen critic audit from its hash-checked public inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile

import numpy as np


def read_collected_rows(blob):
    """Read the original damaged layout and exclude its six overwritten rows."""
    header = struct.unpack_from("<8s14I9f4Q64s256sQ", blob)
    magic, version, rollouts, envs, horizon, rows = header[:6]
    if magic != b"AUDCOL01" or version != 1 or rows != envs * horizon:
        raise ValueError("Unexpected collector contract")
    if header[9] != 2:
        raise ValueError("This replay requires the retained two-input-block layout")
    names = ["value", "next_value", "reward", "x", "y", "z", "elapsed"]
    names += [f"co{i}" for i in [0, 1, 2, 19, 20, 52, 53, 61, 62, 63]]
    dtype = np.dtype(
        [(name, "<f4") for name in names]
        + [(name, "<u4") for name in ["episodes", "steps", "entry"]]
        + [(name, "u1") for name in ["terminal", "truncated", "pad0", "pad1"]]
    )
    row_bytes, input_bytes = rows * dtype.itemsize, rows * 64 * 4
    offsets = [0, row_bytes + input_bytes]
    offsets += [2 * (row_bytes + input_bytes) + (r - 2) * row_bytes
                for r in range(2, rollouts)]
    if dtype.itemsize != 84 or offsets[-1] + row_bytes != len(blob):
        raise ValueError("Unexpected dump size")
    data = np.concatenate([
        np.frombuffer(blob, dtype=dtype, count=rows, offset=offset)
        for offset in offsets
    ])
    valid = np.ones(len(data), dtype=bool)
    valid[:6] = False
    return header, data, valid


def prediction_metrics(target, value, mask):
    actual, prediction = target[mask], value[mask]
    if not len(actual) or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid graded prediction rows")
    error = actual - prediction
    return {"rows": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "explained_variance": float(1 - np.var(error) / np.var(actual))}


def replay(blob):
    header, data, valid = read_collected_rows(blob)
    rollouts, envs, horizon = header[2:5]
    gamma, gae_lambda = header[15:17]
    shape = (rollouts, horizon, envs)
    value = data["value"].astype(np.float64).reshape(shape)
    next_value = data["next_value"].astype(np.float64).reshape(shape)
    reward = data["reward"].astype(np.float64).reshape(shape)
    terminal = data["terminal"].astype(bool).reshape(shape)
    truncated = data["truncated"].astype(bool).reshape(shape)
    targets = np.empty(shape)
    carry = np.zeros((rollouts, envs))
    for tick in range(horizon - 1, -1, -1):
        bootstrap = np.where(terminal[:, tick], 0, next_value[:, tick])
        delta = reward[:, tick] + gamma * bootstrap - value[:, tick]
        boundary = terminal[:, tick] | truncated[:, tick]
        carry = delta + np.where(boundary, 0, gamma * gae_lambda * carry)
        targets[:, tick] = carry + value[:, tick]
    target_fit = prediction_metrics(targets.ravel(), value.ravel(), valid)

    # Follow actual successive rewards across rollout boundaries. Exclude
    # timeout and unfinished episodes; do not bootstrap with the same critic.
    shape = (rollouts * horizon, envs)
    value, reward = value.reshape(shape), reward.reshape(shape)
    terminal, truncated = terminal.reshape(shape), truncated.reshape(shape)
    valid = valid.reshape(shape)
    future = np.full(shape, np.nan)
    remaining = np.full(shape, -1, dtype=np.int32)
    carry = np.full(envs, np.nan)
    steps_left = np.full(envs, -1, dtype=np.int32)
    for tick in range(shape[0] - 1, -1, -1):
        complete = np.isfinite(carry)
        current = np.where(complete, reward[tick] + gamma * carry, np.nan)
        count = np.where(complete, steps_left + 1, -1)
        current = np.where(truncated[tick], np.nan, current)
        count = np.where(truncated[tick], -1, count)
        current = np.where(terminal[tick], reward[tick], current)
        count = np.where(terminal[tick], 0, count)
        current = np.where(valid[tick], current, np.nan)
        future[tick], remaining[tick] = current, count
        carry, steps_left = current, count
    complete = np.isfinite(future) & valid
    return {
        "scope": "One frozen stochastic source policy; correlated sampled outcomes, not conditional value truth or a causal learning result.",
        "main_transitions": int(rollouts * horizon * envs),
        "total_diagnostic_transitions": 1097728,
        "excluded_overwritten_rows": 6,
        "gae_target_fit": target_fit,
        "observed_future_returns": prediction_metrics(future, value, complete),
        "at_least32_future_steps": prediction_metrics(future, value, complete & (remaining >= 32)),
        "at_least64_future_steps": prediction_metrics(future, value, complete & (remaining >= 64)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent / "evidence/inputs/critic-audit/records.tar.gz")
    args = parser.parse_args()
    with tarfile.open(args.archive, "r:gz") as archive:
        manifest = json.load(archive.extractfile("SHA256.json"))
        for name, expected in manifest.items():
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"Input is not a regular file: {name}")
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected:
                raise ValueError(f"Input hash mismatch: {name}")
        actual = replay(archive.extractfile("collector.bin").read())
        expected = json.load(archive.extractfile("replay-expected.json"))
    for key in ["gae_target_fit", "observed_future_returns", "at_least32_future_steps", "at_least64_future_steps"]:
        if actual[key]["rows"] != expected[key]["rows"]:
            raise ValueError(f"Row count differs: {key}")
        for metric in ["rmse", "explained_variance"]:
            if abs(actual[key][metric] - expected[key][metric]) > 1e-9:
                raise ValueError(f"Prediction metric differs: {key}/{metric}")
    print(json.dumps({"verified_inputs": len(manifest), "replay": actual}, indent=2))


if __name__ == "__main__":
    main()
