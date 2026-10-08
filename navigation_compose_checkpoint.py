"""Compose frozen XYZ and yaw teachers into one exactly equivalent MLP.

Inputs and hidden activations are shared conventions: 184 inputs, tanh hidden,
four Gaussian means. This joins hidden units and selects output-head weights.
It produces parameter-only inference/warmstart weights, never resumable state.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

INPUTS = 184
HEADER_BYTES = 136


def read_model(path):
    data = path.read_bytes()
    if data[:7] != b"PPOFIX1" or len(data) < HEADER_BYTES:
        raise ValueError("Expected PPOFIX1 parameter header")
    version, count, critic = struct.unpack_from("<3I", data, 8)
    width, remainder = divmod(count - 8, INPUTS + 5)
    if version not in (8, 9, 10) or remainder or width < 1 or critic != 4225:
        raise ValueError("Expected compatible 184/H/4 actor, 64/64/1 critic")
    size = HEADER_BYTES + 4 * (count + critic)
    if len(data) < size:
        raise ValueError("Truncated parameters")
    actor = np.frombuffer(data, dtype="<f4", offset=HEADER_BYTES, count=count)
    if not np.isfinite(actor).all():
        raise ValueError("Non-finite actor")
    return data, actor, width, count, size


def compose(xyz_path, yaw_path, destination):
    xyz, x, hx, _, _ = read_model(xyz_path)
    yaw, y, hy, cy, yaw_size = read_model(yaw_path)
    if xyz[:12] != yaw[:12]:
        raise ValueError("Checkpoint header version mismatch")
    width = hx + hy
    count = 189 * width + 8
    actor = np.zeros(count, dtype="<f4")
    w1 = actor[: width * INPUTS].reshape(width, INPUTS)
    w1[:hx] = x[: hx * INPUTS].reshape(hx, INPUTS)
    w1[hx:] = y[: hy * INPUTS].reshape(hy, INPUTS)
    actor[width * INPUTS : width * (INPUTS + 1)] = np.concatenate(
        (x[hx * INPUTS : hx * (INPUTS + 1)], y[hy * INPUTS : hy * (INPUTS + 1)])
    )
    w2 = actor[width * (INPUTS + 1) : width * (INPUTS + 5)].reshape(4, width)
    w2[:3, :hx] = x[hx * (INPUTS + 1) : hx * (INPUTS + 5)].reshape(4, hx)[:3]
    w2[3, hx:] = y[hy * (INPUTS + 1) : hy * (INPUTS + 5)].reshape(4, hy)[3]
    actor[-8:-5] = x[-8:-5]
    actor[-5] = y[-5]
    actor[-4:-1] = x[-4:-1]
    actor[-1] = y[-1]
    header = bytearray(yaw[:HEADER_BYTES])
    struct.pack_into("<I", header, 12, count)
    struct.pack_into("<I", header, 36, 0)
    struct.pack_into("<Q", header, 40, 0)
    critic = yaw[HEADER_BYTES + 4 * cy : yaw_size]
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + actor.tobytes() + critic)

    # Independent random input proof in float64 avoids a hardware reduction-order
    # claim; real source flight parity is verified separately by the caller.
    def means(weights, h, obs):
        hidden = np.tanh(
            obs @ weights[: h * INPUTS].reshape(h, INPUTS).astype(np.float64).T
            + weights[h * INPUTS : h * (INPUTS + 1)]
        )
        return (
            hidden
            @ weights[h * (INPUTS + 1) : h * (INPUTS + 5)]
            .reshape(4, h)
            .astype(np.float64)
            .T
            + weights[-8:-4]
        )

    obs = np.random.default_rng(20261218).normal(0, 0.3, (256, INPUTS))
    expected = means(x, hx, obs)
    expected[:, 3] = means(y, hy, obs)[:, 3]
    error = float(np.max(np.abs(means(actor, width, obs) - expected)))
    if error > 1e-12:
        raise ValueError(f"Composition failed functional proof: {error}")
    record = {
        "purpose": "Parameter-only composed actor; not resumable or deployed by default",
        "xyz_sha256": hashlib.sha256(xyz).hexdigest(),
        "yaw_sha256": hashlib.sha256(yaw).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "actor_inputs": INPUTS,
        "actor_width": width,
        "actor_parameters": count,
        "critic_parameters": 4225,
        "float64_probe_max_abs_error": error,
        "probe_rows": 256,
        "critic_source": "yaw actor checkpoint, unchanged; no optimizer or simulator state copied",
    }
    destination.with_suffix(destination.suffix + ".composition.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("yaw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compose(args.xyz, args.yaw, args.output), indent=2))
