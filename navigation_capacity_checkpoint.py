"""Widen the 184-input navigation actor without changing its initial function.

The result contains only the checkpoint header and actor/critic weights. It is
a warm-start input for a matching capacity trainer, never a resumable checkpoint.
Extra neurons get independent input weights and zero output weights so they can
learn distinct features while preserving the existing actor at initialization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

ACTOR_INPUTS = 184
OUTPUTS = 4
HEADER_BYTES = 136


def widen_checkpoint(source, destination, width, seed):
    data = source.read_bytes()
    if len(data) < HEADER_BYTES or data[:7] != b"PPOFIX1":
        raise ValueError("Expected PPO checkpoint header")
    version, actor_count, critic_count = struct.unpack_from("<3I", data, 8)
    old_width, remainder = divmod(actor_count - 8, ACTOR_INPUTS + OUTPUTS + 1)
    if version not in (8, 9, 10) or remainder or old_width != 64 or critic_count != 4225:
        raise ValueError("Expected current 184/64/4 actor and 64/64/1 critic")
    if width < old_width or width % 64:
        raise ValueError("Actor width must be a multiple of 64, at least 64")
    prefix_bytes = HEADER_BYTES + 4 * (actor_count + critic_count)
    if len(data) < prefix_bytes:
        raise ValueError("Checkpoint parameters truncated")
    old = np.frombuffer(data, dtype="<f4", count=actor_count, offset=HEADER_BYTES)
    new_count = (ACTOR_INPUTS + OUTPUTS + 1) * width + 8
    widened = np.zeros(new_count, dtype="<f4")
    old_bias = old_width * ACTOR_INPUTS
    new_bias = width * ACTOR_INPUTS
    old_output = old_bias + old_width
    new_output = new_bias + width
    rng = np.random.default_rng(seed)
    input_weights = widened[:new_bias].reshape(width, ACTOR_INPUTS)
    input_weights[:old_width] = old[:old_bias].reshape(old_width, ACTOR_INPUTS)
    input_weights[old_width:] = rng.normal(0, 1 / math.sqrt(ACTOR_INPUTS), (width - old_width, ACTOR_INPUTS))
    widened[new_bias:new_output][:old_width] = old[old_bias:old_output]
    widened[new_output:new_output + OUTPUTS * width].reshape(OUTPUTS, width)[:, :old_width] = old[old_output:old_output + OUTPUTS * old_width].reshape(OUTPUTS, old_width)
    widened[-8:] = old[-8:]  # Output biases and Gaussian log standard deviations.
    critic = data[HEADER_BYTES + 4 * actor_count:prefix_bytes]
    header = bytearray(data[:HEADER_BYTES])
    struct.pack_into("<I", header, 12, new_count)
    # No old progress is claimed by this warm-start-only parameter file.
    struct.pack_into("<I", header, 36, 0)
    struct.pack_into("<Q", header, 40, 0)
    if not np.isfinite(widened).all():
        raise ValueError("Nonfinite widened weights")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + widened.tobytes() + critic)
    report = {"purpose": "Function-preserving actor warm start; not resumable",
              "source_sha256": hashlib.sha256(data).hexdigest(),
              "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "actor_inputs": ACTOR_INPUTS, "actor_width": width, "actor_parameters": new_count,
              "critic_parameters": critic_count, "extra_input_seed": seed,
              "extra_output_weights": "zero", "critic_bytes_unchanged": True}
    destination.with_suffix(destination.suffix + ".capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(widen_checkpoint(args.source, args.destination, args.width, args.seed), indent=2))
