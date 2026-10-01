#!/usr/bin/env python3
"""Export the pinned RLtools RAPTOR checkpoint header to the local FP32 ABI.

Usage:
  python3 export_raptor.py /path/to/checkpoint.tar.gz
  python3 export_raptor.py /path/to/checkpoint.h --output assets/raptor.bin

The archive may be the upstream data/raptor-policy-checkpoint.tar.gz. This
script uses only the Python standard library and preserves the little-endian
float bytes emitted by RLtools' checkpoint exporter.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import struct
import tarfile

MAGIC = b"RAPTOR1\0"
FIXTURE_MAGIC = b"RPFIX1\0\0"
PARAMETER_FLOATS = 2084
INPUT_DIM = 22
OUTPUT_DIM = 4
FIXTURE_STEPS = 16


def bytes_array(header: str, qualified_namespace: str) -> bytes:
    names = qualified_namespace.split("::")
    # RLtools emits a qualified outer namespace, then nested layer/parameter
    # namespaces (for example `namespace rl_tools::checkpoint::actor {`).
    outer_count = 4 if names[:3] == ["rl_tools", "checkpoint", "example"] else 3
    outer = "::".join(names[:outer_count])
    path = r"namespace\s+" + re.escape(outer) + r"\s*\{"
    if outer_count < len(names):
        path += r"[\s\S]*?" + r"[\s\S]*?".join(r"namespace\s+" + re.escape(name) + r"\s*\{" for name in names[outer_count:])
    match = re.search(path + r".*?memory\[\]\s*=\s*\{([^}]*)\}", header, re.S)
    if not match:
        raise ValueError(f"checkpoint is missing {qualified_namespace} memory")
    values = [int(value.strip()) for value in match.group(1).split(",") if value.strip()]
    if any(value < 0 or value > 255 for value in values):
        raise ValueError(f"invalid byte in {qualified_namespace}")
    return bytes(values)


def checkpoint_header(path: pathlib.Path) -> str:
    if path.is_dir():
        candidates = list(path.rglob("checkpoint.h"))
        if len(candidates) != 1:
            raise ValueError(f"expected one checkpoint.h under {path}, found {len(candidates)}")
        return candidates[0].read_text()
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            candidates = [item for item in archive.getmembers() if item.isfile() and item.name.endswith("/checkpoint.h")]
            if len(candidates) != 1:
                raise ValueError(f"expected one checkpoint.h in {path}, found {len(candidates)}")
            stream = archive.extractfile(candidates[0])
            if stream is None:
                raise ValueError("could not read checkpoint.h from archive")
            return stream.read().decode("utf-8")
    return path.read_text()


def extract(header: str) -> bytes:
    # The embedded parameter bytes are already little-endian IEEE754 float32.
    sections = [
        ("rl_tools::checkpoint::actor::layer_0::weights::parameters_memory", 16 * 22 * 4),
        ("rl_tools::checkpoint::actor::layer_0::biases::parameters_memory", 16 * 4),
        ("rl_tools::checkpoint::actor::layer_1::weights_input::parameters_memory", 48 * 16 * 4),
        ("rl_tools::checkpoint::actor::layer_1::weights_hidden::parameters_memory", 48 * 16 * 4),
        ("rl_tools::checkpoint::actor::layer_1::biases_input::parameters_memory", 48 * 4),
        ("rl_tools::checkpoint::actor::layer_1::biases_hidden::parameters_memory", 48 * 4),
        ("rl_tools::checkpoint::actor::layer_1::initial_hidden_state::parameters_memory", 16 * 4),
        ("rl_tools::checkpoint::actor::layer_2::weights::parameters_memory", 4 * 16 * 4),
        ("rl_tools::checkpoint::actor::layer_2::biases::parameters_memory", 4 * 4),
    ]
    parameters = bytearray()
    for namespace, expected_size in sections:
        section = bytes_array(header, namespace)
        if len(section) != expected_size:
            raise ValueError(f"{namespace} has {len(section)} bytes; expected {expected_size}")
        parameters.extend(section)
    if len(parameters) != PARAMETER_FLOATS * 4:
        raise AssertionError("internal parameter count is wrong")

    # RLtools' official exported oracle sequence: batch 0, steps 0..15.
    input_bytes = bytes_array(header, "rl_tools::checkpoint::example::input")
    output_bytes = bytes_array(header, "rl_tools::checkpoint::example::output")
    input_shape = (500, 2, INPUT_DIM)
    output_shape = (500, 2, OUTPUT_DIM)
    if len(input_bytes) != input_shape[0] * input_shape[1] * input_shape[2] * 4:
        raise ValueError("unexpected official example input size")
    if len(output_bytes) != output_shape[0] * output_shape[1] * output_shape[2] * 4:
        raise ValueError("unexpected official example output size")
    inputs = bytearray()
    outputs = bytearray()
    for step in range(FIXTURE_STEPS):
        input_offset = (step * 2 * INPUT_DIM) * 4
        output_offset = (step * 2 * OUTPUT_DIM) * 4
        inputs.extend(input_bytes[input_offset:input_offset + INPUT_DIM * 4])
        outputs.extend(output_bytes[output_offset:output_offset + OUTPUT_DIM * 4])

    header_bytes = MAGIC + struct.pack("<II", 1, PARAMETER_FLOATS)
    fixture_header = FIXTURE_MAGIC + struct.pack("<III", FIXTURE_STEPS, INPUT_DIM, OUTPUT_DIM)
    return header_bytes + parameters + fixture_header + inputs + outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path, help="upstream checkpoint.h, its directory, or checkpoint tar archive")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("assets/raptor.bin"))
    args = parser.parse_args()
    blob = extract(checkpoint_header(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"wrote {args.output}: {PARAMETER_FLOATS} FP32 parameters plus {FIXTURE_STEPS}-step official fixture ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
