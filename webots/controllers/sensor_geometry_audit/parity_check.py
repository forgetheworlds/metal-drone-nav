#!/usr/bin/env python3
"""Parity check: production adapter source text vs the Python replication.

Extracts clampf()/normalize_ranges()/pool_ranges() verbatim from the read-only
production controller, compiles them standalone, runs them on the measured native
frames, and compares against the analyzer's Python port. No production file is
modified; generated files live under results/omp-sensor-transfer/sensor-geometry/.

Usage: python3 webots/controllers/sensor_geometry_audit/parity_check.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "webots" / "controllers" / "raptor_webots" / "raptor_webots.cpp"
OUT = ROOT / "results" / "omp-sensor-transfer" / "sensor-geometry" / "parity"
FRAMES = ROOT / "results" / "omp-sensor-transfer" / "sensor-geometry" / "native-frames.jsonl"
WIDTH, HEIGHT, PIXELS = 20, 16, 320

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import adapter_resample, pool  # noqa: E402  (same port the analyzer uses)


def extract() -> str:
    text = SRC.read_text()
    start = text.index("float clampf(float x,float lo,float hi)")
    end = text.index("void quat_to_yaw")
    body = text[start:end]
    assert "normalize_ranges" in body and "pool_ranges" in body
    return body


HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
constexpr float kRangeMax = 12.0f;
%s
int main(int argc,char**argv){
  if(argc!=2) return 2;
  FILE*f=fopen(argv[1],"r"); if(!f) return 3;
  float hfov=0; if(fscanf(f,"%%f",&hfov)!=1) return 4;
  float raw[320];
  for(int i=0;i<320;i++){ char tok[64]; if(fscanf(f,"%%63s",tok)!=1) return 5;
    raw[i]= (strcmp(tok,"null")==0||strcmp(tok,"inf")==0) ? INFINITY : strtof(tok,nullptr); }
  fclose(f);
  float ranges[320], pooled[80];
  normalize_ranges(raw,20,16,hfov,ranges);
  pool_ranges(ranges,pooled);
  for(int i=0;i<320;i++) printf("%%.9g\n",ranges[i]);
  for(int i=0;i<80;i++) printf("%%.9g\n",pooled[i]);
  printf("%%.9g\n",double(kRangeMax));
  return 0;
}
""" % extract()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cpp = OUT / "adapter_extract.cpp"
    cpp.write_text(HARNESS)
    binary = OUT / "adapter_extract"
    build = subprocess.run(["/usr/bin/clang++", "-std=c++17", "-O2", "-o", str(binary), str(cpp)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return 2

    records = [json.loads(l) for l in FRAMES.read_text().splitlines() if l.strip()]
    hfov = records[0]["field_of_view_rad"]
    worst_ranges, worst_pooled, rows = 0.0, 0.0, 0
    for cap in records:
        if cap.get("record") != "capture":
            continue
        feed = OUT / "frame.txt"
        feed.write_text(" ".join(["%.9g" % hfov] +
                                 [("null" if v is None else "%.9g" % v) for v in cap["raw"]]))
        run = subprocess.run([str(binary), str(feed)], capture_output=True, text=True)
        if run.returncode != 0:
            print(f"harness failed rc={run.returncode} case={cap['case']}", file=sys.stderr)
            return 3
        got = [float(x) for x in run.stdout.split()]
        c_ranges, c_pooled = got[:320], got[320:400]
        py_ranges = adapter_resample(cap["raw"], hfov)
        py_pooled = pool(py_ranges)
        for a, b in zip(c_ranges, py_ranges):
            worst_ranges = max(worst_ranges, abs(a - b))
        for a, b in zip(c_pooled, py_pooled):
            worst_pooled = max(worst_pooled, abs(a - b))
        rows += 1

    result = {
        "schema": "adapter-source-text-parity-v1",
        "production_source": str(SRC.relative_to(ROOT)),
        "extracted_symbols": ["clampf", "normalize_ranges", "pool_ranges"],
        "method": "functions extracted verbatim from the production source text, compiled standalone, "
                  "run on the measured native frames, compared against the analyzer Python port",
        "frames_compared": rows,
        "pixels_compared": rows * PIXELS,
        "max_abs_range_difference_m": worst_ranges,
        "max_abs_pooled_difference_m": worst_pooled,
        "status": "PASS" if max(worst_ranges, worst_pooled) < 1e-4 else "FAIL",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
