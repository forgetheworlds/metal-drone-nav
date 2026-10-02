#!/usr/bin/env python3
"""Cold-path evidence gate for Webots/Blender flight videos.

Decodes the movie itself (never trusts container metadata alone), applies
per-frame visibility and temporal-content gates, checks the episode receipt
against the controller's own objective thresholds, cross-checks pairing with
route/trajectory artifacts when present, and binds every input by SHA-256.
Emits a JSON verdict. Exit 0 only for verified navigation evidence or a
labeled synthetic fixture that passed every gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("verify_flight_videos.py requires numpy\n")
    raise

SCHEMA = "flight-video-verification-v1"
TOOL_VERSION = "1.0.0"

THRESHOLDS = {
    "goal_radius_m": {
        "value": 0.35,
        "source": "webots/controllers/raptor_webots/raptor_webots.cpp:435,457 (inside_goal / entry success); frozen scorer constant",
    },
    "hold_speed_max_mps": {
        "value": 0.5,
        "source": "webots/controllers/raptor_webots/raptor_webots.cpp:446,459; frozen scorer constant",
    },
    "hold_dwell_min_s": {
        "value": 0.2,
        "source": "webots/controllers/raptor_webots/raptor_webots.cpp:459 (0.2f-1e-6f); frozen scorer constant",
    },
    "visible_mean_min": {
        "value": 2.0,
        "source": "calibrated: flat-black x264 frames measure mean 0.000; frame 0 of a real Webots render measures mean 141",
    },
    "visible_std_min": {
        "value": 1.0,
        "source": "calibrated: codec noise on flat black measures spatial std 0.055; real render measures std 36",
    },
    "visible_frac_above16_min": {
        "value": 0.01,
        "source": "calibrated: flat-black frames light 0.01% of pixels above luma 16; real render lights 99.98%",
    },
    "visible_fraction_required": {
        "value": 0.90,
        "source": "a watchable run must be visibly rendered nearly throughout; margin covers fades/overlays",
    },
    "frozen_pair_fraction_max": {
        "value": 0.98,
        "source": "rejects static-image loops; tolerance allows a steady hover stretch inside an otherwise moving flight",
    },
    "frame_min_width": {"value": 320, "source": "watchability floor"},
    "frame_min_height": {"value": 180, "source": "watchability floor"},
    "frames_min": {"value": 2, "source": "a single frame is not a run"},
    "duration_ratio_min": {"value": 0.5, "source": "video must cover the episode; Webots native movies run ~1.0x episode time"},
    "duration_ratio_max": {"value": 2.0, "source": "guards against pairing the movie with an unrelated episode receipt"},
    "trace_goal_error_tol_m": {
        "value": 5e-3,
        "source": "trace goal_error column is written at 6 significant digits; recompute from pose+goal",
    },
    "trace_step_gap_max": {
        "value": 100,
        "source": "sparse traces log every 100 steps (raptor_webots.cpp:449); dense traces log every step",
    },
}

LIMITS = [
    "Frame statistics prove the movie is visibly rendered, non-constant content; they cannot prove the pixels depict this specific flight. Pairing rests on duration ratio, movie_bytes/movie_file agreement, route/episode field agreement, and SHA-256 binding.",
    "ffprobe/container metadata alone never satisfies any visual gate: every visual verdict comes from decoded pixel buffers.",
    "A very dark but real scene would be judged by the calibrated visibility floor (mean>=2, std>=1, frac>16 >=1%); scenes darker than that would need recalibration, which must be argued from measurements, not convenience.",
    "Non-16:9 sources keep native dimensions (no letterbox introduced); the visibility metrics operate on native frames.",
    "Collision status is taken from the controller receipt (contact points are not in the trace); this verifier does not independently re-derive collisions.",
    "A Blender reconstruction from recorded poses is accepted only when labeled reconstructed_simulation (trajectory receipt or explicit --provenance), never as camera footage.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_info() -> dict:
    def version(binary: str) -> str:
        try:
            out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=30)
            return (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown"
        except Exception:
            return "unavailable"

    return {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "version": TOOL_VERSION,
        "python": sys.version.split()[0],
        "ffmpeg": version("ffmpeg"),
        "ffprobe": version("ffprobe"),
    }


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, check_id: str, status: str, detail: str, measured=None, threshold=None) -> None:
        self.items.append(
            {
                "id": check_id,
                "status": status,
                "detail": detail,
                "measured": measured,
                "threshold": threshold,
            }
        )

    def ok(self, check_id: str, detail: str, measured=None, threshold=None) -> None:
        self.add(check_id, "pass", detail, measured, threshold)

    def fail(self, check_id: str, detail: str, measured=None, threshold=None) -> None:
        self.add(check_id, "fail", detail, measured, threshold)

    def warn(self, check_id: str, detail: str, measured=None, threshold=None) -> None:
        self.add(check_id, "warn", detail, measured, threshold)

    def missing(self, check_id: str, detail: str) -> None:
        self.add(check_id, "missing", detail, None, None)

    @property
    def has_fail(self) -> bool:
        return any(c["status"] == "fail" for c in self.items)

    def fails(self) -> list[str]:
        return [c["id"] for c in self.items if c["status"] == "fail"]

    def as_list(self) -> list[dict]:
        return self.items


def probe_video(path: Path, checks: Checks) -> dict | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", "-select_streams", "v:0", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:
        checks.fail("probe", f"ffprobe invocation failed: {exc}")
        return None
    if proc.returncode != 0:
        checks.fail("probe", f"ffprobe rejected the container: {proc.stderr.strip()[:400]}")
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        checks.fail("probe", f"ffprobe emitted unparseable JSON: {exc}")
        return None
    streams = data.get("streams") or []
    if not streams:
        checks.fail("probe", "container has no video stream")
        return None
    stream = streams[0]
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        checks.fail("probe", "video stream has no usable dimensions", {"width": width, "height": height})
        return None
    fps = 0.0
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = stream.get("duration")
    if duration in (None, "N/A"):
        duration = (data.get("format") or {}).get("duration")
    try:
        duration_f = float(duration) if duration not in (None, "N/A") else 0.0
    except (TypeError, ValueError):
        duration_f = 0.0
    nb = stream.get("nb_frames")
    try:
        nb_frames = int(nb) if nb not in (None, "N/A") else None
    except (TypeError, ValueError):
        nb_frames = None
    checks.ok(
        "probe",
        "container parsed",
        {
            "codec": stream.get("codec_name"),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_f,
            "nb_frames": nb_frames,
        },
    )
    return {
        "codec": stream.get("codec_name"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": duration_f,
        "nb_frames": nb_frames,
    }


def check_dimensions(probe: dict, checks: Checks) -> None:
    w, h = probe["width"], probe["height"]
    if w < THRESHOLDS["frame_min_width"]["value"] or h < THRESHOLDS["frame_min_height"]["value"]:
        checks.fail(
            "dimensions",
            "frame dimensions below watchability floor",
            {"width": w, "height": h},
            {"min_width": THRESHOLDS["frame_min_width"]["value"], "min_height": THRESHOLDS["frame_min_height"]["value"]},
        )
    else:
        checks.ok("dimensions", "frame dimensions usable", {"width": w, "height": h},
                  {"min_width": THRESHOLDS["frame_min_width"]["value"], "min_height": THRESHOLDS["frame_min_height"]["value"]})


def decode_and_analyze(path: Path, probe: dict, samples: int, checks: Checks) -> dict | None:
    width, height = probe["width"], probe["height"]
    frame_bytes = width * height
    expected = probe.get("nb_frames")
    if not expected and probe.get("fps") and probe.get("duration_s"):
        expected = int(round(probe["fps"] * probe["duration_s"]))
    stride = max(1, math.ceil((expected or samples) / max(1, samples)))
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        checks.fail("decode", f"ffmpeg invocation failed: {exc}")
        return None
    assert proc.stdout is not None
    means: list[float] = []
    stds: list[float] = []
    fracs: list[float] = []
    deltas: list[float] = []
    prev_small = None
    n_decoded = 0
    partial = b""
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) < frame_bytes:
                partial = buf
                break
            if n_decoded % stride == 0:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
                f32 = frame.astype(np.float32)
                means.append(float(f32.mean()))
                stds.append(float(f32.std()))
                fracs.append(float((frame > 16).mean()))
                small = frame[::4, ::4]
                if prev_small is not None:
                    deltas.append(float(np.mean(np.abs(small.astype(np.int16) - prev_small.astype(np.int16)))))
                prev_small = small
            n_decoded += 1
    except Exception as exc:
        proc.kill()
        checks.fail("decode", f"decode aborted mid-stream: {exc}")
        return None
    stderr = b""
    try:
        _, stderr = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        checks.fail("decode", "ffmpeg decode timed out after 120s")
        return None
    decode_ok = proc.returncode == 0 and not partial and n_decoded > 0
    if not decode_ok:
        checks.fail(
            "decode",
            "full decode failed",
            {
                "returncode": proc.returncode,
                "frames_decoded": n_decoded,
                "trailing_partial_bytes": len(partial),
                "stderr": stderr.decode(errors="replace").strip()[:400],
            },
        )
        return None
    checks.ok("decode", "every frame decoded cleanly",
              {"frames_decoded": n_decoded, "stride": stride, "frames_analyzed": len(means)})

    if expected:
        if n_decoded < 0.98 * expected:
            checks.fail("frame_count", "decoded fewer frames than the container declares",
                        {"decoded": n_decoded, "declared": expected}, {"min_ratio": 0.98})
        elif n_decoded != expected:
            checks.warn("frame_count", "decoded frame count differs slightly from container declaration",
                        {"decoded": n_decoded, "declared": expected})
        else:
            checks.ok("frame_count", "decoded frame count matches container", {"decoded": n_decoded})
    else:
        checks.missing("frame_count", "container does not declare nb_frames; decoded count recorded only")

    if n_decoded < THRESHOLDS["frames_min"]["value"]:
        checks.fail("frames_min", "video has too few frames to be a run",
                    {"frames": n_decoded}, {"min": THRESHOLDS["frames_min"]["value"]})
        return {"frames_decoded": n_decoded, "frames_analyzed": len(means), "stride": stride}

    mean_a = np.array(means, dtype=np.float64)
    std_a = np.array(stds, dtype=np.float64)
    frac_a = np.array(fracs, dtype=np.float64)
    visible = (mean_a >= THRESHOLDS["visible_mean_min"]["value"]) & \
              (std_a >= THRESHOLDS["visible_std_min"]["value"]) & \
              (frac_a >= THRESHOLDS["visible_frac_above16_min"]["value"])
    visible_fraction = float(visible.mean()) if visible.size else 0.0
    if deltas:
        frozen_fraction = float(np.mean(np.array(deltas) < 0.01))
        median_delta = float(np.median(deltas))
    else:
        frozen_fraction, median_delta = 1.0, 0.0

    if visible_fraction >= THRESHOLDS["visible_fraction_required"]["value"]:
        checks.ok("visible_frames", "sampled frames are visibly rendered",
                  {"visible_fraction": visible_fraction, "samples": int(visible.size)},
                  {"min": THRESHOLDS["visible_fraction_required"]["value"]})
    else:
        checks.fail("visible_frames", "sampled frames are black/near-constant; movie is not watchable",
                    {"visible_fraction": visible_fraction, "samples": int(visible.size),
                     "min_frame_mean": float(mean_a.min()) if mean_a.size else None,
                     "max_frame_mean": float(mean_a.max()) if mean_a.size else None},
                    {"min": THRESHOLDS["visible_fraction_required"]["value"]})

    if frozen_fraction <= THRESHOLDS["frozen_pair_fraction_max"]["value"]:
        checks.ok("temporal_content", "frame content changes over time",
                  {"frozen_pair_fraction": frozen_fraction, "median_frame_delta": median_delta},
                  {"max": THRESHOLDS["frozen_pair_fraction_max"]["value"]})
    else:
        checks.fail("temporal_content", "video is a frozen/static image loop, not a flight",
                    {"frozen_pair_fraction": frozen_fraction, "median_frame_delta": median_delta},
                    {"max": THRESHOLDS["frozen_pair_fraction_max"]["value"]})

    return {
        "frames_decoded": n_decoded,
        "frames_analyzed": len(means),
        "stride": stride,
        "visible_fraction": visible_fraction,
        "frozen_pair_fraction": frozen_fraction,
        "median_frame_delta": median_delta,
        "frame_mean_min": float(mean_a.min()) if mean_a.size else None,
        "frame_mean_max": float(mean_a.max()) if mean_a.size else None,
        "frame_std_median": float(np.median(std_a)) if std_a.size else None,
        "visible": visible,
    }


def load_episode(path: Path | None, checks: Checks) -> tuple[dict | None, Path | None]:
    if path is None:
        checks.missing("episode", "episode receipt not provided and not discoverable next to the video")
        return None, None
    if not path.is_file():
        checks.missing("episode", f"episode receipt not found at {path}")
        return None, path
    try:
        episode = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        checks.fail("episode", f"episode receipt unreadable: {exc}")
        return None, path
    if not isinstance(episode, dict):
        checks.fail("episode", "episode receipt is not a JSON object")
        return None, path
    checks.ok("episode", "episode receipt parsed", {"path": str(path)})
    return episode, path


def check_episode_semantics(episode: dict, objective: str, video_meta: dict | None,
                            video_path: Path, checks: Checks) -> str:
    synthetic = bool(episode.get("synthetic_fixture"))
    ep_obj = episode.get("goal_objective")
    if ep_obj in ("entry", "hold"):
        declared_obj = ep_obj if objective == "auto" else objective
        if objective != "auto" and objective != ep_obj:
            checks.warn("objective_declared", f"CLI --objective {objective} overrides episode {ep_obj}")
        else:
            checks.ok("objective_declared", f"evaluating {declared_obj} objective", {"objective": declared_obj})
    elif objective != "auto":
        declared_obj = objective
        checks.warn("objective_declared", f"episode lacks goal_objective; using CLI --objective {objective}")
    else:
        declared_obj = "entry"
        checks.warn("objective_declared", "episode does not declare goal_objective; evaluating controller default entry")

    if episode.get("success") is True:
        checks.ok("success", "episode reports success=true")
    else:
        checks.fail("success", "episode does not report success=true", {"success": episode.get("success")})

    if episode.get("collision") is False:
        checks.ok("collision", "episode reports collision=false")
    else:
        checks.fail("collision", "episode reports a collision", {"collision": episode.get("collision")})

    if episode.get("timeout") is True:
        checks.fail("timeout", "episode ran into its step budget", {"timeout": True})
    elif "timeout" in episode:
        checks.ok("timeout", "episode did not time out")

    steps, time_s = episode.get("steps"), episode.get("time_s")
    if isinstance(steps, (int, float)) and steps >= 1 and isinstance(time_s, (int, float)) and time_s > 0:
        checks.ok("episode_progress", "episode recorded motion", {"steps": steps, "time_s": time_s})
    else:
        checks.fail("episode_progress", "episode has no recorded progress", {"steps": steps, "time_s": time_s})

    entries = episode.get("goal_radius_entry_count")
    first_entry = episode.get("goal_radius_entry_first_time_s")
    final_error = episode.get("final_error_m")

    if isinstance(final_error, (int, float)) and isinstance(entries, (int, float)) and entries >= 1:
        radius = THRESHOLDS["goal_radius_m"]["value"]
        if declared_obj == "entry":
            ok = final_error < radius
            detail = "entry objective: final_error_m strictly inside goal radius"
        else:
            ok = final_error <= radius
            detail = "hold objective: final_error_m inside goal radius"
        if ok:
            checks.ok("goal_radius", detail, {"final_error_m": final_error}, {"goal_radius_m": radius})
        else:
            checks.fail("goal_radius", "final error outside the frozen goal radius",
                        {"final_error_m": final_error}, {"goal_radius_m": radius})
        if isinstance(first_entry, (int, float)) and first_entry >= 0 and isinstance(time_s, (int, float)) and first_entry <= time_s + 1e-9:
            checks.ok("goal_entry", "goal radius entry recorded within episode time",
                      {"entries": entries, "first_entry_s": first_entry})
        else:
            checks.fail("goal_entry", "goal entry timestamp missing or outside episode time",
                        {"entries": entries, "first_entry_s": first_entry})
    else:
        checks.fail("goal_entry", "episode lacks goal radius entry evidence",
                    {"final_error_m": final_error, "entries": entries})

    if declared_obj == "hold":
        dwell = episode.get("goal_dwell_s")
        speed = episode.get("final_world_speed_mps")
        dwell_min = THRESHOLDS["hold_dwell_min_s"]["value"]
        speed_max = THRESHOLDS["hold_speed_max_mps"]["value"]
        if isinstance(dwell, (int, float)) and dwell >= dwell_min - 1e-6:
            checks.ok("hold_dwell", "dwell requirement satisfied", {"goal_dwell_s": dwell}, {"min_s": dwell_min})
        else:
            checks.fail("hold_dwell", "insufficient low-speed dwell inside the goal",
                        {"goal_dwell_s": dwell}, {"min_s": dwell_min})
        if isinstance(speed, (int, float)) and speed <= speed_max:
            checks.ok("hold_speed", "final speed within hold limit",
                      {"final_world_speed_mps": speed}, {"max_mps": speed_max})
        else:
            checks.fail("hold_speed", "final speed above the hold limit",
                        {"final_world_speed_mps": speed}, {"max_mps": speed_max})

    if "movie_recording" in episode:
        if episode.get("movie_recording") is True and episode.get("movie_failed") is False:
            checks.ok("movie_receipt", "episode claims a native Webots movie that completed",
                      {"movie_failed": False})
        elif episode.get("movie_failed") is True:
            checks.fail("movie_receipt", "controller reported movie_failed=true")
        else:
            checks.warn("movie_receipt", "episode does not claim a native Webots movie",
                        {"movie_recording": episode.get("movie_recording")})

    declared_bytes = episode.get("movie_bytes")
    actual_bytes = video_path.stat().st_size if video_path.is_file() else None
    if isinstance(declared_bytes, (int, float)) and actual_bytes is not None:
        if int(declared_bytes) == actual_bytes:
            checks.ok("movie_bytes", "video size matches episode movie_bytes",
                      {"bytes": actual_bytes})
        else:
            checks.fail("movie_bytes", "video size does not match episode movie_bytes (wrong file pairing)",
                        {"declared": int(declared_bytes), "actual": actual_bytes})

    declared_file = episode.get("movie_file") or ""
    if declared_file and Path(declared_file).name != video_path.name:
        checks.fail("movie_file", "video filename does not match episode movie_file (wrong file pairing)",
                    {"declared": Path(declared_file).name, "actual": video_path.name})
    elif declared_file:
        checks.ok("movie_file", "video filename matches episode movie_file")

    if video_meta and isinstance(time_s, (int, float)) and time_s > 0:
        ratio = video_meta["duration_s"] / time_s if time_meta_valid(video_meta) else None
        if ratio is None:
            checks.missing("duration_match", "container declares no duration to compare against episode time_s")
        else:
            lo = THRESHOLDS["duration_ratio_min"]["value"]
            hi = THRESHOLDS["duration_ratio_max"]["value"]
            if lo <= ratio <= hi:
                checks.ok("duration_match", "video duration covers the episode",
                          {"video_duration_s": video_meta["duration_s"], "episode_time_s": time_s, "ratio": ratio},
                          {"range": [lo, hi]})
            else:
                checks.fail("duration_match", "video duration does not match episode time (wrong file pairing)",
                            {"video_duration_s": video_meta["duration_s"], "episode_time_s": time_s, "ratio": ratio},
                            {"range": [lo, hi]})

    return "synthetic_fixture" if synthetic else "navigation"


def time_meta_valid(video_meta: dict) -> bool:
    return isinstance(video_meta.get("duration_s"), (int, float)) and video_meta["duration_s"] > 0


def parse_route(path: Path) -> dict | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    match = re.search(r'customData\s+"([^"]*)"', text)
    if not match:
        return None
    fields = {}
    for part in match.group(1).split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    goal = None
    if "goal" in fields:
        try:
            goal = [float(v) for v in fields["goal"].split(",")]
        except ValueError:
            goal = None
    return {"fields": fields, "goal": goal}


def check_route(route: dict | None, episode: dict | None, checks: Checks) -> list[float] | None:
    if route is None:
        checks.missing("route", "no route.wbt available for independent field cross-check")
        return None
    fields = route["fields"]
    mismatches = {}
    if episode:
        for route_key, ep_key in (("seed", "seed"), ("policy", "policy"), ("goal_objective", "goal_objective")):
            if route_key in fields and ep_key in episode:
                route_val = Path(fields[route_key]).name if route_key == "policy" else fields[route_key]
                ep_val = Path(str(episode[ep_key])).name if ep_key == "policy" else str(episode[ep_key])
                if route_val != ep_val:
                    mismatches[route_key] = {"route": route_val, "episode": ep_val}
        if fields.get("movie_file") and episode.get("movie_file"):
            if fields["movie_file"] != episode.get("movie_file"):
                mismatches["movie_file"] = {"route": fields["movie_file"], "episode": episode.get("movie_file")}
    if mismatches:
        checks.fail("route_episode_consistency", "route.wbt customData disagrees with episode receipt",
                    mismatches)
    elif episode:
        checks.ok("route_episode_consistency", "route.wbt customData agrees with episode receipt",
                  {"fields_checked": [k for k in ("seed", "policy", "goal_objective", "movie_file") if k in fields]})
    else:
        checks.missing("route_episode_consistency", "episode receipt unavailable for route cross-check")
    return route.get("goal")


def check_trace(path: Path | None, episode: dict | None, goal: list[float] | None, checks: Checks) -> None:
    if path is None:
        checks.missing("trajectory", "no trajectory trace provided or discoverable")
        return
    if not path.is_file():
        checks.missing("trajectory", f"trajectory trace not found at {path}")
        return
    import csv

    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        checks.fail("trajectory", f"trajectory trace unreadable: {exc}")
        return
    required = {"step", "time_s", "x", "y", "z"}
    if not required.issubset(header):
        checks.fail("trajectory", "trajectory trace missing required columns",
                    {"missing": sorted(required - set(header)), "header": header})
        return
    if not rows:
        checks.fail("trajectory", "trajectory trace has no data rows")
        return
    try:
        steps_col = [int(float(r["step"])) for r in rows]
        times_col = [float(r["time_s"]) for r in rows]
        xs = [float(r["x"]) for r in rows]
        ys = [float(r["y"]) for r in rows]
        zs = [float(r["z"]) for r in rows]
    except (TypeError, ValueError) as exc:
        checks.fail("trajectory", f"trajectory trace has non-numeric values: {exc}")
        return
    if any(b < a for a, b in zip(times_col, times_col[1:])):
        checks.fail("trajectory", "trajectory time column is not monotonic")
        return
    checks.ok("trajectory", "trajectory trace parsed",
              {"rows": len(rows), "columns": header, "last_step": steps_col[-1], "last_time_s": times_col[-1]})

    if "goal_error" in header and goal and len(goal) == 3:
        recomputed = [
            math.sqrt(sum((g - p) ** 2 for g, p in zip(goal, (x, y, z))))
            for x, y, z in zip(xs, ys, zs)
        ]
        logged = [float(r["goal_error"]) for r in rows]
        worst = max(abs(a - b) for a, b in zip(recomputed, logged))
        tol = THRESHOLDS["trace_goal_error_tol_m"]["value"]
        if worst <= tol:
            checks.ok("trace_goal_error", "logged goal_error matches pose recomputed against route goal",
                      {"max_abs_diff_m": worst}, {"tol_m": tol})
        else:
            checks.fail("trace_goal_error", "logged goal_error inconsistent with pose and route goal",
                        {"max_abs_diff_m": worst}, {"tol_m": tol})
        checks.ok("trace_min_goal_error", "minimum goal error observed in trace (informational)",
                  {"min_goal_error_m": min(recomputed)})
    elif "goal_error" in header:
        checks.missing("trace_goal_error", "route goal vector unavailable; cannot recompute goal_error")

    if episode:
        ep_steps = episode.get("steps")
        if isinstance(ep_steps, (int, float)):
            if steps_col[-1] >= ep_steps:
                checks.fail("trace_pairing", "trace extends beyond the episode step count (different run?)",
                            {"trace_last_step": steps_col[-1], "episode_steps": ep_steps})
            elif ep_steps - steps_col[-1] <= THRESHOLDS["trace_step_gap_max"]["value"]:
                checks.ok("trace_pairing", "trace step coverage consistent with episode (sparse or dense logging)",
                          {"trace_last_step": steps_col[-1], "episode_steps": ep_steps},
                          {"max_gap": THRESHOLDS["trace_step_gap_max"]["value"]})
            else:
                checks.fail("trace_pairing", "trace stops far earlier than the episode (different run?)",
                            {"trace_last_step": steps_col[-1], "episode_steps": ep_steps},
                            {"max_gap": THRESHOLDS["trace_step_gap_max"]["value"]})
        density = len(rows) / ep_steps if isinstance(ep_steps, (int, float)) and ep_steps else 0
        if density >= 0.9 and isinstance(episode.get("final_error_m"), (int, float)):
            final_error_trace = recomputed[-1] if recomputed else None
            if final_error_trace is not None:
                diff = abs(final_error_trace - float(episode["final_error_m"]))
                if diff <= 0.05:
                    checks.ok("trace_final_state", "dense trace final state matches episode final_error_m",
                              {"diff_m": diff}, {"tol_m": 0.05})
                else:
                    checks.fail("trace_final_state", "dense trace final state disagrees with episode",
                                {"diff_m": diff}, {"tol_m": 0.05})
        else:
            checks.missing("trace_final_state", "sparse trace: final state not logged, cross-check not applicable")


def check_encoder_log(video_path: Path, checks: Checks) -> list[str]:
    log_path = video_path.parent / "webots.log"
    if not log_path.is_file():
        checks.missing("encoder_log", "no webots.log next to the video")
        return []
    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        checks.missing("encoder_log", f"webots.log unreadable: {exc}")
        return []
    warnings = []
    for pattern in (r"EOI missing", r"2pass curve failed to converge", r"movie.*failed", r"Movie.*failed"):
        warnings.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    done = re.search(r"WEBOTS_MOVIE_DONE\s+file=(\S+)\s+failed=(\d+)", text)
    if done and done.group(2) == "1":
        checks.fail("encoder_log", "webots.log reports movie encoding failure", {"line": done.group(0)})
    elif done:
        checks.ok("encoder_log", "webots.log reports movie encoding completed")
    else:
        checks.missing("encoder_log", "webots.log has no WEBOTS_MOVIE_DONE line")
    if warnings:
        checks.warn("encoder_log", "encoder symptoms present in webots.log (informational)",
                    {"warnings": sorted(set(warnings))})
    return sorted(set(warnings))


def resolve_policy(episode: dict | None, explicit: Path | None, checks: Checks) -> Path | None:
    if explicit is not None:
        if explicit.is_file():
            checks.ok("policy", "policy binary provided", {"path": str(explicit)})
            return explicit
        checks.missing("policy", f"policy binary not found at {explicit}")
        return None
    if not episode or not episode.get("policy"):
        checks.missing("policy", "no policy path provided and episode does not declare one")
        return None
    name = Path(str(episode["policy"])).name
    candidates = [
        Path(__file__).resolve().parent.parent / "assets" / name,
        Path(__file__).resolve().parent / ".." / "assets" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            checks.ok("policy", "policy binary located from episode declaration",
                      {"path": str(candidate.resolve())})
            return candidate.resolve()
    checks.missing("policy", f"policy binary {name} not found under assets/")
    return None


def determine_provenance(episode: dict | None, trajectory: Path | None, explicit: str,
                         evidence_class_hint: str) -> tuple[str, str]:
    if evidence_class_hint == "synthetic_fixture":
        return "synthetic_fixture", "episode.synthetic_fixture marker"
    if explicit in ("native", "reconstructed"):
        return (f"webots_native_recording" if explicit == "native" else "reconstructed_simulation"), "cli --provenance"
    if episode and episode.get("movie_recording") is True:
        return "webots_native_recording", "episode.movie_recording=true"
    if episode and episode.get("trajectory_trace") is True and trajectory is not None:
        return "reconstructed_simulation", "episode.trajectory_trace + trajectory receipt"
    if trajectory is not None and episode and episode.get("success") is not None:
        return "reconstructed_simulation", "trajectory receipt present (labeled reconstructed)"
    return "unknown", "no native movie claim, no trajectory receipt, no explicit --provenance"


def verify(args: argparse.Namespace) -> tuple[dict, int]:
    checks = Checks()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        checks.fail("video", f"video not found at {video_path}")
        verdict = build_verdict(checks, video_path, None, None, None, "unknown", "unverifiable", [], {}, args)
        return verdict, 2

    episode_path: Path | None
    if args.episode:
        episode_path = Path(args.episode).expanduser()
    else:
        episode_path = video_path.parent / "episode.json"
    trajectory_path: Path | None
    if args.trajectory:
        trajectory_path = Path(args.trajectory).expanduser()
    else:
        candidate = video_path.parent / "trace.csv"
        trajectory_path = candidate if candidate.is_file() else None
    scene_path: Path | None
    if args.scene:
        scene_path = Path(args.scene).expanduser()
    else:
        candidate = video_path.parent / "route.wbt"
        scene_path = candidate if candidate.is_file() else None
    scene_metadata = video_path.parent / "scene-metadata.json"

    hashes = {
        "video_sha256": sha256_file(video_path),
        "video_bytes": video_path.stat().st_size,
    }

    probe = probe_video(video_path, checks)
    video_meta = None
    visual = None
    if probe:
        check_dimensions(probe, checks)
        video_meta = probe
        visual = decode_and_analyze(video_path, probe, args.samples, checks)

    episode, episode_resolved = load_episode(episode_path, checks)
    evidence_class = "unknown"
    if episode is not None:
        if episode_path and Path(episode_path).is_file():
            hashes["episode_sha256"] = sha256_file(Path(episode_path))
        evidence_class = check_episode_semantics(
            episode, args.objective, video_meta, video_path, checks
        )

    route = None
    if scene_path and scene_path.is_file():
        hashes["scene_sha256"] = sha256_file(scene_path)
        if scene_path.suffix == ".wbt":
            route = parse_route(scene_path)
            if route is None:
                checks.missing("route", "route.wbt has no Robot.customData to cross-check")
    elif scene_path:
        checks.missing("scene", f"scene file not found at {scene_path}")
    else:
        checks.missing("scene", "no scene/route file provided or discoverable")

    if scene_metadata.is_file():
        hashes["scene_metadata_sha256"] = sha256_file(scene_metadata)

    goal = check_route(route, episode, checks)

    if trajectory_path and trajectory_path.is_file():
        hashes["trajectory_sha256"] = sha256_file(trajectory_path)
    elif trajectory_path:
        checks.missing("trajectory", f"trajectory trace not found at {trajectory_path}")

    check_trace(trajectory_path if trajectory_path and trajectory_path.is_file() else None,
                episode, goal, checks)

    policy_path = resolve_policy(episode, Path(args.policy).expanduser() if args.policy else None, checks)
    if policy_path:
        hashes["policy_sha256"] = sha256_file(policy_path)
        if episode and episode.get("policy") and Path(str(episode["policy"])).name != policy_path.name:
            checks.fail("policy", "policy binary filename disagrees with episode policy declaration",
                        {"episode": episode.get("policy"), "provided": policy_path.name})

    encoder_warnings = check_encoder_log(video_path, checks)

    provenance, provenance_source = determine_provenance(
        episode, trajectory_path if trajectory_path and trajectory_path.is_file() else None,
        args.provenance, evidence_class,
    )

    missing_receipts = []
    if episode is None:
        missing_receipts.append(f"episode.json receipt (looked at {episode_resolved or 'not provided'})")
    if trajectory_path is None or not trajectory_path.is_file():
        missing_receipts.append("trajectory trace.csv (optional; required only to auto-label a reconstruction)")
    if scene_path is None or not scene_path.is_file():
        missing_receipts.append("route.wbt scene copy (optional; enables pose/goal cross-checks)")
    if policy_path is None:
        missing_receipts.append("policy .bin binary (optional; hash binding only)")
    if provenance == "unknown":
        missing_receipts.append(
            "provenance label: no native movie claim in the episode, no trajectory receipt, "
            "and no explicit --provenance native|reconstructed"
        )

    visual_check_ids = {"video", "probe", "dimensions", "decode", "frame_count",
                        "frames_min", "visible_frames", "temporal_content"}
    episode_check_ids = {"success", "collision", "timeout", "episode_progress", "goal_radius",
                         "goal_entry", "hold_dwell", "hold_speed", "movie_bytes", "movie_file",
                         "duration_match", "trace_pairing", "trace_goal_error", "trace_final_state",
                         "route_episode_consistency", "movie_receipt", "policy"}
    failed_ids = set(checks.fails())
    visual_failed = bool(failed_ids & visual_check_ids)
    episode_failed = bool(failed_ids & episode_check_ids)

    if visual_failed:
        verdict_name, exit_code = "rejected_visual", 1
    elif episode is None:
        verdict_name, exit_code = "unverifiable", 2
    elif episode_failed:
        verdict_name, exit_code = "rejected_episode", 1
    elif provenance == "unknown":
        verdict_name, exit_code = "unverifiable", 2
    elif evidence_class == "synthetic_fixture":
        verdict_name, exit_code = "fixture_gates_pass", 0
    else:
        verdict_name, exit_code = "verified_navigation_evidence", 0

    verdict = build_verdict(
        checks, video_path, video_meta, visual, episode, provenance, verdict_name,
        missing_receipts, hashes, args,
        provenance_source=provenance_source,
        evidence_class=evidence_class,
        trajectory_path=trajectory_path,
        scene_path=scene_path,
        encoder_warnings=encoder_warnings,
        objective=args.objective,
    )
    return verdict, exit_code


def build_verdict(checks: Checks, video_path: Path, video_meta, visual, episode,
                  provenance: str, verdict_name: str, missing_receipts: list[str],
                  hashes: dict, args: argparse.Namespace, provenance_source: str = "unavailable",
                  evidence_class: str = "unknown", trajectory_path=None, scene_path=None,
                  encoder_warnings=None, objective: str = "auto") -> dict:
    counts: dict[str, int] = {}
    for item in checks.items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    visual_metrics = None
    if visual:
        visual_metrics = {k: v for k, v in visual.items() if k != "visible"}
    return {
        "schema": SCHEMA,
        "verdict": verdict_name,
        "verified": verdict_name in ("verified_navigation_evidence", "fixture_gates_pass"),
        "evidence_class": evidence_class,
        "provenance": {"kind": provenance, "label_source": provenance_source,
                       "camera_footage_claim": False},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "video": str(video_path),
            "episode": str(args.episode or (video_path.parent / "episode.json")),
            "trajectory": str(trajectory_path) if trajectory_path else None,
            "scene": str(scene_path) if scene_path else None,
            "policy": str(args.policy) if args.policy else None,
            "objective": objective,
            "samples": args.samples,
        },
        "video_metadata": video_meta,
        "visual_metrics": visual_metrics,
        "episode_summary": {
            k: episode.get(k) for k in (
                "world", "seed", "policy", "goal_objective", "success", "collision",
                "timeout", "steps", "time_s", "final_error_m", "goal_dwell_s",
                "final_world_speed_mps", "goal_radius_entry_count", "movie_recording",
                "movie_failed", "movie_bytes", "trajectory_samples",
            )
        } if episode else None,
        "hashes": hashes,
        "thresholds": THRESHOLDS,
        "checks": checks.as_list(),
        "check_counts": counts,
        "failed_checks": checks.fails(),
        "missing_receipts": missing_receipts,
        "encoder_warnings": encoder_warnings or [],
        "limits": LIMITS,
        "tool": tool_info(),
    }


def print_summary(verdict: dict, exit_code: int) -> None:
    print(f"verdict: {verdict['verdict']} (exit {exit_code})")
    print(f"evidence_class: {verdict['evidence_class']}  provenance: {verdict['provenance']['kind']}")
    counts = verdict.get("check_counts", {})
    print(f"checks: {counts}")
    if verdict.get("failed_checks"):
        print(f"failed: {verdict['failed_checks']}")
    for receipt in verdict.get("missing_receipts", []):
        print(f"missing receipt: {receipt}")
    vm = verdict.get("visual_metrics") or {}
    if vm:
        print(f"visual: visible_fraction={vm.get('visible_fraction')} "
              f"frames={vm.get('frames_decoded')} frozen={vm.get('frozen_pair_fraction')}")


def write_verdict(verdict: dict, out_path: str | None) -> None:
    if not out_path:
        return
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="flight movie to verify (mp4/mkv/...)")
    parser.add_argument("--episode", help="episode receipt JSON (default: episode.json next to the video)")
    parser.add_argument("--trajectory", help="trace.csv trajectory receipt (default: trace.csv next to the video)")
    parser.add_argument("--scene", help="route.wbt scene copy (default: route.wbt next to the video)")
    parser.add_argument("--policy", help="deployed policy .bin to hash-bind against the episode declaration")
    parser.add_argument("--objective", choices=("auto", "entry", "hold"), default="auto",
                        help="objective semantics; auto uses the episode's goal_objective")
    parser.add_argument("--provenance", choices=("auto", "native", "reconstructed"), default="auto",
                        help="explicit provenance label when it cannot be inferred from receipts")
    parser.add_argument("--samples", type=int, default=96,
                        help="target number of decoded frames to analyze for visual content")
    parser.add_argument("--out", help="write the JSON verdict to this path")
    parser.add_argument("--self-test", action="store_true",
                        help="run validator unit fixtures under results/mimo-video-verification/fixtures/")
    parser.add_argument("--self-test-dir",
                        default=str(Path(__file__).resolve().parent.parent / "results" / "mimo-video-verification"),
                        help="owned directory for self-test fixtures and selftest.json")
    return parser


def run_case(video: Path, episode: Path | None, expected_verdict: str, expected_exit: int,
             provenance: str = "auto", namespace_extra: dict | None = None) -> dict:
    ns = argparse.Namespace(
        video=str(video),
        episode=str(episode) if episode else None,
        trajectory=None, scene=None, policy=None, objective="auto",
        provenance=provenance, samples=96, out=None, self_test=False, self_test_dir=None,
        **(namespace_extra or {}),
    )
    verdict, code = verify(ns)
    return {
        "video": str(video),
        "episode": str(episode) if episode else None,
        "expected_verdict": expected_verdict,
        "expected_exit": expected_exit,
        "actual_verdict": verdict["verdict"],
        "actual_exit": code,
        "pass": verdict["verdict"] == expected_verdict and code == expected_exit,
        "failed_checks": verdict.get("failed_checks"),
        "visual_metrics": verdict.get("visual_metrics"),
        "missing_receipts": verdict.get("missing_receipts"),
    }


def self_test(out_dir: Path) -> int:
    fixtures = out_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "README.txt").write_text(
        "SYNTHETIC VALIDATOR FIXTURES ONLY.\n"
        "These clips are generated by ffmpeg test patterns to unit-test the verifier gates.\n"
        "They are NOT navigation evidence, NOT Webots camera footage, and must never be\n"
        "presented as successful flights. Every fixture episode JSON carries\n"
        "\"synthetic_fixture\": true, which forces evidence_class=synthetic_fixture.\n"
    )
    visible = fixtures / "SYNTHETIC_FIXTURE_visible.mp4"
    black = fixtures / "SYNTHETIC_FIXTURE_black.mp4"
    episode_ok = fixtures / "SYNTHETIC_FIXTURE_episode_ok.json"
    episode_failed = fixtures / "SYNTHETIC_FIXTURE_episode_failed.json"

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=25:duration=7", "-pix_fmt", "yuv420p", str(visible)],
        check=True, timeout=120,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=c=black:size=640x360:rate=25:duration=7", "-pix_fmt", "yuv420p", str(black)],
        check=True, timeout=120,
    )
    common = {
        "phase": "navigation", "seed": 0, "timeout": False, "steps": 700, "time_s": 7.0,
        "goal_objective": "hold", "goal_radius_entry_count": 1,
        "goal_radius_entry_first_time_s": 6.4, "goal_dwell_s": 0.25,
        "final_world_speed_mps": 0.31, "final_error_m": 0.12,
        "synthetic_fixture": True,
        "fixture_note": "SYNTHETIC validator fixture - not navigation evidence",
    }
    episode_ok.write_text(json.dumps({**common, "success": True, "collision": False}, indent=2) + "\n")
    episode_failed.write_text(
        json.dumps({**common, "success": False, "collision": True, "goal_dwell_s": 0.0}, indent=2) + "\n"
    )

    cases = [
        run_case(visible, episode_ok, "fixture_gates_pass", 0),
        run_case(black, episode_ok, "rejected_visual", 1),
        run_case(visible, episode_failed, "rejected_episode", 1),
        run_case(visible, fixtures / "DOES_NOT_EXIST.json", "unverifiable", 2),
    ]
    real_black = Path(
        __file__).resolve().parent / "results" / "navigation-arrival-experimental" / \
        "challenge_doorway_41001-seed-41001-hold-recording" / "flight.mp4"
    if real_black.is_file():
        cases.append(run_case(real_black, real_black.parent / "episode.json", "rejected_visual", 1))

    passed = sum(1 for case in cases if case["pass"])
    report = {
        "schema": "flight-video-verification-selftest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixtures_are_synthetic": True,
        "fixtures_are_navigation_evidence": False,
        "cases": cases,
        "passed": passed,
        "total": len(cases),
        "all_pass": passed == len(cases),
        "tool": tool_info(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selftest.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        mark = "PASS" if case["pass"] else "FAIL"
        print(f"[{mark}] {Path(case['video']).name} -> {case['actual_verdict']} "
              f"(expected {case['expected_verdict']})")
    print(f"self-test: {passed}/{len(cases)} passed; wrote {out_dir / 'selftest.json'}")
    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return self_test(Path(args.self_test_dir).expanduser())
    if not args.video:
        parser.error("--video is required (or use --self-test)")
    verdict, exit_code = verify(args)
    write_verdict(verdict, args.out)
    print_summary(verdict, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
