#!/usr/bin/env python3
"""
Average the "VISION PROCESSING TIMING" blocks TrackStudio prints to stdout
every 30 vision frames, across an entire log file.

Usage:
    trackstudio run -c local_test_config.json -p 8010 2>&1 | tee vision_timing.log
    # let it run for a while, then, in another terminal:
    python3 scripts/analyze_vision_timing.py vision_timing.log
"""

import re
import sys
from collections import defaultdict

STEP_PATTERNS = {
    "layout_ms": r"Step 1 - Layout Calc:\s+([\d.]+)ms",
    "frame_extract_ms": r"Step 2 - Frame Extract:\s+([\d.]+)ms",
    "detect_track_ms": r"Step 3 - Detect\+Track:\s+([\d.]+)ms",
    "bev_transform_ms": r"Step 4 - BEV Transform:\s+([\d.]+)ms",
    "reid_ms": r"Step 5 - ReID Features:\s+([\d.]+)ms",
    "cross_cam_merge_ms": r"Step 6 - Cross-Cam Merge:\s+([\d.]+)ms",
    "total_ms": r"TOTAL PROCESSING TIME:\s+([\d.]+)ms",
}

STREAM_PATTERN = re.compile(
    r"Stream (\d+): detect=([\d.]+)ms \((\d+) objs\), track=([\d.]+)ms \((\d+) tracks\)"
)

SUMMARY_PATTERN = re.compile(
    r"SUMMARY: (\d+) streams, (\d+) detections, (\d+) tracks, (\d+) final"
)

COMPILED_STEPS = {name: re.compile(pattern) for name, pattern in STEP_PATTERNS.items()}


def analyze(path: str) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    frame_count = 0

    with open(path) as f:
        for line in f:
            if "VISION PROCESSING TIMING" in line:
                frame_count += 1
                continue

            for name, pattern in COMPILED_STEPS.items():
                match = pattern.search(line)
                if match:
                    values[name].append(float(match.group(1)))

            stream_match = STREAM_PATTERN.search(line)
            if stream_match:
                stream_id, detect_ms, det_count, track_ms, track_count = stream_match.groups()
                values[f"stream{stream_id}_detect_ms"].append(float(detect_ms))
                values[f"stream{stream_id}_detections"].append(int(det_count))
                values[f"stream{stream_id}_track_ms"].append(float(track_ms))
                values[f"stream{stream_id}_tracks"].append(int(track_count))

            summary_match = SUMMARY_PATTERN.search(line)
            if summary_match:
                _, detections, tracks, final = summary_match.groups()
                values["total_detections"].append(int(detections))
                values["total_tracks"].append(int(tracks))
                values["final_unified"].append(int(final))

    if frame_count == 0:
        print(f"No 'VISION PROCESSING TIMING' blocks found in {path}")
        sys.exit(1)

    print(f"Parsed {frame_count} timing blocks from {path}\n")

    def report(label: str, key: str) -> None:
        samples = values.get(key)
        if not samples:
            return
        avg = sum(samples) / len(samples)
        print(f"  {label:<28} avg={avg:8.2f}   min={min(samples):8.2f}   max={max(samples):8.2f}   n={len(samples)}")

    print("Pipeline step timings (ms):")
    report("Step 1 - Layout Calc", "layout_ms")
    report("Step 2 - Frame Extract", "frame_extract_ms")
    report("Step 3 - Detect+Track", "detect_track_ms")
    report("Step 4 - BEV Transform", "bev_transform_ms")
    report("Step 5 - ReID Features", "reid_ms")
    report("Step 6 - Cross-Cam Merge", "cross_cam_merge_ms")
    report("TOTAL", "total_ms")

    stream_ids = sorted({key.split("_")[0] for key in values if key.startswith("stream")})
    if stream_ids:
        print("\nPer-stream detect/track timings (ms) and counts:")
        for stream_id in stream_ids:
            report(f"{stream_id} detect", f"{stream_id}_detect_ms")
            report(f"{stream_id} track", f"{stream_id}_track_ms")
            report(f"{stream_id} detections (count)", f"{stream_id}_detections")
            report(f"{stream_id} tracks (count)", f"{stream_id}_tracks")

    print("\nPer-frame totals:")
    report("Detections", "total_detections")
    report("Tracks", "total_tracks")
    report("Final unified (post-merge)", "final_unified")

    if values.get("total_ms"):
        avg_total = sum(values["total_ms"]) / len(values["total_ms"])
        print(f"\nEffective achievable vision throughput: {1000.0 / avg_total:.2f} FPS (1000 / avg total ms)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/analyze_vision_timing.py <log_file>")
        sys.exit(1)
    analyze(sys.argv[1])
