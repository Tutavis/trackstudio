# TrackStudio Performance Report

Measured while running inference on real multi-camera footage (Wildtrack dataset,
cameras `cam5`/`cam7`, looped as `camera0`/`camera1`) on `whizdev`.

## Test Setup

| | |
|---|---|
| Input footage | Wildtrack raw recordings (cam5 → camera0, cam7 → camera1), 1920×1080 @ 59.94fps native, looped, downscaled/retimed to 720×480 @ 15fps by ffmpeg before ingestion |
| Detector / Tracker | RF-DETR + DeepSORT (`RFDETRTracker`) |
| Cross-camera merger | BEV Cluster |
| Configured vision FPS target | 10.0 |
| Access path | Browser (VS Code Remote-SSH tunnel) → WebRTC (TURN relay via local `coturn`) → TrackStudio |

## Environment

| | |
|---|---|
| Host | whizdev |
| CPU | Intel(R) Xeon(R) Gold 6258R @ 2.70GHz, 112 logical cores |
| RAM | 376 GiB total |
| GPU | NVIDIA A16, 15.36 GB VRAM, driver 580.105.08 (4× A16 present on host; TrackStudio uses only GPU 0) |

## FPS / Vision Processing Throughput

| Metric | Value |
|---|---|
| Configured vision FPS target | 10.0 |
| Vision frames processed | 5,352 |
| Vision processing ratio | 100% (every frame got a fresh detect→track→merge pass; none served from cache) |
| Avg processing latency (full pipeline, both cameras) | 271.0 ms |
| Min / Max processing latency | 156.7 ms / 314.0 ms |
| **Effective achievable vision throughput** | **≈ 3.7 FPS** (1000 / 271ms) |
| Combined video/WebRTC stream FPS | ~15 fps (ffmpeg push target); see caveat below |

**Note:** the backend's `/ws/stream-stats` endpoint reported `fps: 0, isRunning: false` at query time despite the pipeline actively processing frames (5,352 vision frames is proof it was running) — that endpoint's internal state tracking looks stale/unreliable and shouldn't be trusted as-is. The UI's own live "Combined FPS" counter (Live Streams tab) is the more reliable reading for stream throughput.

## Cross-Camera Tracking

| Metric | Value |
|---|---|
| Total global tracks created (cumulative) | 869 |
| Active global tracks (at query time) | 3 |
| Multi-camera associations (successful cross-camera merges) | 4,284 |
| Active track ID mappings | 315 |

## GPU Utilization (NVIDIA A16, GPU 0)

Continuous `nvidia-smi dmon -s u -i 0` log, 500 samples (~8.3 min @ 1Hz):

| Metric | All samples | Excluding brief zero-gaps* |
|---|---|---|
| Avg SM (compute) utilization | 66.5% | 69.3% |
| Avg memory-controller utilization | 44.3% | 46.1% |
| Min / Max SM utilization | 0% / 83% | 23% / 83% |

\* 20 of 500 samples (4%) read 0%, appearing as isolated 1–4 sample gaps scattered through the log rather than one sustained idle stretch — consistent with brief inter-frame pauses in the ~271ms-per-frame vision cycle, not the GPU sitting idle for extended periods.

Supplementary point-in-time reading (`nvidia-smi --query-gpu`, not covered by the dmon log above):

| Metric | Value |
|---|---|
| Memory used | ~0.9–1.3 GB / 15.36 GB (~6–8%) |
| Power draw | 38.4 W / 62.5 W limit (~61%) |
| Other GPUs (1, 2, 3) | Idle (0% util, ~3 MiB used) — all inference runs on GPU 0 |

## CPU / RAM Utilization (`trackstudio` process)

Sampled once per second for ~4 minutes (235 samples) via `ps -o %cpu,%mem,rss`:

| Metric | Value |
|---|---|
| Samples | 235 (~4 min) |
| Avg CPU | 1578.5% (≈ 15.8 of 112 logical cores, ≈ 14.1% of total system CPU capacity) |
| Avg memory | 0.7% of system RAM |
| Avg resident memory (RSS) | ≈ 2.84 GB |

## Notes / Caveats

- GPU figures above are point-in-time `nvidia-smi` snapshots taken during active inference, not a continuous log. CPU/RAM figures are a genuine ~4-minute continuous sample, averaged.
- Vision throughput is currently **latency-bound**, not target-bound: the configured 10 FPS target implies a 100ms/frame budget, but the actual full pipeline (detect → track → BEV transform → ReID → cross-camera merge, across 2 cameras) averages 271ms — so the real achievable rate is ~3.7 FPS regardless of the configured target. Server console output (search terminal logs for `🎯 VISION PROCESSING TIMING`, printed automatically every 30 vision frames) breaks this down step-by-step; Step 3 (detection + tracking) is the dominant cost. Options to close this gap: reduce detector input resolution/model size, or profile/optimize the per-step code — lowering `vision_fps` further won't help since it's already the bottleneck, not the throttle.
