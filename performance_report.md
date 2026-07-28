# TrackStudio Performance Report

Measured while running inference on real multi-camera footage (Wildtrack dataset,
cameras `cam5`/`cam7`, looped as `camera0`/`camera1`) on `whizdev`. Two measurement
passes are included: an initial **baseline** run, and a **post-fix** run taken after
three changes landed — deduplicating a redundant ReID forward pass, switching to a
real Market-1501 ReID checkpoint, and adding gap-tolerant cross-camera
re-identification (see Notes for details).

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

## Models Used

| Component | Model / Architecture | Package (version) | Params | FLOPs | Precision | Checkpoint |
|---|---|---|---|---|---|---|
| Detector | RF-DETR (`RFDETRBase`) — DINOv2 ViT-Small windowed backbone + DETR transformer head | `rfdetr` 1.1.0 (pinned to git commit `7fb9e50`, not a tagged release) | Not measured¹ | Not measured¹ | fp32 (no quantization, `torch.compile`, or ONNX/TensorRT export — all supported by the package but unused here) | `rf-detr-base.pth`, COCO-pretrained (91 classes; filtered to person only), auto-downloaded, 372MB |
| Tracker | DeepSORT (Kalman filter + IoU/appearance cost matching) | `trackers` 2.0.1 (Roboflow's own implementation, not the classic `nwojke/deep_sort`) | N/A — no neural network of its own | N/A | N/A | N/A |
| ReID | OSNet (`osnet_x1_0`) | `torchreid` 0.2.5 | 2,193,616 (~2.19M) | 978,878,352 (~0.98 GFLOPs) at 256×128 input | fp32, not quantized | `osnet_x1_0_market1501.pt` — Market-1501 person-ReID task-finetuned. Previously an ImageNet-classification-only backbone was silently in use instead (fixed — see Notes) |

¹ RF-DETR's params/FLOPs weren't measured directly in this session — would need `fvcore`/`ptflops`.

## FPS / Vision Processing Throughput

| Metric | Baseline | Post-fix |
|---|---|---|
| Configured vision FPS target | 10.0 | 10.0 |
| Vision processing ratio | 100% | 100% |
| Avg total processing latency (both cameras) | 271.0 ms | **183.9 ms** (range 178–198ms across samples) |
| Min / Max processing latency | 156.7 ms / 314.0 ms | 140.2 ms / 213.9 ms |
| **Effective achievable throughput** | **≈ 3.7 FPS** | **≈ 5.2–5.4 FPS** |

Post-fix step breakdown (78 timing blocks ≈ 2,340 vision frames, via `scripts/analyze_vision_timing.py`):

| Step | Avg (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| 1 — Layout Calc | 0.00 | 0.00 | 0.00 |
| 2 — Frame Extract | 0.01 | 0.01 | 0.03 |
| 3 — Detect+Track | 191.16 | 141.65 | 221.76 |
| 4 — BEV Transform | 0.26 | 0.15 | 0.39 |
| 5 — ReID Features | **0.04** | 0.02 | 0.06 |
| 6 — Cross-Cam Merge | 1.99 | 0.08 | 5.62 |
| **Total** | **193.46** | 141.92 | 224.99 |

Step 3 splits into Detect (RF-DETR) and Track (DeepSORT) per camera — Track is
consistently the more expensive half, not Detect:

| | Stream 0 | Stream 1 | Combined |
|---|---|---|---|
| Detect (RF-DETR) | 46.04 ms | 45.59 ms | **91.63 ms** |
| Track (DeepSORT) | 52.13 ms | 47.39 ms | **99.52 ms** |

Combined Detect + Track (91.63 + 99.52 = 191.15ms) accounts for essentially all of
Step 3's 191.16ms average. Track costs more because it isn't just Kalman-filter
math — `DeepSORTTracker.update()` also runs the OSNet ReID feature extraction
internally (to compute the appearance cost it blends with IoU for matching), so
both halves of Step 3 are really "one neural network forward pass each" (RF-DETR,
then OSNet), not a cheap classical-CV step next to an expensive one.

Step 5 collapsing from ~63–93ms (baseline) to ~0.04ms (post-fix) is the ReID dedup fix working as intended — it now reuses the appearance feature DeepSORT's own association step already computed instead of running OSNet a second time. Step 3 absorbs that cost instead (it's where DeepSORT's feature extraction actually happens), so total latency dropped by roughly the old Step 5 cost, consistent with the ~87ms overall improvement (271ms → 184ms).

**Note:** the backend's `/ws/stream-stats` endpoint reports `fps: 0, isRunning: false` regardless of actual pipeline state — that endpoint's internal state tracking is unreliable and shouldn't be trusted. The UI's own live "Combined FPS" counter (Live Streams tab) is the reliable reading for stream throughput.

## Cross-Camera Tracking

| Metric | Baseline | Post-fix |
|---|---|---|
| Total global tracks created (cumulative) | 869 | 314 |
| Active global tracks (at query time) | 3 | 8 |
| Multi-camera associations (simultaneous cross-camera merges) | 4,284 | 7,706 |
| Re-identifications after a gap (new capability) | N/A (didn't exist yet) | 596 |
| Active track ID mappings | 315 | 297 |

The 596 gap re-identifications (out of 3,016 vision frames processed in this run) confirm the new appearance-based re-acquisition logic is firing regularly on real footage, not just in the synthetic unit test used to validate it during development.

## GPU Utilization (NVIDIA A16, GPU 0)

Continuous `nvidia-smi dmon -s u -i 0` logs:

| Metric | Baseline (500 samples, ~8.3 min) | Post-fix (120 samples, ~2 min) |
|---|---|---|
| Avg SM (compute) utilization | 66.5% (69.3% excl. zero-gaps) | 64.4% |
| Avg memory-controller utilization | 44.3% (46.1% excl. zero-gaps) | 39.2% |
| Min / Max SM utilization | 0% / 83% | 0% / 78% |
| Zero-utilization samples | 20/500 (4%) | 6/120 (5%) |

Zero-utilization samples appear as isolated 1–4 sample gaps scattered through both logs rather than sustained idle stretches — consistent with brief inter-frame pauses in the vision cycle, not the GPU sitting idle for extended periods. GPU load is essentially unchanged between runs, as expected — the ReID dedup fix removes redundant *inference calls*, but each individual OSNet forward pass costs the same; fewer of them run per frame, not cheaper ones.

Supplementary point-in-time reading (`nvidia-smi --query-gpu`, baseline run only):

| Metric | Value |
|---|---|
| Memory used | ~0.9–1.3 GB / 15.36 GB (~6–8%) |
| Power draw | 38.4 W / 62.5 W limit (~61%) |
| Other GPUs (1, 2, 3) | Idle (0% util, ~3 MiB used) — all inference runs on GPU 0 |

## CPU / RAM Utilization (`trackstudio` process)

| Metric | Baseline (~4 min, 235 samples) | Post-fix (~1 min, 60 samples) |
|---|---|---|
| Avg CPU | 1578.5% (≈15.8 of 112 cores) | 1193.0% (≈11.9 of 112 cores) |
| Avg memory | 0.7% of system RAM | 0.6% of system RAM |
| Avg resident memory (RSS) | ≈2.84 GB | ≈2.58 GB |

CPU usage dropped noticeably post-fix (~15.8 → ~11.9 cores), consistent with removing one full OSNet forward pass per frame from the hot path.

## Camera Calibration

Camera-to-BEV alignment uses a classical single-plane **homography**, computed
independently per camera via OpenCV's `cv2.findHomography` from exactly four
manually-selected point correspondences — an operator captures a live frame per
camera through the web UI, clicks four points on known floor locations in the
image, and clicks four matching points on a shared 600×600px top-down canvas
(configured to represent a 12m×12m real-world area). Because exactly the minimum
four points are used, the solve is an exact fit rather than a least-squares
refinement over redundant points, so calibration accuracy depends entirely on the
precision of the manual point selection. For each tracked person, only the
bottom-center of their bounding box (the estimated foot position) is projected
through the homography — the standard technique for ground-plane homography, since
only points that actually lie on the calibrated floor plane transform correctly
under this model. All cameras' homographies map onto the same shared BEV
coordinate space, which is what allows the cross-camera merger to associate
detections spatially. Notably, this pipeline does not use full 3D camera
calibration — no intrinsic camera matrix, no lens distortion coefficients, and no
extrinsic rotation/translation vectors are computed or consumed anywhere; it is a
purely 2D, extrinsic-only, flat-floor mapping. (This is a different, less rigorous
approach than the Wildtrack dataset's own published calibration, which does
provide proper OpenCV-computed intrinsics/distortion coefficients and
solvePnP-derived extrinsics per camera — TrackStudio has no integration for
consuming that data as-is.)

## Notes / Caveats

- Both GPU logs are genuine continuous samples (`nvidia-smi dmon`), not point-in-time snapshots, except where noted. CPU/RAM figures are genuine continuous samples in both runs.
- The three fixes behind the baseline → post-fix comparison: (1) `TorchReIDExtractor` now auto-downloads and loads a Market-1501 person-ReID-finetuned checkpoint instead of a silently-in-use ImageNet-classification-only one; (2) the ReID feature computed during DeepSORT's own per-frame association is now reused for the cross-camera merger instead of re-running OSNet a second time on the same crops; (3) the cross-camera merger can now re-identify a track against existing global tracks by appearance when it has no live local-track mapping (e.g. after a brief occlusion), instead of always spawning a new global identity.
- Vision throughput remains **latency-bound**, not target-bound, even post-fix: a 10 FPS target implies a 100ms/frame budget, but the pipeline averages ~184ms, so ~5.2-5.4 FPS is the real ceiling regardless of the configured target. Step 3 (RF-DETR detection + DeepSORT tracking, including its ReID feature extraction) remains the dominant cost by a wide margin. Lowering `vision_fps` further won't help since it's already the bottleneck, not the throttle — reducing detector input resolution/model size, or batching both cameras into a single forward pass instead of the current sequential per-camera loop, are the levers that would actually help.
