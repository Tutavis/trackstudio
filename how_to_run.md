# How to Run TrackStudio (Local Dev Setup)

This covers running TrackStudio on `whizdev` with a locally-run MediaMTX (not Docker
Compose) and synthetic test camera streams, including the extra WebRTC setup needed
because this box is accessed remotely (e.g. VS Code Remote-SSH / port forwarding)
rather than from a browser on the same machine.

## Prerequisites

- `local_mediamtx/` present (built with `mediamtx` binary + `mediamtx.yml`, configured
  for ports `9554` RTSP / `2935` RTMP / `9888` HLS / `9997` API)
- `ffmpeg` installed
- Docker, for the TURN relay (only needed if the browser is on a **different**
  machine than the one running `trackstudio` — see below)

## 1. Start MediaMTX

```bash
cd local_mediamtx
./mediamtx
```

## 2. Start the two test camera streams

In two separate terminals:

```bash
./tests/stream_camera0_ffmpeg.sh
./tests/stream_camera1_ffmpeg.sh
```

`tests/videos/` is empty by default, so these fall back to a synthetic `testsrc2`
test pattern. That fallback branch uses `-re` (added — see Troubleshooting below) so
it's paced to real time; without it, ffmpeg pumps frames as fast as the CPU allows
and corrupts the stream.

## 3. Start TrackStudio

If you're viewing the UI from a browser on this **same machine**, you can just run:

```bash
trackstudio run -c local_test_config.json -p 8010
```

If you're viewing it from a **different machine** (VS Code Remote-SSH, SSH tunnel,
etc. — see next section for why), export the TURN env vars first:

```bash
export TURN_URL="turn:localhost:3478?transport=tcp"
export TURN_USERNAME="tsuser"
export TURN_CREDENTIAL="tspass123"
trackstudio run -c local_test_config.json -p 8010
```

Open `http://localhost:8010` in the browser.

## 4. (Remote access only) Run a TURN relay

TrackStudio's WebRTC video only uses STUN by default, which is not enough when the
browser and the server aren't on the same reachable network (e.g. this box is
headless and you're viewing the UI through a forwarded port). STUN just discovers a
NAT'd address — it doesn't relay media — so ICE negotiation stalls and no video ever
arrives, even though the page itself loads fine (HTTP/WebSocket is plain TCP and gets
forwarded; WebRTC's UDP media path does not).

The fix is a local TURN relay (`coturn`), configured to accept TCP connections on a
single port, since only TCP survives an SSH/VS Code port-forward tunnel:

```bash
sudo docker run -d --name trackstudio-turn --network host coturn/coturn \
  -n --log-file=stdout \
  -a -f -r trackstudio \
  -u tsuser:tspass123 \
  --no-cli \
  --listening-ip=127.0.0.1 \
  --relay-ip=127.0.0.1 \
  --listening-port=3478 \
  --min-port=49160 --max-port=49200
```

`--listening-ip`/`--relay-ip` must be set explicitly — without them, `coturn`
auto-detects an interface and may pick something other than `127.0.0.1` (it did on
this box, binding to a Docker bridge address instead), which neither `trackstudio`
nor a forwarded browser connection could reach.

Then, in VS Code's **Ports** panel, forward port `3478` (TCP) the same way `8010` is
already forwarded, so the browser (on your laptop) can reach `localhost:3478` too.

This only needs to be started once — it keeps running in the background across
`trackstudio` restarts. Check it's still up with:

```bash
sudo docker ps --filter name=trackstudio-turn
sudo docker logs trackstudio-turn --tail 50
```

The TURN server is already wired into the code:
- Backend: `trackstudio/core/config.py` (`TURN_URL`/`TURN_USERNAME`/`TURN_CREDENTIAL`
  env vars) and `trackstudio/core/api/webrtc.py` (adds it to the `RTCIceServer` list).
- Frontend: `web/src/services/WebRTCManager.ts` (hardcoded to match the same
  `coturn` credentials). If you ever change the TURN username/password/port, rebuild
  the frontend afterwards:

```bash
python3 build_frontend.py
```

then hard-refresh the browser (Ctrl+Shift+R) so it picks up the new bundle instead of
a cached one.

## Troubleshooting

**Page loads but no video ever appears.** Almost always a WebRTC/ICE problem, not a
camera/MediaMTX problem. Check:
- Browser DevTools Console (F12) for `🧊 Combined stream ICE connection state:` — if
  it's stuck on `checking` or goes to `failed`, the browser can't reach either the
  server directly or the TURN relay.
- Terminal running `trackstudio` for `🧊 ICE connection state changed to:` — if it
  never gets past `checking`, same story from the server side.
- If you're viewing from a different machine than the one running `trackstudio` and
  haven't set up the TURN relay (step 4), that's very likely the cause.

**Video is glitchy/corrupted (color blocks, diagonal streaks) or the stream
disconnects after a while.** Check the `trackstudio` terminal for a flood of
`RTP: PT=60: bad cseq`, `corrupted macroblock`, `decode_slice_header error`. This
happened because the synthetic test-pattern branch in
`tests/stream_camera{0,1}_ffmpeg.sh` was missing `-re`, so ffmpeg encoded and pushed
frames as fast as the CPU allowed instead of pacing to real time (15fps), overloading
the RTSP/RTP timing between MediaMTX and TrackStudio's OpenCV-based consumer — and
enough CPU load to eventually starve WebRTC's keepalives and drop the connection.
Both scripts now have `-re -f lavfi -i testsrc2=...`; if you ever edit them again,
keep it there.

**`ffprobe`/`ffmpeg` can quickly sanity-check the raw streams** independent of
TrackStudio or the browser:

```bash
ffprobe -rtsp_transport tcp -i rtsp://localhost:9554/camera0
ffprobe -rtsp_transport tcp -i rtsp://localhost:9554/camera1
```

If these show a valid `Stream #0:0: Video: h264 ...` line with no errors, MediaMTX
and the publishers are healthy and any remaining issue is in the WebRTC layer above.
