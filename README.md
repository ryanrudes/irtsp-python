# irtsp

**Typed, SI-unit, one-shared-clock sensor streams from an iPhone.**

[iRTSP](https://ryanrudes.github.io/irtsp-support/) turns an iPhone into a streaming
sensor rig: RTSP video plus side channels for fused IMU, GNSS, compass heading,
barometric altitude, ARKit 6-DOF pose, camera intrinsics, and LiDAR metric depth.
This library is the Python client. Every sample arrives as a small, frozen,
pattern-matchable dataclass in SI units, carrying two timestamps on a **single clock
anchor captured once per session** — the same anchor that drives the video's RTP/RTCP
timeline — so video and odometry align with **no time offset to estimate**. The core
is pure stdlib, Python ≥ 3.10.

## Install

| Command | What you get |
|---|---|
| `pip install irtsp` | The core client — odometry + depth channels, all record types. Zero dependencies. |
| `pip install "irtsp[numpy]"` | `DepthFrame.meters` arrays and `point_cloud()` back-projection. |
| `pip install "irtsp[discovery]"` | `irtsp.discover()` — find phones over Bonjour/mDNS (zeroconf). |
| `pip install "irtsp[video]"` | Decoded video frames + `synced()` bundles (PyAV + numpy). **Experimental.** |
| `pip install "irtsp[all]"` | Everything above. |

## Quickstart

```python
import irtsp

with irtsp.connect("192.168.1.24") as phone:   # or "ryans-iphone.local"
    for imu in phone.imu:
        print(imu.gyro, imu.accel)             # rad/s, m/s²
```

Start streaming in the iRTSP app, point `connect()` at the phone's address, iterate.
Records are regular frozen dataclasses — unpack them, stick them in a queue, feed them
to your filter.

## Pattern-match the whole odometry channel

`phone.odometry` yields every record in arrival order (including depth frames when the
depth channel is open), and every record type supports structural pattern matching:

```python
import irtsp

with irtsp.connect("192.168.1.24", depth=True) as phone:
    for rec in phone.odometry:
        match rec:
            case irtsp.IMU(gyro=g, accel=a):
                propagate(g, a, rec.host_ts)
            case irtsp.Pose(position=p, tracking=irtsp.Tracking.NORMAL):
                update(p, rec.unix_ts)
            case irtsp.GNSS(lat=lat, lon=lon) as fix if fix.h_accuracy is not None:
                fuse_gps(lat, lon, fix.h_accuracy)
            case irtsp.DepthFrame() as d:
                enqueue_depth(d)
```

Unknown record types from a newer app version arrive as `irtsp.Unknown` (with the raw
payload bytes) instead of raising — old clients keep working.

## The streams

Each property on the session (`phone.imu`, `phone.gnss`, …) is a **fresh, independent
iterator** with its own buffer — create as many as you like, they don't compete.

| Record | Stream | Rate | SI fields | Wire-native view |
|---|---|---|---|---|
| `IMU` | `phone.imu` | ≤ ~100 Hz | `gyro` rad/s · `accel` m/s² (gravity **included**; face-up at rest ≈ (0, 0, −9.81)) · `quat` (x, y, z, w), body→world | `accel_g` |
| `RawGyro` / `RawAccel` | `phone.raw_gyro` / `phone.raw_accel` | raw sensor mode | rad/s · m/s² | `accel_g` |
| `Intrinsics` | `phone.intrinsics` | on zoom/lens change, + 10 s keyframes (v2) | `fx fy cx cy` px at `width×height` · `matrix` (3×3 K) · `scaled()` · `snapshot` | — |
| `GNSS` | `phone.gnss` | ~1 Hz | `lat`/`lon` deg · `altitude` m · `speed` m/s · `h_accuracy`/`v_accuracy` m · `course_deg` | `course_rad` |
| `Altitude` | `phone.altitude` | ~1 Hz | `relative_altitude` m · `pressure` **Pa** | `pressure_kpa`, `pressure_hpa` |
| `Heading` | `phone.heading` | on-change, capped ~1 Hz, + 10 s keyframes (v2) | `true_deg` · `magnetic_deg` · `accuracy_deg` · `snapshot` | `true_rad`, `magnetic_rad` |
| `Pose` | `phone.pose` | ~60 Hz (AR mode) | `position` m (gravity-aligned world frame) · `orientation` quat · `tracking` · `discontinuity`/`relocalized`/`jump` · `gravity_tilt_deg` · `is_level()` | `gravity_tilt_rad`, `gravity_world` |
| `DepthFrame` | `phone.depth` | ≤ 30 Hz | half-float **meters**; `meters` (ndarray), `at(x, y)`, `point_cloud(K)` | — |

Unit conventions, in one breath: SI everywhere by default — the wire's g-units become
m/s² (× 9.80665) and its kPa becomes Pa at decode time, with the wire-native value
always one property away. Angles conventionally spoken in degrees (lat/lon, compass,
GNSS course) stay degrees under explicit `*_deg` names, with `*_rad` properties.
Fields CoreLocation marks invalid (negative on the wire) decode to `None`, and an
attitude-off session's zeroed quaternion slots decode to `quat=None` — no sentinel
values ever reach your code.

Every record also carries `host_ts`, `unix_ts`, `seq`, `gap`, and a `time` property
(`unix_ts` as an aware UTC `datetime`). More on the two clocks below.

## Two things ARKit will not tell you (but `Pose` will)

`tracking == NORMAL` is **not** a promise that the pose is usable. Two failure modes hide
completely behind it, and both are silent — no exception, no state change, no callback:

**The world frame can move under you.** ARKit re-anchors its map on loop closure, and it
does so without ever leaving `NORMAL`. Positions jump metres between consecutive frames.
`discontinuity` is set on every such sample — re-anchor there and never integrate across
it. `relocalized` and `jump` say *which* kind it was.

`reset` (the operator reset tracking) is different in kind, not degree: it means a **brand-new
world frame** — new origin, new yaw, new gravity — and *no transform relates it to the old one*.
Close your epoch and re-derive everything; skipping one sample and carrying on with your existing
transform will silently produce confident, wrong results. (`host_ts` stays continuous across a
reset; only the spatial frame is replaced.)

**The world frame can be tilted.** `worldAlignment = .gravity` promises world +Y is up,
but ARKit learns gravity from *motion*. Start a session with the phone sitting still and
barely move it, and the frame can settle tens of degrees off vertical — for the whole
capture, with `NORMAL` tracking throughout. Everything derived from it (ground planes,
registration, reprojection) is then wrong by that angle, and nothing in ARKit says so.

Worse, a *badly* tilted frame **never heals**: ARKit fixes its gravity early in a session and does
not revisit it. A measured frame 110° off was still 100° off after 40 s of walking with good
tracking. Only a tracking reset recovers it. And the way frames get broken is the default rig
workflow — leaving the phone face-down on the table while you set up other gear, so ARKit
initialises with no parallax. **Carry the phone while you rig, or reset before you record.**

The phone measures the true tilt against CoreMotion and sends it, because **a client
cannot compute it**: recovering it here would mean fitting a device→camera rotation from
gravity samples, and that fit is rank-deficient whenever the phone stays upright — it
absorbs the tilt and confidently reports ~0°. On-device the relationship is a known
constant, so one sample settles it.

```python
for pose in phone.pose:
    if not pose.is_level():          # nan (old app / raw mode) is NOT level
        raise SystemExit(f"ARKit frame is {pose.gravity_tilt_deg:.0f}° off gravity — "
                         "walk the phone around before capturing")
    if pose.discontinuity:
        registration.reanchor()      # world frame moved; do not integrate across this
```

`gravity_world` rebuilds gravity as a unit vector in ARKit's frame, so you can derive the
rotation that *levels* the capture rather than merely rejecting it.

The tilt is **already a robust on-device estimate — don't median it yourself.** CoreMotion's
gravity is a fusion whose accelerometer correction goes transiently wrong under hand
acceleration, so the phone rejects samples taken while it is accelerating and medians the rest.
`nan` means the phone **cannot currently vouch for a value** — raw IMU mode, an app too old to
send it, or the device in sustained motion long enough that every trustworthy sample aged out.
`is_level()` treats `nan` as **not** level: a phone that cannot vouch for its frame must not be
trusted by default.

## Discovery

```python
import irtsp                       # pip install "irtsp[discovery]"

for device in irtsp.discover():
    print(device.name, device.host, device.ports)
    # iPhone 192.168.1.24 {'video': 8554, 'imu': 8555, 'depth': 8556}

phone = irtsp.discover()[0].connect(depth=True)
```

Only devices advertising the iRTSP odometry service (`_irtsp-imu._tcp`) are returned,
so a random RTSP camera on your network won't show up. No zeroconf installed? Just
connect by IP.

## Callbacks and `run()`

If you'd rather push than pull:

```python
import irtsp

phone = irtsp.connect("192.168.1.24", reconnect=True)
phone.on(irtsp.GNSS, lambda fix: print(f"{fix.lat:.6f}, {fix.lon:.6f}"))
phone.on((irtsp.IMU, irtsp.Pose), sink.write)
phone.run()   # blocks until the session closes; Ctrl-C exits cleanly
```

Callbacks run on the reader thread, so keep them quick; a callback that raises is
logged and never kills the reader. `phone.latest(irtsp.Intrinsics, wait=2.0)` gives
you the most recent record of a type — handy for intrinsics, which the server replays
to late joiners on connect.

## asyncio

The async client is a native asyncio implementation (not a thread wrapper) with the
same shapes:

```python
import asyncio, irtsp

async def main():
    async with await irtsp.connect_async("192.168.1.24") as phone:
        async for imu in phone.imu:
            print(imu.gyro, imu.accel)

asyncio.run(main())
```

## Depth → numpy → point cloud

```python
import irtsp                       # pip install "irtsp[numpy]"

with irtsp.connect("192.168.1.24", depth=True) as phone:
    K = phone.latest(irtsp.Intrinsics, wait=2.0)      # replayed on connect

    for frame in phone.depth:
        depth = frame.meters                          # (H, W) float32, meters
        center = frame.at(frame.width // 2, frame.height // 2)  # stdlib, no numpy
        pts = frame.point_cloud(K, stride=4)          # (N, 3) float32, camera frame
```

`point_cloud()` accepts intrinsics at any resolution (the stream's intrinsics are for
the video) and rescales them to the depth map automatically. The camera frame is the
standard pinhole convention: +X right, +Y down, +Z forward. Non-finite depths are
dropped.

## Video + `synced()` — EXPERIMENTAL

With the `video` extra, the session can decode the RTSP video and hand you each frame
bundled with the odometry that belongs to it, all on one clock:

```python
import irtsp                       # pip install "irtsp[video]"

with irtsp.connect("192.168.1.24", video=True, depth=True) as phone:
    for f in phone.synced():
        f.image        # (H, W, 3) RGB uint8
        f.timestamp    # unix seconds — same axis as every record's unix_ts
        f.imu          # IMU records since the previous frame (pre-integration ready)
        f.pose         # Pose SLERP-interpolated at f.timestamp, or None
        f.depth        # nearest DepthFrame within tolerance, or None
        f.intrinsics   # latest camera matrix
        f.approx_clock # ← read the note below
```

**The honest note.** Frame times are exact only when FFmpeg exposes the RTCP
wall-clock anchor (`start_time_realtime`); iRTSP builds its RTCP Sender Reports from
the same anchor as the odometry, so when that value is available the alignment is as
good as the record timestamps. When FFmpeg *doesn't* expose it, the stream falls back
to anchoring the first frame at local receive time — alignment is then off by network
plus decode latency (typically a few tens of ms) and every frame is flagged
`approx_clock=True`. Check that flag before trusting tight sync, and if you need
guaranteed-exact video timing, consume the RTSP stream with your own RTCP-aware
pipeline as described in the
[integration guide](https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md) §4.

## How the synchronization works

At the start of each streaming session the phone captures **one anchor pair** — a
host-clock reading (`mach_absolute_time`, seconds) and the Unix wall time at the same
instant — and *every* stream derives its timestamps from it. That's the whole trick:
the streams were never on different clocks, so there is no offset to estimate and
nothing to cross-correlate.

Every record carries both axes:

- **`host_ts`** — seconds on the phone's monotonic host clock. The same axis as the
  video/audio presentation timestamps, CoreMotion, ARKit, and the depth frames.
  Cleanest for intra-session alignment; not comparable across reboots.
- **`unix_ts`** — wall-clock seconds, `unix_ts = wall_anchor + (host_ts − host_anchor)`.
  The same axis as the video's RTCP Sender-Report NTP timeline, and comparable across
  machines.

The handshake ships both anchors; `phone.clock` is a `StreamClock` that converts
either way (`to_unix()` / `to_host()`). Because the anchor is frozen at session start,
no mid-session NTP adjustment will ever warp your timeline. The full derivation —
64-byte record layout, depth framing, RTP→wall-time math — is in the
[integration guide](https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md).

## Reliability notes

- **Slow consumers can't stall capture.** Each iterator/callback subscription has its
  own bounded buffer (default 8192 records); a consumer that falls behind drops its
  *own* oldest records and never affects the reader or other consumers. The count is
  on `stream.dropped`.
- **Nothing is lost silently.** Each channel carries a wire sequence number; if
  records were lost upstream (or dropped before your consumer attached), the next
  record's `gap` field says how many went missing right before it (`gap=0` means
  none).
- **Reconnects.** By default a dropped connection closes the session (iterators end,
  `run()` returns). With `connect(..., reconnect=True)` the client redials with
  backoff and re-reads the handshake — picking up fresh clock anchors if the phone
  restarted its stream.
- **Bad bytes fail loudly.** Connecting to a non-iRTSP port or desyncing raises
  `irtsp.ProtocolError` rather than yielding garbage.

## Links

- iRTSP app + support: <https://ryanrudes.github.io/irtsp-support/>
- Wire protocol & synchronization spec: [INTEGRATION.md](https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md)

## License

MIT — see [LICENSE](LICENSE).
