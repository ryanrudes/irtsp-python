"""End-to-end tests for :mod:`irtsp.session` against ``tests/mockserver.MockPhone``.

Every test speaks the real wire protocol over loopback TCP: a u32-LE
length-prefixed JSON handshake, then 64-byte odometry records / length-prefixed
depth frames, exactly as the iRTSP app's servers frame them.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):  # import irtsp + mockserver uninstalled
    if _p not in sys.path:
        sys.path.insert(0, _p)

import irtsp
from mockserver import MockPhone

G = irtsp.STANDARD_GRAVITY


# --------------------------------------------------------------------- helpers


def wait_until(pred, timeout: float = 3.0, interval: float = 0.005) -> bool:
    """Poll ``pred`` until true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def collect(stream, n: int, timeout: float = 3.0) -> list:
    """Take the next ``n`` records off a (blocking) stream, with a deadline."""
    out: list = []

    def run() -> None:
        for rec in stream:
            out.append(rec)
            if len(out) >= n:
                return

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    assert len(out) >= n, f"collected only {len(out)}/{n} records in {timeout}s"
    return out[:n]


def drain(stream, n: int, timeout: float = 3.0) -> list:
    """Accumulate ``n`` records via repeated non-blocking ``pop_all`` calls."""
    got: list = []

    def step() -> bool:
        got.extend(stream.pop_all())
        return len(got) >= n

    assert wait_until(step, timeout), f"drained only {len(got)}/{n} records"
    return got


# -------------------------------------------------------------------- fixtures


@pytest.fixture
def phone():
    server = MockPhone().start()
    yield server
    server.close()


@pytest.fixture
def session(phone):
    s = irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0)
    yield s
    s.close()


# ------------------------------------------------------------------- handshake


def test_handshake_exposed(phone, session):
    info = session.info
    assert info is not None
    assert info.protocol == "irtsp-imu"
    assert info.version == 1
    assert info.record_bytes == 64
    assert info.streams == phone.streams
    assert info.video_url == phone.video_url
    assert info.video_codec == "h264"
    assert info.video_clock_rate == 90000
    assert info.rate_hz == pytest.approx(200.0)
    assert info.attitude_enabled is True
    # everything the server sent is preserved verbatim
    assert info.raw["endianness"] == "little"
    assert info.raw["record_types"]["pose"] == 9


def test_clock_anchors(phone, session):
    clock = session.clock
    assert clock.host_anchor == pytest.approx(phone.host_anchor)
    assert clock.wall_anchor == pytest.approx(phone.wall_anchor)
    assert clock.to_unix(phone.host_anchor + 2.5) == pytest.approx(phone.wall_anchor + 2.5)
    assert clock.to_host(phone.wall_anchor + 2.5) == pytest.approx(phone.host_anchor + 2.5)


# ------------------------------------------------------------------- odometry


def test_imu_iteration_decodes_si_units(phone, session):
    stream = session.imu
    phone.emit_imu(
        seq=1, host_ts=1001.0,
        gyro=(0.5, -0.25, 1.5), accel_g=(0.0, 0.0, -1.0), quat=(0.0, 0.0, 0.0, 1.0),
    )
    phone.emit_imu(seq=2, gyro=(0.0, 0.0, 0.0), accel_g=(1.0, 0.0, 0.0))
    first, second = collect(stream, 2)
    assert isinstance(first, irtsp.IMU)
    assert first.seq == 1
    assert first.host_ts == pytest.approx(1001.0)
    assert first.unix_ts == pytest.approx(phone.unix_from_host(1001.0))
    assert session.clock.to_unix(first.host_ts) == pytest.approx(first.unix_ts)
    assert first.gyro == pytest.approx((0.5, -0.25, 1.5))
    assert first.accel == pytest.approx((0.0, 0.0, -G))  # wire g -> SI m/s²
    assert first.accel_g.z == pytest.approx(-1.0)
    assert first.quat == irtsp.Quat(0.0, 0.0, 0.0, 1.0)
    assert second.seq == 2
    assert second.accel.x == pytest.approx(G)


def test_attitude_off_quat_is_none(phone, session):
    stream = session.imu
    phone.emit_imu(seq=1, quat=None)  # zeroed quat slots on the wire
    (rec,) = collect(stream, 1)
    assert rec.quat is None


def test_raw_sensor_records(phone, session):
    stream = session.stream(irtsp.RawGyro, irtsp.RawAccel)
    phone.emit_raw_gyro(seq=1, gyro=(0.125, -0.5, 2.0))
    phone.emit_raw_accel(seq=2, accel_g=(0.0, 1.0, 0.0))
    gyro, accel = collect(stream, 2)
    assert isinstance(gyro, irtsp.RawGyro)
    assert gyro.gyro == pytest.approx((0.125, -0.5, 2.0))
    assert isinstance(accel, irtsp.RawAccel)
    assert accel.accel.y == pytest.approx(G)


def test_two_subscribers_both_get_every_record(phone, session):
    a = session.imu
    b = session.imu
    for i in (1, 2, 3):
        phone.emit_imu(seq=i)
    got_a = collect(a, 3)
    got_b = collect(b, 3)
    assert [r.seq for r in got_a] == [1, 2, 3]
    assert [r.seq for r in got_b] == [1, 2, 3]  # independent buffers, no stealing


def test_multi_type_stream_filters(phone, session):
    mixed = session.stream(irtsp.GNSS, irtsp.Pose)
    phone.emit_imu(seq=1)  # must NOT appear on `mixed`
    phone.emit_gnss(
        seq=2, lat=40.75, lon=-73.98, altitude=10.0,
        h_acc=5.0, v_acc=-1.0, speed=-1.0, course=-1.0, speed_acc=-1.0,
    )
    phone.emit_pose(seq=3, position=(1.0, 2.0, 3.0), quat=(0.0, 0.0, 0.0, 1.0), tracking=2)
    phone.emit_imu(seq=4)
    fix, pose = collect(mixed, 2)
    assert isinstance(fix, irtsp.GNSS)
    assert fix.lat == pytest.approx(40.75)  # f64 on the wire, so exact-ish
    assert fix.lon == pytest.approx(-73.98)
    assert fix.h_accuracy == pytest.approx(5.0)
    # CoreLocation negative sentinels -> None
    assert fix.v_accuracy is None
    assert fix.speed is None
    assert fix.course_deg is None
    assert fix.speed_accuracy is None
    assert isinstance(pose, irtsp.Pose)
    assert pose.position == pytest.approx((1.0, 2.0, 3.0))
    assert pose.tracking is irtsp.Tracking.NORMAL
    # nothing else ever lands on the filtered stream
    assert wait_until(lambda: (r := session.latest(irtsp.IMU)) is not None and r.seq == 4)
    time.sleep(0.05)
    assert mixed.pop_all() == []


def test_pop_all_drains_without_blocking(phone, session):
    stream = session.altitude
    phone.emit_altitude(seq=1, relative_altitude=1.5, pressure_kpa=101.325)
    phone.emit_altitude(seq=2, relative_altitude=2.5, pressure_kpa=100.0)
    got = drain(stream, 2)
    assert [r.seq for r in got] == [1, 2]
    assert got[0].relative_altitude == pytest.approx(1.5)
    assert got[0].pressure == pytest.approx(101325.0, rel=1e-6)  # wire kPa -> SI Pa
    assert got[0].pressure_kpa == pytest.approx(101.325, rel=1e-6)
    assert got[1].pressure == pytest.approx(100_000.0, rel=1e-6)
    assert stream.pop_all() == []  # already drained; still non-blocking


def test_heading_sentinels(phone, session):
    stream = session.heading
    phone.emit_heading(seq=1, true_deg=-1.0, magnetic_deg=123.5, accuracy_deg=-1.0)
    (h,) = collect(stream, 1)
    assert h.true_deg is None
    assert h.accuracy_deg is None
    assert h.magnetic_deg == pytest.approx(123.5)


def test_unknown_record_type_preserved(phone, session):
    stream = session.stream(irtsp.Unknown)
    phone.emit_unknown(seq=7, type_id=42, payload=b"\xab" * 8)
    (rec,) = collect(stream, 1)
    assert rec.type_id == 42
    assert rec.seq == 7
    assert len(rec.payload) == 40  # the whole 24..64 slot
    assert rec.payload[:8] == b"\xab" * 8


# --------------------------------------------------------------- gap detection


def test_gap_detection_on_skipped_seq(phone, session):
    stream = session.imu
    phone.emit_imu(seq=1)
    phone.emit_imu(seq=2)
    phone.emit_imu(seq=5)  # 3 and 4 were lost
    r1, r2, r5 = collect(stream, 3)
    assert (r1.gap, r2.gap, r5.gap) == (0, 0, 2)


def test_gap_detection_across_wraparound(phone, session):
    stream = session.imu
    phone.emit_imu(seq=65533)
    phone.emit_imu(seq=65535)  # 65534 lost
    phone.emit_imu(seq=0)      # 65535 -> 0 is contiguous, NOT a gap
    phone.emit_imu(seq=2)      # 1 lost
    a, b, c, d = collect(stream, 4)
    assert a.gap == 0  # first record can't have a known gap
    assert b.gap == 1
    assert c.gap == 0  # clean wraparound
    assert d.gap == 1


# ---------------------------------------------------------------------- latest


def test_latest_returns_most_recent(phone, session):
    assert session.latest(irtsp.Intrinsics) is None
    phone.emit_intrinsics(seq=1, fx=1000.0, fy=1010.0, cx=960.0, cy=540.0,
                          width=1920, height=1080)
    assert wait_until(lambda: session.latest(irtsp.Intrinsics) is not None)
    k = session.latest(irtsp.Intrinsics)
    assert (k.fx, k.fy, k.cx, k.cy) == (1000.0, 1010.0, 960.0, 540.0)
    assert (k.width, k.height) == (1920, 1080)
    phone.emit_intrinsics(seq=2, fx=500.0, fy=505.0, cx=480.0, cy=270.0,
                          width=960, height=540)
    assert wait_until(lambda: session.latest(irtsp.Intrinsics).seq == 2)
    assert session.latest(irtsp.Intrinsics).fx == pytest.approx(500.0)
    assert session.latest(irtsp.GNSS) is None  # never sent


def test_latest_wait_blocks_until_arrival(phone, session):
    timer = threading.Timer(
        0.15,
        lambda: phone.emit_heading(seq=1, true_deg=90.0, magnetic_deg=88.0, accuracy_deg=5.0),
    )
    timer.start()
    try:
        t0 = time.monotonic()
        rec = session.latest(irtsp.Heading, wait=3.0)
        elapsed = time.monotonic() - t0
        assert rec is not None
        assert rec.true_deg == pytest.approx(90.0)
        assert elapsed < 2.0  # returned on arrival, not at the deadline
    finally:
        timer.cancel()
    # and times out cleanly when nothing arrives
    t0 = time.monotonic()
    assert session.latest(irtsp.Pose, wait=0.2) is None
    assert 0.15 <= time.monotonic() - t0 < 1.5


# ------------------------------------------------------------------- callbacks


def test_callbacks_and_raising_callback_does_not_kill_reader(phone, session):
    good: list = []
    bad_calls: list = []

    def bad(rec):
        bad_calls.append(rec)
        raise RuntimeError("boom — must not kill the reader")

    session.on(irtsp.IMU, bad)  # registered first, raises every time
    session.on(irtsp.IMU, good.append)
    phone.emit_imu(seq=1)
    phone.emit_imu(seq=2)
    assert wait_until(lambda: len(good) == 2)
    assert [r.seq for r in good] == [1, 2]
    assert len(bad_calls) == 2  # kept being invoked despite raising
    # the reader thread survived: records still flow to new consumers
    stream = session.imu
    phone.emit_imu(seq=3)
    (rec,) = collect(stream, 1)
    assert rec.seq == 3
    assert not session.closed


def test_callback_tuple_filter(phone, session):
    seen: list = []
    session.on((irtsp.GNSS, irtsp.Altitude), seen.append)
    phone.emit_imu(seq=1)
    phone.emit_gnss(seq=2)
    phone.emit_altitude(seq=3)
    assert wait_until(lambda: len(seen) == 2)
    assert isinstance(seen[0], irtsp.GNSS)
    assert isinstance(seen[1], irtsp.Altitude)
    time.sleep(0.05)
    assert len(seen) == 2  # the IMU record never matched


# ----------------------------------------------------------------------- depth


def test_depth_channel_end_to_end(phone):
    with irtsp.connect(
        "127.0.0.1", imu_port=phone.imu_port,
        depth=True, depth_port=phone.depth_port, timeout=2.0,
    ) as session:
        assert session.depth_info is not None
        assert session.depth_info.protocol == "irtsp-depth"
        assert session.depth_info.raw["pixel_format"] == "depth_float16"
        assert session.depth_info.clock.host_anchor == pytest.approx(phone.host_anchor)
        assert session.depth_info.clock.wall_anchor == pytest.approx(phone.wall_anchor)

        stream = session.depth
        meters = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # exactly representable in fp16
        phone.emit_depth(seq=1, host_ts=1002.0, width=3, height=2, samples=meters)
        phone.emit_depth(seq=3, width=3, height=2, samples=[7.0] * 6)  # 2 was lost

        f1, f2 = collect(stream, 2)
        assert isinstance(f1, irtsp.DepthFrame)
        assert (f1.width, f1.height) == (3, 2)
        assert f1.seq == 1
        assert f1.host_ts == pytest.approx(1002.0)
        assert f1.unix_ts == pytest.approx(phone.unix_from_host(1002.0))
        arr = f1.meters
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32
        assert np.array_equal(arr, np.array(meters, dtype=np.float32).reshape(2, 3))
        assert f1.at(2, 1) == pytest.approx(3.0)  # dependency-free pixel access
        assert f2.gap == 1  # depth channel has its own gap tracking


# ----------------------------------------------- protocol v2: state channels


def test_handshake_v2_accepted_and_exposed(request):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        info = session.info
        assert info.version == 2
        assert info.emission["imu"] == "continuous"
        assert info.emission["heading"] == "state"
        assert info.state_channels["keyframe_interval_s"] == 10
        # a v2 session streams exactly like v1
        stream = session.imu
        phone.emit_imu(seq=1)
        (rec,) = collect(stream, 1)
        assert rec.seq == 1


def test_snapshot_flag_surfaces_on_state_channels(phone, session):
    stream = session.stream(irtsp.Intrinsics, irtsp.Heading)
    phone.emit_intrinsics(seq=1, snapshot=True)   # connect-snapshot / keyframe
    phone.emit_heading(seq=2, snapshot=False)     # a real change event
    phone.emit_heading(seq=3, snapshot=True)      # 10 s keyframe
    k, h_change, h_key = collect(stream, 3)
    assert isinstance(k, irtsp.Intrinsics) and k.snapshot is True
    assert h_change.snapshot is False
    assert h_key.snapshot is True


# --------------------------------------------- protocol v2.1: camera format channel


def test_handshake_v2_1_accepted_without_version_warning(request, caplog):
    import logging

    phone = MockPhone(version=2, revision=1).start()
    request.addfinalizer(phone.close)
    with caplog.at_level(logging.WARNING, logger="irtsp"):
        with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
            info = session.info
            # `version` is UNCHANGED (v2 consumers key off it); revision distinguishes 2.1
            assert info.version == 2
            assert info.revision == 1
            assert info.raw["record_types"]["format"] == 11
            assert info.emission["format"] == "state"
            assert info.streams["format"] is True
            assert info.format_channel  # the descriptive object is surfaced
            assert "readout_time" in info.format_channel
    # an unrecognized revision must NOT trip the version warning
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_format_stream_and_latest_end_to_end(request):
    phone = MockPhone(version=2, revision=1).start()
    request.addfinalizer(phone.close)
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        stream = session.format
        phone.emit_format(
            seq=1, snapshot=True, format_id=0xABCD1234, width=1920, height=1440,
            fps=30.0, readout=0.03125, camera=1, capture_path=0,
            binned=True, cropped=False,
        )
        (fmt,) = collect(stream, 1)
        assert isinstance(fmt, irtsp.CameraFormat)
        assert fmt.format_id == 0xABCD1234
        assert (fmt.width, fmt.height) == (1920, 1440)
        assert fmt.fps == pytest.approx(30.0)
        assert fmt.readout_time_s == pytest.approx(0.03125)
        assert fmt.readout_provenance is irtsp.ReadoutProvenance.PROBED
        assert fmt.camera is irtsp.Camera.BACK_WIDE
        assert fmt.capture_path is irtsp.CapturePath.AVCAPTURE
        assert fmt.binned is True and fmt.cropped is False
        assert fmt.snapshot is True
        # and it lands in latest() like any other state channel
        assert wait_until(lambda: session.latest(irtsp.CameraFormat) is not None)
        assert session.latest(irtsp.CameraFormat).format_id == 0xABCD1234


def test_format_absent_readout_end_to_end(request):
    phone = MockPhone(version=2, revision=1).start()
    request.addfinalizer(phone.close)
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        stream = session.format
        phone.emit_format(seq=1, readout=None)  # NaN + provenance absent
        (fmt,) = collect(stream, 1)
        assert fmt.readout_time_s is None
        assert fmt.readout_provenance is irtsp.ReadoutProvenance.ABSENT


def test_v2_0_handshake_without_format_keys_still_works(request):
    # A plain v2.0 server (revision 0): no format channel, but everything else works
    # exactly as before — the additive 2.1 keys are simply absent.
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        assert session.info.version == 2
        assert session.info.revision == 0
        assert "format" not in session.info.raw.get("record_types", {})
        assert session.info.format_channel == {}
        phone.emit_imu(seq=1)
        (rec,) = collect(session.imu, 1)
        assert rec.seq == 1


# ------------------------------------------- protocol v2: depth compression


def depth_session(phone, **kwargs):
    return irtsp.connect(
        "127.0.0.1", imu_port=phone.imu_port,
        depth=True, depth_port=phone.depth_port, timeout=2.0, **kwargs,
    )


def test_depth_compression_opt_in_sends_control_message(request):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    with depth_session(phone, depth_compression="zlib") as session:
        # the exact wire bytes: [u32 LE length][UTF-8 JSON]
        assert phone.recv_depth_control() == {"compression": "zlib"}
        assert session.depth_codec == "zlib"

        # compressed frames decode; a raw frame after opt-in decodes too
        # (incompressible fallback — decoding follows per-frame flags)
        stream = session.depth
        meters = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        phone.emit_depth(seq=1, width=3, height=2, samples=meters, codec="zlib")
        phone.emit_depth(seq=2, width=3, height=2, samples=meters)  # raw fallback
        f1, f2 = collect(stream, 2)
        assert f1.at(2, 1) == pytest.approx(3.0)
        assert f1.data == f2.data


def test_depth_compression_auto_prefers_lzfse_when_available(request, monkeypatch):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    monkeypatch.setattr("irtsp.session._lzfse_available", lambda: True)
    with depth_session(phone) as session:  # depth_compression defaults to "auto"
        assert phone.recv_depth_control() == {"compression": "lzfse"}
        assert session.depth_codec == "lzfse"


def test_depth_compression_auto_falls_back_to_zlib(request, monkeypatch):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    monkeypatch.setattr("irtsp.session._lzfse_available", lambda: False)
    with depth_session(phone) as session:
        assert phone.recv_depth_control() == {"compression": "zlib"}
        assert session.depth_codec == "zlib"


def test_depth_compression_none_sends_nothing(request):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    with depth_session(phone, depth_compression=None) as session:
        assert session.depth_codec is None
        assert phone.recv_depth_control(timeout=0.3) is None


def test_depth_compression_never_offered_to_v1_server(phone):
    # the fixture phone is a v1 server: no `compression` key in its handshake,
    # so the client must not send a control message no matter what was asked
    with depth_session(phone, depth_compression="zlib") as session:
        assert session.depth_codec is None
        assert phone.recv_depth_control(timeout=0.3) is None
        # and the channel still works, raw, exactly like v1
        stream = session.depth
        phone.emit_depth(seq=1, width=2, height=1, samples=[1.0, 2.0])
        (frame,) = collect(stream, 1)
        assert frame.at(1, 0) == 2.0


def test_depth_compression_invalid_value_raises(phone):
    with pytest.raises(ValueError, match="depth_compression"):
        depth_session(phone, depth_compression="gzip")


def test_depth_compression_renegotiated_after_reconnect(request):
    phone = MockPhone(version=2).start()
    request.addfinalizer(phone.close)
    with depth_session(phone, depth_compression="zlib", reconnect=True) as session:
        assert phone.recv_depth_control() == {"compression": "zlib"}
        phone.close_depth_clients()  # codec choice is per-connection
        assert phone.wait_for_depth_client(3.0)
        assert phone.recv_depth_control() == {"compression": "zlib"}
        assert session.depth_codec == "zlib"


# ------------------------------------------------------------------- buffering


def test_tiny_buffer_drops_oldest_and_counts(phone, session):
    stream = session.stream(irtsp.IMU, buffer=4)
    for i in range(1, 11):
        phone.emit_imu(seq=i)
    assert wait_until(lambda: stream.dropped == 6), f"dropped={stream.dropped}"
    assert [r.seq for r in stream.pop_all()] == [7, 8, 9, 10]  # newest survive
    assert stream.dropped == 6
    # an unconstrained sibling subscribed later is unaffected
    other = session.imu
    phone.emit_imu(seq=11)
    (rec,) = collect(other, 1)
    assert rec.seq == 11
    assert other.dropped == 0


# -------------------------------------------------------------------- teardown


def test_close_unblocks_iteration_promptly(phone, session):
    stream = session.imu
    result: dict = {}

    def block():
        try:
            next(stream)
            result["outcome"] = "record"
        except StopIteration:
            result["outcome"] = "stopped"

    t = threading.Thread(target=block, daemon=True)
    t.start()
    time.sleep(0.2)  # let it park inside __next__
    t0 = time.monotonic()
    session.close()
    t.join(2.0)
    assert not t.is_alive(), "iterator did not unblock after close()"
    assert time.monotonic() - t0 < 1.5
    assert result["outcome"] == "stopped"
    assert session.closed


def test_stream_close_ends_only_that_stream(phone, session):
    stream = session.imu
    other = session.imu
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)
    phone.emit_imu(seq=1)
    (rec,) = collect(other, 1)  # sibling unaffected
    assert rec.seq == 1


def test_server_close_ends_session(phone, session):
    stream = session.imu
    phone.emit_imu(seq=1)
    (rec,) = collect(stream, 1)
    assert rec.seq == 1
    phone.close_odometry_clients()  # phone stops streaming (reconnect=False)
    assert wait_until(lambda: session.closed, timeout=3.0)
    with pytest.raises(StopIteration):
        next(stream)


def test_context_manager(phone):
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        assert not session.closed
        assert session.info is not None
        phone.emit_imu(seq=1)
        assert session.latest(irtsp.IMU, wait=2.0) is not None
    assert session.closed


# ----------------------------------------------------------------------- video


def test_video_url_rebuilt_against_dialed_host(phone):
    # the phone advertises its own mDNS name; we dialed loopback
    with irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0) as session:
        assert session.info.video_url == "rtsp://ryans-iphone.local:8554/live"
        assert session.video_url == "rtsp://127.0.0.1:8554/live"


def test_video_url_auth_and_custom_port(phone):
    phone.video_url = "rtsp://ryans-iphone.local:9554/cam"  # handshake built at accept
    with irtsp.connect(
        "127.0.0.1", imu_port=phone.imu_port,
        video_auth=("alice", "s3cret"), timeout=2.0,
    ) as session:
        assert session.video_url == "rtsp://alice:s3cret@127.0.0.1:9554/cam"


def test_video_url_override_wins(phone):
    with irtsp.connect(
        "127.0.0.1", imu_port=phone.imu_port,
        video_url="rtsp://elsewhere:1234/x", timeout=2.0,
    ) as session:
        assert session.video_url == "rtsp://elsewhere:1234/x"
