"""Golden-byte tests for irtsp.wire against the Swift wire format.

Every record here is hand-packed with struct.pack_into following the offsets in
IMUWireFormat.swift / DepthStreamServer.swift exactly:

64-byte odometry record (little-endian):
    0   u8    type   (1=imu, 2=gyro, 3=accel, 5=intrinsics, 6=gnss, 7=altitude,
                      8=heading, 9=pose)
    1   u8    flags  (0)
    2   u16   seq
    4   u32   reserved (0)
    8   f64   host_ts
    16  f64   unix_ts
    24..64    type-specific payload (f32/f64 slots, see each test)

Depth frame (after the u32 length prefix, which decode_depth_frame never sees):
    0   u8    type (=10)      1  u8  flags (bit0: float16)
    2   u16   seq             4  u32 reserved
    8   f64   host_ts         16 f64 unix_ts
    24  u16   width           26 u16 height
    28  u8    bytesPerPixel   29..31 pad
    32.. row-major width*height IEEE-754 half floats (meters)
"""

from __future__ import annotations

import json
import math
import socket
import struct

import pytest

from irtsp.records import (
    GNSS,
    IMU,
    STANDARD_GRAVITY,
    Altitude,
    DepthFrame,
    Heading,
    Intrinsics,
    Pose,
    Quat,
    RawAccel,
    RawGyro,
    Tracking,
    Unknown,
    Vec3,
)
from irtsp.wire import (
    RECORD_SIZE,
    ConnectionClosed,
    ProtocolError,
    RecordType,
    decode_depth_frame,
    decode_record,
    read_exact,
    recv_handshake,
    recv_length_prefixed,
)

HOST_TS = 1234.5625  # exactly representable f64
UNIX_TS = 1700000000.5
SEQ = 0xBEEF


def f32(v: float) -> float:
    """Round-trip a Python float through IEEE-754 binary32, like the wire does."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def make_header(type_id: int, *, seq: int = SEQ, host_ts: float = HOST_TS,
                unix_ts: float = UNIX_TS, flags: int = 0) -> bytearray:
    """A fresh 64-byte record with only the shared 24-byte header filled."""
    buf = bytearray(RECORD_SIZE)
    struct.pack_into("<B", buf, 0, type_id)
    struct.pack_into("<B", buf, 1, flags)
    struct.pack_into("<H", buf, 2, seq)
    struct.pack_into("<I", buf, 4, 0)  # reserved
    struct.pack_into("<d", buf, 8, host_ts)
    struct.pack_into("<d", buf, 16, unix_ts)
    return buf


def assert_common(rec, *, seq: int = SEQ, host_ts: float = HOST_TS,
                  unix_ts: float = UNIX_TS) -> None:
    assert rec.seq == seq
    assert rec.host_ts == host_ts
    assert rec.unix_ts == unix_ts
    assert rec.gap == 0


# --------------------------------------------------------------------------- #
# Type 1: fused IMU  (gyro@24 rad/s, accel@36 in g, quat xyzw@48)
# --------------------------------------------------------------------------- #


def test_imu_record_golden() -> None:
    buf = make_header(1)
    struct.pack_into("<3f", buf, 24, 0.5, -0.25, 1.5)        # gyro rad/s
    struct.pack_into("<3f", buf, 36, 0.0, 0.0, -1.0)         # accel in g
    struct.pack_into("<4f", buf, 48, 0.0, 0.0, 0.0, 1.0)     # identity quat xyzw

    rec = decode_record(bytes(buf))
    assert type(rec) is IMU
    assert_common(rec)
    assert rec.gyro == Vec3(0.5, -0.25, 1.5)
    # wire g -> SI m/s²: face-up rest reads (0, 0, -1) g == (0, 0, -9.80665) m/s²
    assert rec.accel == Vec3(0.0, 0.0, -STANDARD_GRAVITY)
    assert rec.quat == Quat(0.0, 0.0, 0.0, 1.0)


def test_imu_record_g_to_si_conversion_arbitrary_values() -> None:
    ax, ay, az = 0.125, -2.0, 1.0625  # dyadic -> exact in f32
    buf = make_header(1)
    struct.pack_into("<3f", buf, 36, ax, ay, az)
    struct.pack_into("<4f", buf, 48, 0.0, 0.0, 0.0, 1.0)

    rec = decode_record(bytes(buf))
    assert isinstance(rec, IMU)
    assert rec.accel == Vec3(ax * STANDARD_GRAVITY, ay * STANDARD_GRAVITY,
                             az * STANDARD_GRAVITY)
    # and the native-unit property recovers the wire values
    assert rec.accel_g.x == pytest.approx(ax)
    assert rec.accel_g.y == pytest.approx(ay)
    assert rec.accel_g.z == pytest.approx(az)


def test_imu_record_zeroed_quat_decodes_to_none() -> None:
    # Attitude-off sessions leave the quat slots zeroed (48..64 all zero).
    buf = make_header(1)
    struct.pack_into("<3f", buf, 24, 0.5, 0.5, 0.5)
    struct.pack_into("<3f", buf, 36, 0.0, 0.0, -1.0)
    # offsets 48..64 stay zero
    rec = decode_record(bytes(buf))
    assert isinstance(rec, IMU)
    assert rec.quat is None


def test_imu_record_nonidentity_quat() -> None:
    s = math.sqrt(0.5)  # 90° about Z: (0, 0, sin45, cos45)
    buf = make_header(1)
    struct.pack_into("<4f", buf, 48, 0.0, 0.0, s, s)
    rec = decode_record(bytes(buf))
    assert isinstance(rec, IMU)
    assert rec.quat == Quat(0.0, 0.0, f32(s), f32(s))


# --------------------------------------------------------------------------- #
# Type 2: raw gyro  (xyz f32 @24, rad/s — no unit conversion)
# --------------------------------------------------------------------------- #


def test_raw_gyro_record_golden() -> None:
    buf = make_header(2, seq=7)
    struct.pack_into("<3f", buf, 24, -0.5, 2.25, 0.0078125)
    rec = decode_record(bytes(buf))
    assert type(rec) is RawGyro
    assert_common(rec, seq=7)
    assert rec.gyro == Vec3(-0.5, 2.25, 0.0078125)


# --------------------------------------------------------------------------- #
# Type 3: raw accel  (xyz f32 in the ACCEL slots @36, wire unit is g)
# --------------------------------------------------------------------------- #


def test_raw_accel_record_golden() -> None:
    buf = make_header(3)
    struct.pack_into("<3f", buf, 36, 1.0, -0.5, 0.25)
    # the gyro slots @24 are unused for type 3 — poison them to prove the
    # decoder reads offset 36, exactly like IMUWireFormat.swift writes it
    struct.pack_into("<3f", buf, 24, 99.0, 99.0, 99.0)

    rec = decode_record(bytes(buf))
    assert type(rec) is RawAccel
    assert_common(rec)
    assert rec.accel == Vec3(1.0 * STANDARD_GRAVITY, -0.5 * STANDARD_GRAVITY,
                             0.25 * STANDARD_GRAVITY)
    assert rec.accel_g.x == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Type 5: intrinsics  (fx@24, fy@28, ox@32, oy@36, width@40, height@44, all f32)
# --------------------------------------------------------------------------- #


def test_intrinsics_record_golden() -> None:
    buf = make_header(5)
    struct.pack_into("<3f", buf, 24, 1000.5, 1001.25, 640.25)   # fx, fy, ox
    struct.pack_into("<3f", buf, 36, 360.125, 1280.0, 720.0)    # oy, width, height
    # quat slots 48..64 are 0,0,0,0 per the Swift encoder — already zero

    rec = decode_record(bytes(buf))
    assert type(rec) is Intrinsics
    assert_common(rec)
    assert rec.fx == 1000.5
    assert rec.fy == 1001.25
    assert rec.cx == 640.25
    assert rec.cy == 360.125
    assert rec.width == 1280 and isinstance(rec.width, int)
    assert rec.height == 720 and isinstance(rec.height, int)


# --------------------------------------------------------------------------- #
# Type 6: GNSS  (lat f64@24, lon f64@32, then f32 alt@40, hAcc@44, vAcc@48,
#                speed@52, course@56, speedAcc@60; negatives = invalid)
# --------------------------------------------------------------------------- #


def _gnss_buf(*, alt=12.5, h_acc=3.5, v_acc=4.25, speed=1.25, course=90.0,
              speed_acc=0.5) -> bytes:
    buf = make_header(6)
    struct.pack_into("<d", buf, 24, 37.7749)
    struct.pack_into("<d", buf, 32, -122.4194)
    struct.pack_into("<f", buf, 40, alt)
    struct.pack_into("<f", buf, 44, h_acc)
    struct.pack_into("<f", buf, 48, v_acc)
    struct.pack_into("<f", buf, 52, speed)
    struct.pack_into("<f", buf, 56, course)
    struct.pack_into("<f", buf, 60, speed_acc)
    return bytes(buf)


def test_gnss_record_golden_all_valid() -> None:
    rec = decode_record(_gnss_buf())
    assert type(rec) is GNSS
    assert_common(rec)
    assert rec.lat == 37.7749  # f64: exact round-trip
    assert rec.lon == -122.4194
    assert rec.altitude == 12.5
    assert rec.h_accuracy == 3.5
    assert rec.v_accuracy == 4.25
    assert rec.speed == 1.25
    assert rec.course_deg == 90.0
    assert rec.speed_accuracy == 0.5


def test_gnss_record_negative_sentinels_become_none() -> None:
    rec = decode_record(_gnss_buf(h_acc=-1.0, v_acc=-1.0, speed=-1.0,
                                  course=-1.0, speed_acc=-1.0))
    assert isinstance(rec, GNSS)
    assert rec.h_accuracy is None
    assert rec.v_accuracy is None
    assert rec.speed is None
    assert rec.course_deg is None
    assert rec.speed_accuracy is None
    assert rec.course_rad is None
    # lat/lon/altitude are not sentinel-coded
    assert rec.lat == 37.7749 and rec.lon == -122.4194 and rec.altitude == 12.5


def test_gnss_zero_speed_is_valid_not_none() -> None:
    rec = decode_record(_gnss_buf(speed=0.0, course=0.0))
    assert isinstance(rec, GNSS)
    assert rec.speed == 0.0  # 0 is a legal value; only negatives are invalid
    assert rec.course_deg == 0.0


# --------------------------------------------------------------------------- #
# Type 7: altitude  (relativeAltitude f32@24 m, pressure f32@28 in kPa -> Pa)
# --------------------------------------------------------------------------- #


def test_altitude_record_golden_kpa_to_pa() -> None:
    buf = make_header(7)
    struct.pack_into("<2f", buf, 24, -3.75, 101.25)  # m, kPa
    rec = decode_record(bytes(buf))
    assert type(rec) is Altitude
    assert_common(rec)
    assert rec.relative_altitude == -3.75  # negative here is real data, not a sentinel
    assert rec.pressure == 101250.0        # kPa -> SI Pa
    assert rec.pressure_kpa == 101.25
    assert rec.pressure_hpa == 1012.5


# --------------------------------------------------------------------------- #
# Type 8: heading  (true@24, magnetic@28, accuracy@32, all f32 degrees;
#                   negative true/accuracy = invalid)
# --------------------------------------------------------------------------- #


def test_heading_record_golden() -> None:
    buf = make_header(8)
    struct.pack_into("<3f", buf, 24, 350.0, 337.5, 15.0)
    rec = decode_record(bytes(buf))
    assert type(rec) is Heading
    assert_common(rec)
    assert rec.true_deg == 350.0
    assert rec.magnetic_deg == 337.5
    assert rec.accuracy_deg == 15.0
    assert rec.magnetic_rad == pytest.approx(math.radians(337.5))
    assert rec.true_rad == pytest.approx(math.radians(350.0))


def test_heading_record_negative_sentinels() -> None:
    buf = make_header(8)
    struct.pack_into("<3f", buf, 24, -1.0, 337.5, -1.0)
    rec = decode_record(bytes(buf))
    assert isinstance(rec, Heading)
    assert rec.true_deg is None
    assert rec.accuracy_deg is None
    assert rec.true_rad is None
    assert rec.magnetic_deg == 337.5  # magnetic has no invalid sentinel


# --------------------------------------------------------------------------- #
# Type 9: pose  (t xyz f32@24, trackingState f32@36, quat xyzw f32@48)
# --------------------------------------------------------------------------- #


def _pose_buf(tracking: float) -> bytes:
    buf = make_header(9)
    struct.pack_into("<3f", buf, 24, 1.5, -2.25, 0.5)
    struct.pack_into("<f", buf, 36, tracking)
    struct.pack_into("<4f", buf, 48, 0.0, 0.0, 0.5, 0.5)
    return bytes(buf)


def test_pose_record_golden() -> None:
    rec = decode_record(_pose_buf(2.0))
    assert type(rec) is Pose
    assert_common(rec)
    assert rec.position == Vec3(1.5, -2.25, 0.5)
    assert rec.orientation == Quat(0.0, 0.0, 0.5, 0.5)
    assert rec.tracking is Tracking.NORMAL
    assert rec.discontinuity is False  # flags byte 0 by default


def test_pose_discontinuity_flag() -> None:
    # flags @1 bit0 = first pose after an ARKit interruption/relocalization (app >= 1.1)
    buf = bytearray(_pose_buf(2.0))
    buf[1] = 0x01
    assert decode_record(bytes(buf)).discontinuity is True
    buf[1] = 0xFE  # other flag bits set, bit0 clear -> False (forward compat)
    assert decode_record(bytes(buf)).discontinuity is False
    buf[1] = 0xFF
    assert decode_record(bytes(buf)).discontinuity is True


@pytest.mark.parametrize(
    ("wire_state", "expected"),
    [(0.0, Tracking.NONE), (1.0, Tracking.LIMITED), (2.0, Tracking.NORMAL),
     (7.0, Tracking.NONE)],  # unknown future state degrades to NONE
)
def test_pose_tracking_states(wire_state: float, expected: Tracking) -> None:
    rec = decode_record(_pose_buf(wire_state))
    assert isinstance(rec, Pose)
    assert rec.tracking is expected


# --------------------------------------------------------------------------- #
# Unknown types
# --------------------------------------------------------------------------- #


def test_unknown_record_type_is_kept_not_raised() -> None:
    buf = make_header(42)
    payload = bytes(range(24, 64))  # distinctive 40-byte pattern
    buf[24:64] = payload
    rec = decode_record(bytes(buf))
    assert type(rec) is Unknown
    assert_common(rec)
    assert rec.type_id == 42
    assert rec.payload == payload
    assert len(rec.payload) == 40


def test_decode_record_accepts_bytearray_and_memoryview() -> None:
    buf = make_header(2)
    struct.pack_into("<3f", buf, 24, 1.0, 2.0, 3.0)
    for view in (bytes(buf), buf, memoryview(bytes(buf))):
        rec = decode_record(view)
        assert isinstance(rec, RawGyro)
        assert rec.gyro == Vec3(1.0, 2.0, 3.0)


def test_decode_record_truncated_raises() -> None:
    with pytest.raises(ProtocolError):
        decode_record(bytes(make_header(1))[:63])
    with pytest.raises(ProtocolError):
        decode_record(b"")


# --------------------------------------------------------------------------- #
# Depth frames
# --------------------------------------------------------------------------- #


def make_depth_payload(width: int, height: int, samples: bytes, *,
                       type_id: int = 10, flags: int = 1, seq: int = SEQ,
                       host_ts: float = HOST_TS, unix_ts: float = UNIX_TS,
                       bytes_per_pixel: int = 2) -> bytes:
    header = bytearray(32)
    struct.pack_into("<B", header, 0, type_id)
    struct.pack_into("<B", header, 1, flags)          # bit0: float16
    struct.pack_into("<H", header, 2, seq)
    struct.pack_into("<I", header, 4, 0)              # reserved
    struct.pack_into("<d", header, 8, host_ts)
    struct.pack_into("<d", header, 16, unix_ts)
    struct.pack_into("<H", header, 24, width)
    struct.pack_into("<H", header, 26, height)
    struct.pack_into("<B", header, 28, bytes_per_pixel)
    # 29..31 pad, already zero
    return bytes(header) + samples


def test_depth_frame_golden() -> None:
    values = (1.0, 2.0, 0.5, 4.0, 0.25, 8.0)  # exact in binary16
    samples = struct.pack("<6e", *values)
    frame = decode_depth_frame(make_depth_payload(3, 2, samples))

    assert type(frame) is DepthFrame
    assert_common(frame)
    assert frame.width == 3
    assert frame.height == 2
    assert frame.data == samples
    # row-major: at(x, y) reads sample y*width + x
    assert frame.at(0, 0) == 1.0
    assert frame.at(2, 0) == 0.5
    assert frame.at(0, 1) == 4.0
    assert frame.at(2, 1) == 8.0


def test_depth_frame_wrong_pixel_size_raises() -> None:
    samples = struct.pack("<6f", *range(6))  # pretend float32 samples
    payload = make_depth_payload(3, 2, samples, bytes_per_pixel=4)
    with pytest.raises(ProtocolError):
        decode_depth_frame(payload)


def test_depth_frame_truncated_samples_raises() -> None:
    samples = struct.pack("<6e", *range(6))
    payload = make_depth_payload(3, 2, samples)
    with pytest.raises(ProtocolError):
        decode_depth_frame(payload[:-2])  # one sample short


def test_depth_frame_extra_samples_raises() -> None:
    samples = struct.pack("<7e", *range(7))  # one sample too many for 3x2
    with pytest.raises(ProtocolError):
        decode_depth_frame(make_depth_payload(3, 2, samples))


def test_depth_frame_short_header_raises() -> None:
    with pytest.raises(ProtocolError):
        decode_depth_frame(b"\x0a" + b"\x00" * 30)  # 31 < 32-byte header


def test_depth_frame_wrong_type_raises() -> None:
    samples = struct.pack("<6e", *range(6))
    with pytest.raises(ProtocolError):
        decode_depth_frame(make_depth_payload(3, 2, samples, type_id=1))


def test_record_type_enum_matches_swift_ids() -> None:
    assert RecordType.IMU == 1
    assert RecordType.GYRO == 2
    assert RecordType.ACCEL == 3
    assert RecordType.INTRINSICS == 5
    assert RecordType.GNSS == 6
    assert RecordType.ALTITUDE == 7
    assert RecordType.HEADING == 8
    assert RecordType.POSE == 9
    assert RecordType.DEPTH == 10


# --------------------------------------------------------------------------- #
# Handshake / length-prefixed framing over a real socket pair
# --------------------------------------------------------------------------- #


def _served(data: bytes) -> socket.socket:
    """A socket whose peer has already sent `data` and closed."""
    a, b = socket.socketpair()
    a.sendall(data)
    a.close()
    return b


def test_read_exact_reassembles_and_detects_close() -> None:
    sock = _served(b"abcdef")
    try:
        assert read_exact(sock, 4) == b"abcd"
        assert read_exact(sock, 2) == b"ef"
        with pytest.raises(ConnectionClosed):
            read_exact(sock, 1)
    finally:
        sock.close()


def test_recv_length_prefixed_golden() -> None:
    body = b"\x01\x02\x03\x04\x05"
    sock = _served(struct.pack("<I", len(body)) + body)
    try:
        assert recv_length_prefixed(sock) == body
    finally:
        sock.close()


@pytest.mark.parametrize("length", [0, 0xFFFFFFFF])
def test_recv_length_prefixed_implausible_length_raises(length: int) -> None:
    sock = _served(struct.pack("<I", length))
    try:
        with pytest.raises(ProtocolError):
            recv_length_prefixed(sock)
    finally:
        sock.close()


def test_recv_handshake_golden_json() -> None:
    handshake = {"protocol": "irtsp-imu", "version": 1, "record_size": 64}
    raw = json.dumps(handshake).encode()
    sock = _served(struct.pack("<I", len(raw)) + raw)
    try:
        assert recv_handshake(sock) == handshake
    finally:
        sock.close()


@pytest.mark.parametrize("raw", [b"not json at all", b"[1, 2, 3]"])
def test_recv_handshake_rejects_non_object(raw: bytes) -> None:
    sock = _served(struct.pack("<I", len(raw)) + raw)
    try:
        with pytest.raises(ProtocolError):
            recv_handshake(sock)
    finally:
        sock.close()
