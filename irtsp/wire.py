"""Low-level wire decoding for the iRTSP side-channels.

You normally never need this module — :func:`irtsp.connect` hands you typed
records. It exists so the byte-exact protocol lives in one auditable place.

Framing (both channels start the same way):

* On connect the server sends ``[u32 LE length][UTF-8 JSON handshake]``.
* The **odometry** channel (default port 8555) then streams back-to-back fixed
  **64-byte** little-endian records — no per-record framing. Byte 0 is the type.
* The **depth** channel (default port 8556) streams ``[u32 LE length][frame]``
  where each frame is a 32-byte header + tightly-packed half-float samples.

Reference: https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md
(and, upstream, ``IMUWireFormat.swift`` / ``DepthStreamServer.swift`` in the app).
"""

from __future__ import annotations

import json
import math
import socket
import struct
from enum import IntEnum
from typing import Any

from .records import (
    STANDARD_GRAVITY,
    Altitude,
    DepthFrame,
    GNSS,
    Heading,
    IMU,
    Intrinsics,
    Pose,
    Quat,
    RawAccel,
    RawGyro,
    Record,
    Tracking,
    Unknown,
    Vec3,
)

__all__ = [
    "RECORD_SIZE",
    "MAX_MESSAGE",
    "RecordType",
    "ProtocolError",
    "ConnectionClosed",
    "read_exact",
    "recv_handshake",
    "recv_length_prefixed",
    "decode_record",
    "decode_depth_frame",
]

#: Fixed size of every odometry record on the wire.
RECORD_SIZE = 64

#: Maximum sane length-prefixed message (guards against desync garbage).
MAX_MESSAGE = 64 * 1024 * 1024


class RecordType(IntEnum):
    """Wire record type ids (byte 0 of each record)."""

    IMU = 1
    GYRO = 2
    ACCEL = 3
    INTRINSICS = 5
    GNSS = 6
    ALTITUDE = 7
    HEADING = 8
    POSE = 9
    DEPTH = 10  # depth channel only


class ProtocolError(ValueError):
    """The bytes on the wire don't look like the iRTSP protocol."""


class ConnectionClosed(ConnectionError):
    """The server closed the connection (or the stream was stopped)."""


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from a socket (TCP reads may return short)."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionClosed("stream closed by server")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_length_prefixed(sock: socket.socket) -> bytes:
    """Read one ``[u32 LE length][payload]`` message."""
    (length,) = struct.unpack("<I", read_exact(sock, 4))
    if not 0 < length <= MAX_MESSAGE:
        raise ProtocolError(f"implausible message length {length} — desynced or not iRTSP")
    return read_exact(sock, length)


def recv_handshake(sock: socket.socket) -> dict[str, Any]:
    """Read and parse the JSON handshake the server sends on connect."""
    raw = recv_length_prefixed(sock)
    try:
        handshake = json.loads(raw)
    except ValueError as e:
        raise ProtocolError("handshake is not valid JSON — is this an iRTSP port?") from e
    if not isinstance(handshake, dict):
        raise ProtocolError("handshake JSON is not an object")
    return handshake


# --------------------------------------------------------------------------- #
# 64-byte odometry records
#
# Shared header (little-endian):
#   0  u8   type      1  u8   flags     2  u16  seq     4  u32  reserved
#   8  f64  host_ts   16 f64  unix_ts   24..64  type-specific payload
# --------------------------------------------------------------------------- #


def _optional(value: float) -> float | None:
    """CoreLocation encodes 'invalid' as a negative value."""
    return None if value < 0 else value


def decode_record(buf: bytes | bytearray | memoryview) -> Record:
    """Decode one 64-byte odometry record into its typed :class:`~irtsp.Record`.

    Unknown types come back as :class:`~irtsp.Unknown` rather than raising, so
    old clients keep working when new record types appear.
    """
    if len(buf) < RECORD_SIZE:
        raise ProtocolError(f"record needs {RECORD_SIZE} bytes, got {len(buf)}")

    type_id = buf[0]
    (seq,) = struct.unpack_from("<H", buf, 2)
    host_ts, unix_ts = struct.unpack_from("<dd", buf, 8)
    common = {"host_ts": host_ts, "unix_ts": unix_ts, "seq": seq}

    if type_id == RecordType.IMU:
        gx, gy, gz, ax, ay, az, qx, qy, qz, qw = struct.unpack_from("<10f", buf, 24)
        # Attitude-off sessions send NaN quat slots (MotionSampler.swift fills the
        # sample with .nan); older/other encoders may zero them. Either way: None.
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        quat = Quat(qx, qy, qz, qw) if math.isfinite(norm_sq) and norm_sq > 0.25 else None
        return IMU(
            gyro=Vec3(gx, gy, gz),
            accel=Vec3(ax, ay, az) * STANDARD_GRAVITY,  # wire is g → SI m/s²
            quat=quat,
            **common,
        )

    if type_id == RecordType.GYRO:
        gx, gy, gz = struct.unpack_from("<3f", buf, 24)
        return RawGyro(gyro=Vec3(gx, gy, gz), **common)

    if type_id == RecordType.ACCEL:
        ax, ay, az = struct.unpack_from("<3f", buf, 36)
        return RawAccel(accel=Vec3(ax, ay, az) * STANDARD_GRAVITY, **common)

    if type_id == RecordType.INTRINSICS:
        fx, fy, cx, cy, width, height = struct.unpack_from("<6f", buf, 24)
        return Intrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy, width=int(width), height=int(height), **common
        )

    if type_id == RecordType.GNSS:
        lat, lon = struct.unpack_from("<dd", buf, 24)
        altitude, h_acc, v_acc, speed, course, speed_acc = struct.unpack_from("<6f", buf, 40)
        return GNSS(
            lat=lat,
            lon=lon,
            altitude=altitude,
            h_accuracy=_optional(h_acc),
            v_accuracy=_optional(v_acc),
            speed=_optional(speed),
            course_deg=_optional(course),
            speed_accuracy=_optional(speed_acc),
            **common,
        )

    if type_id == RecordType.ALTITUDE:
        relative, pressure_kpa = struct.unpack_from("<2f", buf, 24)
        return Altitude(
            relative_altitude=relative,
            pressure=pressure_kpa * 1000.0,  # wire is kPa → SI Pa
            **common,
        )

    if type_id == RecordType.HEADING:
        true_h, magnetic, accuracy = struct.unpack_from("<3f", buf, 24)
        return Heading(
            true_deg=_optional(true_h),
            magnetic_deg=magnetic,
            accuracy_deg=_optional(accuracy),
            **common,
        )

    if type_id == RecordType.POSE:
        tx, ty, tz, tracking = struct.unpack_from("<4f", buf, 24)
        tilt, azimuth = struct.unpack_from("<2f", buf, 40)
        qx, qy, qz, qw = struct.unpack_from("<4f", buf, 48)
        state = int(tracking)
        # Bytes 40..48 were a zero-filled hole on older apps. Decoding that as a literal
        # 0.0° tilt would report those captures as PERFECTLY level — the precise false
        # negative this field exists to catch. An app that really measures gravity never
        # emits an exact 0.0/0.0 pair (the azimuth is an atan2 of sensor noise), so treat
        # exact zeros as "not reported" and let them surface as nan.
        if tilt == 0.0 and azimuth == 0.0:
            tilt = azimuth = math.nan
        return Pose(
            position=Vec3(tx, ty, tz),
            orientation=Quat(qx, qy, qz, qw),
            tracking=Tracking(state) if state in (0, 1, 2) else Tracking.NONE,
            # flags bit0: the world frame moved — re-anchor here
            discontinuity=bool(buf[1] & 0x01),
            # bit1: tracking recovered; bit2: silent loop closure / map merge
            # (both zero on older apps, which is indistinguishable from 'did not happen')
            relocalized=bool(buf[1] & 0x02),
            jump=bool(buf[1] & 0x04),
            gravity_tilt_deg=tilt,
            gravity_azimuth_deg=azimuth,
            **common,
        )

    return Unknown(type_id=type_id, payload=bytes(buf[24:RECORD_SIZE]), **common)


# --------------------------------------------------------------------------- #
# Depth frames (their own channel; length-prefixed)
#
# 32-byte header (little-endian):
#   0  u8   type (=10)   1  u8   flags (bit0: samples are float16)
#   2  u16  seq          4  u32  reserved
#   8  f64  host_ts      16 f64  unix_ts
#   24 u16  width        26 u16  height    28 u8  bytesPerPixel   29..31 pad
# --------------------------------------------------------------------------- #

_DEPTH_HEADER = 32


def decode_depth_frame(payload: bytes) -> DepthFrame:
    """Decode one depth-channel frame (header + samples, without the u32 prefix)."""
    if len(payload) < _DEPTH_HEADER:
        raise ProtocolError(f"depth frame needs ≥{_DEPTH_HEADER} bytes, got {len(payload)}")
    type_id = payload[0]
    if type_id != RecordType.DEPTH:
        raise ProtocolError(f"expected depth frame (type 10), got type {type_id}")

    flags = payload[1]
    (seq,) = struct.unpack_from("<H", payload, 2)
    host_ts, unix_ts = struct.unpack_from("<dd", payload, 8)
    width, height = struct.unpack_from("<HH", payload, 24)
    bytes_per_pixel = payload[28]

    data = payload[_DEPTH_HEADER:]
    expected = width * height * bytes_per_pixel
    if not flags & 0x01:  # bit0 = samples are float16 (the only format defined today)
        raise ProtocolError(f"unsupported depth pixel format (flags={flags:#04x}, expected float16)")
    if bytes_per_pixel != 2:
        raise ProtocolError(f"unsupported depth pixel size {bytes_per_pixel} (expected float16)")
    if len(data) != expected:
        raise ProtocolError(
            f"depth payload is {len(data)} bytes, expected {expected} for {width}x{height}"
        )
    return DepthFrame(
        host_ts=host_ts, unix_ts=unix_ts, seq=seq, width=width, height=height, data=data
    )
