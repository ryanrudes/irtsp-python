"""A threaded TCP mock of an iRTSP phone's two side-channels, for tests.

Byte-faithful to the app's servers (``IMUStreamServer.swift`` /
``DepthStreamServer.swift`` / ``IMUWireFormat.swift``):

* On connect, each channel sends ``[u32 LE length][UTF-8 JSON handshake]``.
* The **odometry** channel then streams back-to-back fixed 64-byte
  little-endian records (byte 0 is the type).
* The **depth** channel streams ``[u32 LE length][32-byte header +
  tightly-packed float16 samples]`` frames.

Stdlib only. Typical use::

    phone = MockPhone().start()
    session = irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0)
    phone.emit_imu(seq=1, gyro=(0.1, 0.2, 0.3))
    ...
    phone.close()
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any, Callable, Iterable, Sequence

RECORD_SIZE = 64
DEPTH_HEADER_SIZE = 32

# Wire record type ids (IMUWire.RecordType + DepthStreamServer's frame type).
T_IMU = 1
T_GYRO = 2
T_ACCEL = 3
T_INTRINSICS = 5
T_GNSS = 6
T_ALTITUDE = 7
T_HEADING = 8
T_POSE = 9
T_DEPTH = 10


def length_prefixed(payload: bytes) -> bytes:
    """Frame a payload the way both channels do: ``[u32 LE length][payload]``."""
    return struct.pack("<I", len(payload)) + payload


class _Channel:
    """One listening TCP port that handshakes every client and broadcasts to all."""

    def __init__(self, name: str, handshake_factory: Callable[[], dict[str, Any]]):
        self.name = name
        self._handshake_factory = handshake_factory
        self._lock = threading.Lock()
        self._clients: list[socket.socket] = []
        self.connected = threading.Event()  # set once >=1 client is registered
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def port(self) -> int:
        assert self._listener is not None, f"mock {self.name} channel not started"
        return self._listener.getsockname()[1]

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        srv.settimeout(0.1)  # poll `_closed` so close() is always prompt
        self._listener = srv
        self._thread = threading.Thread(
            target=self._accept_loop, name=f"mock-{self.name}-accept", daemon=True
        )
        self._thread.start()

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            # Like the real servers: the handshake is the first thing on the wire.
            handshake = json.dumps(self._handshake_factory()).encode("utf-8")
            try:
                conn.sendall(length_prefixed(handshake))
            except OSError:
                conn.close()
                continue
            with self._lock:
                self._clients.append(conn)
            self.connected.set()

    def broadcast(self, data: bytes) -> None:
        """Send ``data`` to every connected client (fire-and-forget, like the app).

        Waits briefly for the first client so tests can emit right after
        ``irtsp.connect`` returns without racing the accept thread.
        """
        if not self.connected.wait(timeout=2.0):
            raise RuntimeError(f"no client connected to mock {self.name} channel")
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.sendall(data)
            except OSError:
                self._drop(conn)

    def _drop(self, conn: socket.socket) -> None:
        with self._lock:
            if conn in self._clients:
                self._clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def close_clients(self) -> None:
        """Hard-close every connected client (server-side stream stop)."""
        with self._lock:
            clients, self._clients = self._clients, []
            self.connected.clear()
        for conn in clients:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        self.close_clients()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class MockPhone:
    """Mock of both iRTSP channels on 127.0.0.1 ephemeral ports.

    Emit helpers write the exact little-endian layouts from
    ``IMUWireFormat.swift``. ``seq``/``host_ts``/``unix_ts`` default to
    auto-incrementing / clock-consistent values but can be pinned per record.
    """

    def __init__(
        self,
        *,
        host_anchor: float = 1000.0,
        wall_anchor: float = 1_700_000_000.0,
        video_url: str = "rtsp://ryans-iphone.local:8554/live",
        video_codec: str = "h264",
        video_clock_rate: int = 90000,
        streams: dict[str, bool] | None = None,
        attitude: bool = True,
        rate_hz: float = 200.0,
    ):
        self.host_anchor = host_anchor
        self.wall_anchor = wall_anchor
        self.video_url = video_url
        self.video_codec = video_codec
        self.video_clock_rate = video_clock_rate
        self.attitude = attitude
        self.rate_hz = rate_hz
        self.streams: dict[str, bool] = dict(streams) if streams is not None else {
            "imu": True,
            "intrinsics": True,
            "gnss": True,
            "altitude": True,
            "heading": True,
            "pose": True,
        }
        self._odo = _Channel("odometry", self._odometry_handshake)
        self._depth = _Channel("depth", self._depth_handshake)
        self._odo_seq = 0
        self._depth_seq = 0
        self._ticks = 0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "MockPhone":
        self._odo.start()
        self._depth.start()
        return self

    @property
    def imu_port(self) -> int:
        return self._odo.port

    @property
    def depth_port(self) -> int:
        return self._depth.port

    def wait_for_odometry_client(self, timeout: float = 2.0) -> bool:
        return self._odo.connected.wait(timeout)

    def wait_for_depth_client(self, timeout: float = 2.0) -> bool:
        return self._depth.connected.wait(timeout)

    def close_odometry_clients(self) -> None:
        self._odo.close_clients()

    def close_depth_clients(self) -> None:
        self._depth.close_clients()

    def close(self) -> None:
        self._odo.close()
        self._depth.close()

    # ------------------------------------------------------------ handshakes

    def _clock_dict(self) -> dict[str, Any]:
        return {
            "timebase": "mach_absolute_time_seconds",
            "host_anchor": self.host_anchor,
            "wall_anchor": self.wall_anchor,
            "rtcp_sync": "unix_ts matches RTP RTCP SR NTP timeline",
        }

    def _odometry_handshake(self) -> dict[str, Any]:
        """Mirrors ``IMUStreamServer.makeHandshake``."""
        return {
            "protocol": "irtsp-imu",
            "version": 1,
            "endianness": "little",
            "record_bytes": RECORD_SIZE,
            "mode": "fused",
            "rate_hz": self.rate_hz,
            "gyro_units": "rad/s",
            "accel_units": "g",
            "accel_convention": "gravity+userAcceleration; face-up rest ~ (0,0,-1)",
            "attitude": "quaternion_xyzw" if self.attitude else "none",
            "attitude_frame": "xArbitraryZVertical",
            "body_axes": "X-right, Y-up, Z-out-of-screen",
            "clock": self._clock_dict(),
            "video": {
                "rtsp_url": self.video_url,
                "clock_rate": self.video_clock_rate,
                "codec": self.video_codec,
            },
            "lens_distortion": "none",
            "record_types": {
                "imu": T_IMU, "gyro": T_GYRO, "accel": T_ACCEL,
                "intrinsics": T_INTRINSICS, "gnss": T_GNSS,
                "altitude": T_ALTITUDE, "heading": T_HEADING, "pose": T_POSE,
            },
            "streams": dict(self.streams),
        }

    def _depth_handshake(self) -> dict[str, Any]:
        """Mirrors ``DepthStreamServer.makeHandshake``."""
        return {
            "protocol": "irtsp-depth",
            "version": 1,
            "endianness": "little",
            "frame_type": T_DEPTH,
            "pixel_format": "depth_float16",
            "units": "meters",
            "clock": self._clock_dict(),
            "video": {"rtsp_url": self.video_url, "codec": self.video_codec},
        }

    # -------------------------------------------------------------- encoding

    def unix_from_host(self, host_ts: float) -> float:
        """The unix_ts the app would stamp for ``host_ts`` (one shared anchor)."""
        return self.wall_anchor + (host_ts - self.host_anchor)

    def _times(self, host_ts: float | None, unix_ts: float | None) -> tuple[float, float]:
        if host_ts is None:
            self._ticks += 1
            host_ts = self.host_anchor + 0.01 * self._ticks
        if unix_ts is None:
            unix_ts = self.unix_from_host(host_ts)
        return host_ts, unix_ts

    def _next_odo_seq(self, seq: int | None) -> int:
        if seq is None:
            self._odo_seq = (self._odo_seq + 1) & 0xFFFF
        else:
            self._odo_seq = seq & 0xFFFF
        return self._odo_seq

    @staticmethod
    def _record(rtype: int, seq: int, host_ts: float, unix_ts: float) -> bytearray:
        """The shared 64-byte record header (type/flags/seq/reserved/host/unix)."""
        buf = bytearray(RECORD_SIZE)
        buf[0] = rtype
        buf[1] = 0  # flags
        struct.pack_into("<H", buf, 2, seq & 0xFFFF)
        # 4..8 reserved (already zero)
        struct.pack_into("<d", buf, 8, host_ts)
        struct.pack_into("<d", buf, 16, unix_ts)
        return buf

    def send_odometry(self, raw: bytes) -> None:
        """Escape hatch: push raw bytes down the odometry channel."""
        self._odo.broadcast(raw)

    # --------------------------------------------------------- odometry emits

    def emit_imu(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        gyro: Sequence[float] = (0.0, 0.0, 0.0),
        accel_g: Sequence[float] = (0.0, 0.0, -1.0),
        quat: Sequence[float] | None = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Type 1 fused sample. ``quat=None`` = attitude-off (zeroed quat slots)."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_IMU, seq, host_ts, unix_ts)
        struct.pack_into("<3f", buf, 24, *gyro)
        struct.pack_into("<3f", buf, 36, *accel_g)
        if quat is not None:
            struct.pack_into("<4f", buf, 48, *quat)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_raw_gyro(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        gyro: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Type 2 raw gyroscope (gyro xyz rad/s in the @24 slots)."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_GYRO, seq, host_ts, unix_ts)
        struct.pack_into("<3f", buf, 24, *gyro)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_raw_accel(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        accel_g: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> int:
        """Type 3 raw accelerometer (accel xyz g in the @36 slots)."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_ACCEL, seq, host_ts, unix_ts)
        struct.pack_into("<3f", buf, 36, *accel_g)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_intrinsics(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        fx: float = 1000.0,
        fy: float = 1000.0,
        cx: float = 960.0,
        cy: float = 540.0,
        width: int = 1920,
        height: int = 1080,
    ) -> int:
        """Type 5: [fx, fy, ox] @24 then [oy, width, height] @36."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_INTRINSICS, seq, host_ts, unix_ts)
        struct.pack_into("<6f", buf, 24, fx, fy, cx, cy, float(width), float(height))
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_gnss(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        lat: float = 40.7128,
        lon: float = -74.0060,
        altitude: float = 10.0,
        h_acc: float = 5.0,
        v_acc: float = 4.0,
        speed: float = -1.0,
        course: float = -1.0,
        speed_acc: float = -1.0,
    ) -> int:
        """Type 6: lat/lon f64 @24/@32, then 6×f32 @40 (negatives = invalid)."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_GNSS, seq, host_ts, unix_ts)
        struct.pack_into("<dd", buf, 24, lat, lon)
        struct.pack_into("<6f", buf, 40, altitude, h_acc, v_acc, speed, course, speed_acc)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_altitude(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        relative_altitude: float = 0.0,
        pressure_kpa: float = 101.325,
    ) -> int:
        """Type 7: relativeAltitude (m) @24, pressure (kPa) @28."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_ALTITUDE, seq, host_ts, unix_ts)
        struct.pack_into("<2f", buf, 24, relative_altitude, pressure_kpa)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_heading(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        true_deg: float = 90.0,
        magnetic_deg: float = 88.0,
        accuracy_deg: float = 5.0,
    ) -> int:
        """Type 8: true @24, magnetic @28, accuracy @32 (negatives = invalid)."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_HEADING, seq, host_ts, unix_ts)
        struct.pack_into("<3f", buf, 24, true_deg, magnetic_deg, accuracy_deg)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_pose(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        tracking: int = 2,
    ) -> int:
        """Type 9: translation @24, trackingState f32 @36, quaternion @48."""
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_POSE, seq, host_ts, unix_ts)
        struct.pack_into("<3f", buf, 24, *position)
        struct.pack_into("<f", buf, 36, float(tracking))
        struct.pack_into("<4f", buf, 48, *quat)
        self._odo.broadcast(bytes(buf))
        return seq

    def emit_unknown(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        type_id: int = 42,
        payload: bytes = b"",
    ) -> int:
        """A record type this library version doesn't know (forward compat)."""
        assert len(payload) <= RECORD_SIZE - 24, "payload is the 24..64 slot"
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(type_id, seq, host_ts, unix_ts)
        buf[24 : 24 + len(payload)] = payload
        self._odo.broadcast(bytes(buf))
        return seq

    # ------------------------------------------------------------ depth emits

    def emit_depth(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        width: int,
        height: int,
        samples: bytes | Iterable[float],
    ) -> int:
        """One length-prefixed depth frame: 32-byte header + float16 meters.

        ``samples`` is either pre-packed little-endian float16 bytes or a
        row-major iterable of ``width*height`` distances in meters.
        """
        if seq is None:
            self._depth_seq = (self._depth_seq + 1) & 0xFFFF
        else:
            self._depth_seq = seq & 0xFFFF
        seq = self._depth_seq
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        if not isinstance(samples, (bytes, bytearray)):
            values = list(samples)
            assert len(values) == width * height, "need width*height samples"
            samples = struct.pack(f"<{len(values)}e", *values)
        header = bytearray(DEPTH_HEADER_SIZE)
        header[0] = T_DEPTH
        header[1] = 1  # flags bit0: samples are float16
        struct.pack_into("<H", header, 2, seq & 0xFFFF)
        # 4..8 reserved
        struct.pack_into("<d", header, 8, host_ts)
        struct.pack_into("<d", header, 16, unix_ts)
        struct.pack_into("<HH", header, 24, width, height)
        header[28] = 2  # bytesPerPixel
        # 29..31 pad
        self._depth.broadcast(length_prefixed(bytes(header) + bytes(samples)))
        return seq
