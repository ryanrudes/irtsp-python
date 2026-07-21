"""Threaded TCP mocks of an iRTSP phone's servers, for tests.

:class:`MockPhone` is the two side-channels, byte-faithful to the app's servers
(``IMUStreamServer.swift`` / ``DepthStreamServer.swift`` / ``IMUWireFormat.swift``):

* On connect, each channel sends ``[u32 LE length][UTF-8 JSON handshake]``.
* The **odometry** channel then streams back-to-back fixed 64-byte
  little-endian records (byte 0 is the type).
* The **depth** channel streams ``[u32 LE length][32-byte header +
  tightly-packed float16 samples]`` frames.

:class:`MockAudioPhone` is the RTSP server's audio track, byte-faithful to
``RTSPConnection.swift`` / ``SDPBuilder.swift`` / ``RTPCore.swift`` / ``RTCP.swift``:
the RTSP handshake (with Basic/Digest auth), then ``$``-framed interleaved RTP
and compound RTCP Sender Reports.

Stdlib only. Typical use::

    phone = MockPhone().start()
    session = irtsp.connect("127.0.0.1", imu_port=phone.imu_port, timeout=2.0)
    phone.emit_imu(seq=1, gyro=(0.1, 0.2, 0.3))
    ...
    phone.close()

    mic = MockAudioPhone(codec="L16", sample_rate=48000, channels=1).start()
    audio = irtsp.audio_stream("127.0.0.1", port=mic.port, timeout=2.0)
    mic.emit_l16([0, 1, 2, 3])
    mic.end_stream()
    mic.close()
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import socket
import struct
import threading
import zlib
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
T_FORMAT = 11


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
        # Everything any client sent back (v2 clients send control messages).
        self._recv_cond = threading.Condition()
        self._received = bytearray()

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
            threading.Thread(
                target=self._read_loop, args=(conn,),
                name=f"mock-{self.name}-read", daemon=True,
            ).start()
            self.connected.set()

    def _read_loop(self, conn: socket.socket) -> None:
        """Collect client→server bytes (the v2 compression opt-in rides here)."""
        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            with self._recv_cond:
                self._received += chunk
                self._recv_cond.notify_all()

    def wait_received(self, n: int, timeout: float = 2.0) -> bytes:
        """Block until clients have sent ≥ ``n`` bytes total; return a snapshot."""
        with self._recv_cond:
            self._recv_cond.wait_for(lambda: len(self._received) >= n, timeout)
            return bytes(self._received)

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
        version: int = 1,
        revision: int = 0,
        depth_codecs: Sequence[str] = ("lzfse", "zlib"),
    ):
        self.host_anchor = host_anchor
        self.wall_anchor = wall_anchor
        self.video_url = video_url
        self.video_codec = video_codec
        self.video_clock_rate = video_clock_rate
        self.attitude = attitude
        self.rate_hz = rate_hz
        self.version = version
        self.revision = revision  # >=1 (with version>=2) advertises the v2.1 format channel
        self.depth_codecs = list(depth_codecs)  # advertised when version >= 2
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
        self._depth_ctrl_offset = 0  # parse cursor into the depth channel's inbox

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
        """Mirrors ``IMUStreamServer.makeHandshake`` (v2 adds the state-channel keys)."""
        handshake: dict[str, Any] = {
            "protocol": "irtsp-imu",
            "version": self.version,
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
        if self.version >= 2:
            handshake["emission"] = {
                "imu": "continuous", "gyro": "continuous", "accel": "continuous",
                "pose": "continuous", "gnss": "event", "altitude": "event",
                "intrinsics": "state", "heading": "state",
            }
            handshake["state_channels"] = {
                "keyframe_interval_s": 10,
                "flags": {"bit0": "snapshot_or_keyframe"},
            }
            if self.revision >= 1:
                # Protocol 2.1: purely additive — `version` stays 2, a new `revision`
                # plus the type-11 camera-format channel across the maps.
                handshake["revision"] = self.revision
                handshake["record_types"]["format"] = T_FORMAT
                handshake["emission"]["format"] = "state"
                handshake["streams"]["format"] = True
                handshake["format_channel"] = {
                    "note": "type-11 priors for rolling-shutter (§5.3)",
                    "readout_time": "per-format constant, probed from .mov metadata",
                    "pts_convention": "declared, with provenance",
                }
        return handshake

    def _depth_handshake(self) -> dict[str, Any]:
        """Mirrors ``DepthStreamServer.makeHandshake`` (v2 advertises compression)."""
        handshake: dict[str, Any] = {
            "protocol": "irtsp-depth",
            "version": self.version,
            "endianness": "little",
            "frame_type": T_DEPTH,
            "pixel_format": "depth_float16",
            "units": "meters",
            "clock": self._clock_dict(),
            "video": {"rtsp_url": self.video_url, "codec": self.video_codec},
        }
        if self.version >= 2:
            handshake["compression"] = {
                "supported": list(self.depth_codecs),
                "request": '[u32 LE length][{"compression": "<codec>"} JSON]',
            }
        return handshake

    # ------------------------------------------------- client control messages

    def recv_depth_control(self, timeout: float = 2.0) -> dict[str, Any] | None:
        """The next ``[u32 LE length][JSON]`` message a depth client sent, or None.

        This is how a v2 client opts in to compression; a well-behaved client
        sends nothing at all to a v1 handshake.
        """
        start = self._depth_ctrl_offset
        buf = self._depth.wait_received(start + 4, timeout)
        if len(buf) < start + 4:
            return None
        (length,) = struct.unpack_from("<I", buf, start)
        buf = self._depth.wait_received(start + 4 + length, timeout)
        if len(buf) < start + 4 + length:
            return None
        self._depth_ctrl_offset = start + 4 + length
        return json.loads(buf[start + 4 : start + 4 + length].decode("utf-8"))

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
        snapshot: bool = False,
    ) -> int:
        """Type 5: [fx, fy, ox] @24 then [oy, width, height] @36.

        ``snapshot=True`` sets flags bit0 (v2 state-channel snapshot/keyframe).
        """
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_INTRINSICS, seq, host_ts, unix_ts)
        buf[1] = 0x01 if snapshot else 0x00
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
        snapshot: bool = False,
    ) -> int:
        """Type 8: true @24, magnetic @28, accuracy @32 (negatives = invalid).

        ``snapshot=True`` sets flags bit0 (v2 state-channel snapshot/keyframe).
        """
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        buf = self._record(T_HEADING, seq, host_ts, unix_ts)
        buf[1] = 0x01 if snapshot else 0x00
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

    def emit_format(
        self,
        *,
        seq: int | None = None,
        host_ts: float | None = None,
        unix_ts: float | None = None,
        format_id: int = 0x1234ABCD,
        width: int = 1920,
        height: int = 1440,
        fps: float = 30.0,
        readout: float | None = None,
        camera: int = 1,
        capture_path: int = 0,
        binned: bool = False,
        cropped: bool = False,
        readout_direction: int = 1,
        pts_convention: int = 1,
        pts_provenance: int = 1,
        readout_provenance: int | None = None,
        snapshot: bool = False,
    ) -> int:
        """Type 11 camera-format record (protocol v2.1).

        ``readout`` is the full-frame readout time in seconds; ``None`` writes NaN.
        ``readout_provenance`` defaults to ``probed`` (1) when a readout is given and
        ``absent`` (0) otherwise — override it to exercise the decoder's edge cases.
        ``snapshot=True`` sets flags bit0 (state-channel snapshot/keyframe).
        """
        seq = self._next_odo_seq(seq)
        host_ts, unix_ts = self._times(host_ts, unix_ts)
        readout_time = math.nan if readout is None else readout
        if readout_provenance is None:
            readout_provenance = 1 if readout is not None else 0
        buf = self._record(T_FORMAT, seq, host_ts, unix_ts)
        buf[1] = 0x01 if snapshot else 0x00
        struct.pack_into("<I", buf, 24, format_id & 0xFFFFFFFF)
        struct.pack_into("<HH", buf, 28, width, height)
        struct.pack_into("<2f", buf, 32, fps, readout_time)
        buf[40] = camera
        buf[41] = capture_path
        buf[42] = (0x01 if binned else 0) | (0x02 if cropped else 0)
        buf[43] = readout_direction
        buf[44] = pts_convention
        buf[45] = pts_provenance
        buf[46] = readout_provenance
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
        codec: str | None = None,
    ) -> int:
        """One length-prefixed depth frame: 32-byte header + float16 meters.

        ``samples`` is either pre-packed little-endian float16 bytes or a
        row-major iterable of ``width*height`` distances in meters.
        ``codec`` compresses the payload the v2 way (flags bit1 + codec id in
        header byte 29): ``"zlib"`` is raw DEFLATE, ``"lzfse"`` needs the
        optional ``liblzfse`` package. ``None`` sends raw — which a v2 server
        may legitimately do even after a client opted in.
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
        flags = 1  # bit0: samples are float16
        codec_id = 0
        if codec == "zlib":
            compressor = zlib.compressobj(wbits=-15)  # raw DEFLATE, RFC 1951
            samples = compressor.compress(bytes(samples)) + compressor.flush()
            flags |= 2  # bit1: payload compressed
            codec_id = 2
        elif codec == "lzfse":
            import liblzfse

            samples = liblzfse.compress(bytes(samples))
            flags |= 2
            codec_id = 1
        elif codec is not None:
            raise ValueError(f"unknown mock depth codec {codec!r}")
        header = bytearray(DEPTH_HEADER_SIZE)
        header[0] = T_DEPTH
        header[1] = flags
        struct.pack_into("<H", header, 2, seq & 0xFFFF)
        # 4..8 reserved
        struct.pack_into("<d", header, 8, host_ts)
        struct.pack_into("<d", header, 16, unix_ts)
        struct.pack_into("<HH", header, 24, width, height)
        header[28] = 2  # bytesPerPixel
        header[29] = codec_id
        # 30..31 pad
        self._depth.broadcast(length_prefixed(bytes(header) + bytes(samples)))
        return seq


# --------------------------------------------------------------------------- #
# RTSP audio track
#
# Byte-faithful to RTSPConnection.swift / SDPBuilder.swift / RTPCore.swift /
# RTCP.swift: the RTSP handshake, `$`-framed interleaved RTP, and compound
# SR+SDES Sender Reports.
# --------------------------------------------------------------------------- #

#: RTP payload type the app uses for audio, whatever the codec.
AUDIO_PT = 97

#: Seconds between the NTP epoch (1900) and the Unix epoch (1970).
NTP_UNIX_DELTA = 2_208_988_800


def interleaved(channel: int, packet: bytes) -> bytes:
    """Frame a packet the way the RTSP server does: ``$``, channel, BE16 length."""
    return b"$" + bytes([channel]) + struct.pack(">H", len(packet)) + packet


def rtp_packet(
    payload: bytes,
    *,
    seq: int,
    timestamp: int,
    ssrc: int,
    marker: bool = False,
    payload_type: int = AUDIO_PT,
    csrcs: Sequence[int] = (),
    extension: bytes | None = None,
    padding: int = 0,
) -> bytes:
    """One RTP packet.

    The app only ever emits the plain 12-byte header (V=2, no padding, no
    extension, no CSRCs); ``csrcs``/``extension``/``padding`` exist so the
    parser can be tested against the headers the RFC permits.
    """
    flags = 0x80 | (0x20 if padding else 0) | (0x10 if extension is not None else 0) | len(csrcs)
    out = struct.pack(
        ">BBHII", flags, (0x80 if marker else 0) | payload_type,
        seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF,
    )
    for csrc in csrcs:
        out += struct.pack(">I", csrc & 0xFFFFFFFF)
    if extension is not None:
        assert len(extension) % 4 == 0, "an RTP extension is a whole number of words"
        out += struct.pack(">HH", 0xBEDE, len(extension) // 4) + extension
    out += payload
    if padding:
        out += b"\x00" * (padding - 1) + bytes([padding])
    return out


def sender_report(
    *,
    ssrc: int,
    ntp_unix: float,
    rtp_timestamp: int,
    packet_count: int = 0,
    octet_count: int = 0,
    cname: str = "iRTSP@127.0.0.1",
    with_sdes: bool = True,
) -> bytes:
    """A 28-byte SR, compound with an SDES behind it exactly like the app's."""
    seconds = int(ntp_unix) + NTP_UNIX_DELTA
    fraction = int((ntp_unix - int(ntp_unix)) * 2**32) & 0xFFFFFFFF
    sr = struct.pack(
        ">BBHIIIIII", 0x80, 200, 6, ssrc & 0xFFFFFFFF, seconds, fraction,
        rtp_timestamp & 0xFFFFFFFF, packet_count, octet_count,
    )
    if not with_sdes:
        return sr
    items = bytes([1, len(cname)]) + cname.encode("utf-8") + b"\x00"
    while (4 + len(items)) % 4:
        items += b"\x00"
    sdes = struct.pack(">BBHI", 0x81, 202, (8 + len(items)) // 4 - 1, ssrc & 0xFFFFFFFF)
    return sr + sdes + items


class MockAudioPhone:
    """Mock of the iRTSP RTSP server's **audio** track on a 127.0.0.1 ephemeral port.

    Serves OPTIONS/DESCRIBE/SETUP/PLAY/TEARDOWN (with optional Basic or Digest
    auth), then streams whatever the test emits. Timestamps and sequence numbers
    advance themselves, so a test only names the *anomalies*::

        mic.emit_l16([0] * 480)                       # contiguous
        mic.emit_l16([0] * 480, gap_frames=960, marker=True)   # a capture gap
        mic.emit_l16([0] * 480, lost=2)               # two packets never sent
    """

    def __init__(
        self,
        *,
        codec: str = "l16",
        sample_rate: int = 48000,
        channels: int = 1,
        path: str = "live",
        auth: str | None = None,
        username: str = "alice",
        password: str = "s3cret",
        realm: str = "iRTSP",
        nonce: str = "bm9uY2UtZm9yLXRlc3Rz",
        ssrc: int = 0x1A2B3C4D,
        start_seq: int = 1000,
        start_timestamp: int = 0,
        fmtp: str | None = None,
        video: bool = True,
        session_id: str = "0BADCAFE",
    ):
        self.codec = codec.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.path = path
        self.auth = auth  # None | "basic" | "digest"
        self.username = username
        self.password = password
        self.realm = realm
        self.nonce = nonce
        self.ssrc = ssrc
        self.video = video  # the SDP's video m-line, which the client must skip
        self.session_id = session_id
        self.fmtp = fmtp if fmtp is not None else self._default_fmtp()

        #: Every request line the client sent, as ``(method, url)``.
        self.requests: list[tuple[str, str]] = []
        #: Every ``Authorization`` header seen (``None`` when it sent none).
        self.authorizations: list[str | None] = []
        self.challenges_sent = 0
        self.played = threading.Event()
        self.torn_down = threading.Event()

        self._next_seq = start_seq & 0xFFFF
        self._next_ts = start_timestamp & 0xFFFFFFFF
        self._rtp_channel = 0
        self._rtcp_channel = 1
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "MockAudioPhone":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        srv.settimeout(0.1)  # poll `_closed` so close() is always prompt
        self._listener = srv
        self._thread = threading.Thread(
            target=self._accept_loop, name="mock-rtsp-accept", daemon=True
        )
        self._thread.start()
        return self

    @property
    def port(self) -> int:
        assert self._listener is not None, "mock RTSP server not started"
        return self._listener.getsockname()[1]

    @property
    def content_base(self) -> str:
        """``Content-Base``, with the trailing slash the app sends."""
        return f"rtsp://127.0.0.1:{self.port}/{self.path}/"

    def end_stream(self) -> None:
        """Stop sending (half-close), so the client sees a clean end of stream.

        Everything already written is delivered first — the client's final,
        partially-filled block still arrives.
        """
        conn = self._conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------- RTSP

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._conn = conn
            threading.Thread(
                target=self._serve, args=(conn,), name="mock-rtsp-serve", daemon=True
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        reader = conn.makefile("rb")
        while not self._closed:
            try:
                line = reader.readline()
            except OSError:
                return
            if not line:
                return
            request = line.decode("utf-8", "replace").strip()
            if not request:
                continue
            method, _, rest = request.partition(" ")
            url = rest.split(" ")[0]
            headers: dict[str, str] = {}
            while True:
                raw = reader.readline()
                if not raw or raw in (b"\r\n", b"\n"):
                    break
                key, _, value = raw.decode("utf-8", "replace").partition(":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", 0) or 0)
            if length:
                reader.read(length)
            self.requests.append((method, url))
            self._handle(conn, method, url, headers)

    def _handle(
        self, conn: socket.socket, method: str, url: str, headers: dict[str, str]
    ) -> None:
        cseq = headers.get("cseq", "0")
        session = [("Session", self.session_id)]
        if method == "OPTIONS":  # the app never authenticates OPTIONS
            self._respond(conn, 200, "OK", cseq, [
                ("Public", "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, "
                           "GET_PARAMETER, SET_PARAMETER"),
            ])
        elif method == "DESCRIBE":
            if not self._authorized(method, url, headers):
                return self._challenge(conn, cseq)
            body = self.sdp().encode("utf-8")
            self._respond(conn, 200, "OK", cseq, [
                ("Content-Type", "application/sdp"),
                ("Content-Base", self.content_base),
            ], body)
        elif method == "SETUP":
            if not self._authorized(method, url, headers):
                return self._challenge(conn, cseq)
            transport = headers.get("transport", "")
            assert "RTP/AVP/TCP" in transport, f"expected a TCP transport, got {transport!r}"
            self._rtp_channel, self._rtcp_channel = _mock_interleaved(transport)
            self._respond(conn, 200, "OK", cseq, [
                ("Transport", f"RTP/AVP/TCP;unicast;interleaved="
                              f"{self._rtp_channel}-{self._rtcp_channel};"
                              f"ssrc={self.ssrc:08X}"),
                ("Session", f"{self.session_id};timeout=60"),
            ])
        elif method == "PLAY":
            if not self._authorized(method, url, headers):
                return self._challenge(conn, cseq)
            base = self.content_base.rstrip("/")
            self._respond(conn, 200, "OK", cseq, session + [
                ("Range", "npt=0.000-"),
                ("RTP-Info", f"url={base}/trackID=1;seq={self._next_seq};"
                             f"rtptime={self._next_ts}"),
            ])
            self.played.set()
        elif method == "TEARDOWN":
            self._respond(conn, 200, "OK", cseq, session)
            self.torn_down.set()
        elif method in ("PAUSE", "GET_PARAMETER", "SET_PARAMETER"):
            self._respond(conn, 200, "OK", cseq, session)
        else:
            self._respond(conn, 405, "Method Not Allowed", cseq, [])

    def _respond(
        self,
        conn: socket.socket,
        status: int,
        reason: str,
        cseq: str,
        extra: Sequence[tuple[str, str]],
        body: bytes = b"",
    ) -> None:
        """``RTSPMessage.make``: status line, CSeq, Server, extras, Content-Length."""
        lines = [f"RTSP/1.0 {status} {reason}", f"CSeq: {cseq}", "Server: iRTSP/1.0"]
        lines += [f"{key}: {value}" for key, value in extra]
        if body:
            lines.append(f"Content-Length: {len(body)}")
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
        try:
            with self._send_lock:
                conn.sendall(head + body)
        except OSError:  # the client hung up (or we already half-closed)
            pass

    def _challenge(self, conn: socket.socket, cseq: str) -> None:
        """The app's 401: Digest **and** Basic, one header line each."""
        self.challenges_sent += 1
        if self.auth == "digest":
            extra = [
                ("WWW-Authenticate",
                 f'Digest realm="{self.realm}", nonce="{self.nonce}", algorithm=MD5'),
                ("WWW-Authenticate", f'Basic realm="{self.realm}"'),
            ]
        else:
            extra = [("WWW-Authenticate", f'Basic realm="{self.realm}"')]
        self._respond(conn, 401, "Unauthorized", cseq, extra)

    def _authorized(self, method: str, url: str, headers: dict[str, str]) -> bool:
        header = headers.get("authorization")
        self.authorizations.append(header)
        if self.auth is None:
            return True
        if not header:
            return False
        scheme, _, rest = header.partition(" ")
        scheme = scheme.lower()
        if scheme == "basic":  # accepted in digest mode too, like DigestAuth.swift
            expected = base64.b64encode(
                f"{self.username}:{self.password}".encode("utf-8")
            ).decode()
            return rest.strip() == expected
        if scheme == "digest" and self.auth == "digest":
            params: dict[str, str] = {}
            for chunk in rest.split(","):
                key, sep, value = chunk.strip().partition("=")
                if sep:
                    params[key.strip().lower()] = value.strip().strip('"')
            if params.get("realm") != self.realm or params.get("nonce") != self.nonce:
                return False
            if params.get("username") != self.username:
                return False
            ha1 = _md5(f"{self.username}:{self.realm}:{self.password}")
            ha2 = _md5(f"{method}:{params.get('uri', '')}")
            if params.get("qop") == "auth":
                expected = _md5(f"{ha1}:{self.nonce}:{params.get('nc', '')}:"
                                f"{params.get('cnonce', '')}:auth:{ha2}")
            else:  # RFC 2069, which is what the app's challenge asks for
                expected = _md5(f"{ha1}:{self.nonce}:{ha2}")
            return params.get("response", "").lower() == expected
        return False

    # -------------------------------------------------------------------- SDP

    def _rtpmap(self) -> str:
        if self.codec == "aac":
            return f"mpeg4-generic/{self.sample_rate}/{self.channels}"
        if self.codec == "opus":
            return "opus/48000/2"  # hardcoded by the app, whatever was captured
        return f"L16/{self.sample_rate}/{self.channels}"

    def _default_fmtp(self) -> str | None:
        if self.codec == "aac":
            return ("streamtype=5;profile-level-id=1;mode=AAC-hbr;sizeLength=13;"
                    "indexLength=3;indexDeltaLength=3;config=1190")
        if self.codec == "opus":
            return f"sprop-stereo={1 if self.channels > 1 else 0};useinbandfec=0"
        return None

    def sdp(self) -> str:
        """The app's SDP: session block, the video track, then the audio track."""
        lines = [
            "v=0",
            f"o=- {self.session_id} {self.session_id} IN IP4 127.0.0.1",
            "s=iRTSP Live",
            "i=iPhone camera",
            "c=IN IP4 0.0.0.0",
            "t=0 0",
            "a=tool:iRTSP 1.0",
            "a=type:broadcast",
            "a=control:*",
            "a=range:npt=0-",
        ]
        if self.video:
            lines += [
                "m=video 0 RTP/AVP 96",
                "a=rtpmap:96 H264/90000",
                "a=fmtp:96 packetization-mode=1;profile-level-id=640028",
                "a=control:trackID=0",
            ]
        lines += [f"m=audio 0 RTP/AVP {AUDIO_PT}", f"a=rtpmap:{AUDIO_PT} {self._rtpmap()}"]
        if self.fmtp:
            lines.append(f"a=fmtp:{AUDIO_PT} {self.fmtp}")
        lines.append("a=control:trackID=1")
        return "\r\n".join(lines) + "\r\n"

    # ------------------------------------------------------------------ media

    def wait_for_play(self, timeout: float = 2.0) -> bool:
        return self.played.wait(timeout)

    def _write(self, channel: int, packet: bytes) -> None:
        assert self.wait_for_play(), "no client has sent PLAY"
        conn = self._conn
        assert conn is not None
        with self._send_lock:
            conn.sendall(interleaved(channel, packet))

    def emit_rtp(
        self,
        payload: bytes,
        *,
        frames: int,
        marker: bool = False,
        gap_frames: int = 0,
        lost: int = 0,
        seq: int | None = None,
        timestamp: int | None = None,
        payload_type: int = AUDIO_PT,
        ssrc: int | None = None,
        **packet_kwargs: Any,
    ) -> tuple[int, int]:
        """One RTP packet; returns its ``(seq, timestamp)``.

        ``frames`` is how far the timestamp advances afterwards, ``gap_frames``
        opens a hole in the timeline before this packet (a capture gap), and
        ``lost`` skips that many sequence numbers — packets that were captured
        and never sent, so the timestamp skips their audio too (``lost``
        packets' worth of ``frames``).
        """
        if seq is None:
            seq = (self._next_seq + lost) & 0xFFFF
        if timestamp is None:
            timestamp = (self._next_ts + gap_frames + lost * frames) & 0xFFFFFFFF
        self._write(self._rtp_channel, rtp_packet(
            payload, seq=seq, timestamp=timestamp, ssrc=self.ssrc if ssrc is None else ssrc,
            marker=marker, payload_type=payload_type, **packet_kwargs,
        ))
        self._next_seq = (seq + 1) & 0xFFFF
        self._next_ts = (timestamp + frames) & 0xFFFFFFFF
        return seq, timestamp

    def emit_l16(self, samples: Sequence[int], **kwargs: Any) -> tuple[int, int]:
        """One L16 packet: interleaved int16 samples, **big-endian** on the wire."""
        assert len(samples) % self.channels == 0, "need whole frames"
        payload = struct.pack(f">{len(samples)}h", *samples)
        return self.emit_rtp(payload, frames=len(samples) // self.channels, **kwargs)

    def emit_sr(
        self,
        *,
        ntp_unix: float,
        rtp_timestamp: int | None = None,
        packet_count: int = 0,
        octet_count: int = 0,
        with_sdes: bool = True,
    ) -> None:
        """One compound RTCP Sender Report on the RTCP channel."""
        self._write(self._rtcp_channel, sender_report(
            ssrc=self.ssrc,
            ntp_unix=ntp_unix,
            rtp_timestamp=self._next_ts if rtp_timestamp is None else rtp_timestamp,
            packet_count=packet_count,
            octet_count=octet_count,
            with_sdes=with_sdes,
        ))

    def send_raw(self, data: bytes) -> None:
        """Escape hatch: push arbitrary bytes down the RTSP connection."""
        assert self.wait_for_play(), "no client has sent PLAY"
        conn = self._conn
        assert conn is not None
        with self._send_lock:
            conn.sendall(data)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _mock_interleaved(transport: str) -> tuple[int, int]:
    """The channels the client asked for — the app uses them verbatim."""
    for part in transport.split(";"):
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "interleaved" and value:
            first, _, second = value.partition("-")
            return int(first), int(second) if second else int(first) + 1
    raise AssertionError(f"no interleaved= in Transport {transport!r}")
