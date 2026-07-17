"""End-to-end tests for irtsp.aio against a scripted localhost server.

The mock server speaks the real wire protocol (INTEGRATION.md, and upstream
IMUStreamServer.swift / DepthStreamServer.swift / IMUWireFormat.swift):

* on connect: ``[u32 LE length][UTF-8 JSON handshake]``;
* odometry channel: back-to-back fixed 64-byte little-endian records;
* depth channel: ``[u32 LE length][32-byte header + half-float samples]``.

No pytest-asyncio — each test is a plain function driving ``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import struct
import zlib

import pytest

import irtsp
from irtsp import (
    DepthFrame,
    IMU,
    Pose,
    ProtocolError,
    STANDARD_GRAVITY,
    StreamClock,
    Vec3,
)

HOST_ANCHOR = 12_345.678
WALL_ANCHOR = 1_752_000_000.25

# --------------------------------------------------------------------------- #
# Wire encoding (mirrors IMUWireFormat.swift / DepthStreamServer.swift)
# --------------------------------------------------------------------------- #


def imu_record(
    seq: int,
    *,
    host_ts: float = 1.0,
    unix_ts: float = WALL_ANCHOR + 1.0,
    gyro: tuple[float, float, float] = (0.5, -1.25, 2.0),
    accel_g: tuple[float, float, float] = (0.25, -0.5, -1.0),
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> bytes:
    """One 64-byte type-1 (fused IMU) odometry record, little-endian.

    Header: u8 type | u8 flags | u16 seq | u32 reserved | f64 host_ts | f64 unix_ts.
    Payload @24: 10 x f32 = gyro xyz (rad/s), accel xyz (g), quat xyzw.
    """
    buf = bytearray(64)
    buf[0] = 1  # RecordType.IMU
    buf[1] = 0  # flags
    struct.pack_into("<H", buf, 2, seq & 0xFFFF)
    struct.pack_into("<I", buf, 4, 0)  # reserved
    struct.pack_into("<dd", buf, 8, host_ts, unix_ts)
    struct.pack_into("<10f", buf, 24, *gyro, *accel_g, *quat)
    return bytes(buf)


def depth_message(
    seq: int,
    *,
    width: int,
    height: int,
    values: list[float],
    host_ts: float = 2.0,
    unix_ts: float = WALL_ANCHOR + 2.0,
    compress: bool = False,
) -> bytes:
    """One length-prefixed depth-channel frame (u32 LE length + 32B header + f16s).

    ``compress=True`` sends the samples the v2 "zlib" way: raw DEFLATE payload,
    flags bit1 set, codec id 2 in header byte 29.
    """
    assert len(values) == width * height
    header = bytearray(32)
    header[0] = 10  # RecordType.DEPTH
    header[1] = 1  # flags bit0: samples are float16
    struct.pack_into("<H", header, 2, seq & 0xFFFF)
    struct.pack_into("<I", header, 4, 0)  # reserved
    struct.pack_into("<dd", header, 8, host_ts, unix_ts)
    struct.pack_into("<HH", header, 24, width, height)
    header[28] = 2  # bytesPerPixel
    samples = b"".join(struct.pack("<e", v) for v in values)
    if compress:
        header[1] |= 2  # flags bit1: payload compressed
        header[29] = 2  # codec id: zlib (raw DEFLATE, RFC 1951)
        compressor = zlib.compressobj(wbits=-15)
        samples = compressor.compress(samples) + compressor.flush()
    payload = bytes(header) + samples
    return struct.pack("<I", len(payload)) + payload


def imu_handshake() -> dict:
    """The odometry-channel handshake, shaped like IMUStreamServer.makeHandshake."""
    return {
        "protocol": "irtsp-imu",
        "version": 1,
        "endianness": "little",
        "record_bytes": 64,
        "mode": "fused",
        "rate_hz": 100,
        "gyro_units": "rad/s",
        "accel_units": "g",
        "accel_convention": "gravity+userAcceleration; face-up rest ~ (0,0,-1)",
        "attitude": "quaternion_xyzw",
        "attitude_frame": "xArbitraryZVertical",
        "body_axes": "X-right, Y-up, Z-out-of-screen",
        "clock": {
            "timebase": "mach_absolute_time_seconds",
            "host_anchor": HOST_ANCHOR,
            "wall_anchor": WALL_ANCHOR,
            "rtcp_sync": "unix_ts matches RTP RTCP SR NTP timeline",
        },
        "video": {
            "rtsp_url": "rtsp://192.168.1.24:8554/live",
            "clock_rate": 90000,
            "codec": "h264",
        },
        "lens_distortion": "none",
        "record_types": {
            "imu": 1, "gyro": 2, "accel": 3, "intrinsics": 5,
            "gnss": 6, "altitude": 7, "heading": 8, "pose": 9,
        },
        "streams": {
            "imu": True, "intrinsics": True, "gnss": False,
            "altitude": False, "heading": False, "pose": False,
        },
    }


def depth_handshake() -> dict:
    """The depth-channel handshake, shaped like DepthStreamServer.makeHandshake."""
    return {
        "protocol": "irtsp-depth",
        "version": 1,
        "endianness": "little",
        "frame_type": 10,
        "pixel_format": "depth_float16",
        "units": "meters",
        "clock": {
            "timebase": "mach_absolute_time_seconds",
            "host_anchor": HOST_ANCHOR,
            "wall_anchor": WALL_ANCHOR,
            "rtcp_sync": "unix_ts matches RTP RTCP SR NTP timeline",
        },
        "video": {"rtsp_url": "rtsp://192.168.1.24:8554/live", "codec": "h264"},
    }


def imu_handshake_v2() -> dict:
    """The odometry handshake a v2 app sends (state-channel contract added)."""
    handshake = imu_handshake()
    handshake["version"] = 2
    handshake["emission"] = {
        "imu": "continuous", "gyro": "continuous", "accel": "continuous",
        "pose": "continuous", "gnss": "event", "altitude": "event",
        "intrinsics": "state", "heading": "state",
    }
    handshake["state_channels"] = {
        "keyframe_interval_s": 10, "flags": {"bit0": "snapshot_or_keyframe"},
    }
    return handshake


def depth_handshake_v2(supported: tuple[str, ...] = ("lzfse", "zlib")) -> dict:
    """The depth handshake a v2 app sends (compression advertised)."""
    handshake = depth_handshake()
    handshake["version"] = 2
    handshake["compression"] = {"supported": list(supported)}
    return handshake


# --------------------------------------------------------------------------- #
# Scripted server + async plumbing
# --------------------------------------------------------------------------- #

_CLOSE = object()  # tells the handler to hang up on its client


class ScriptedServer:
    """A localhost asyncio.start_server speaking one iRTSP side-channel.

    Sends the JSON handshake on connect, then whatever bytes the test queues
    via :meth:`send`; :meth:`close_client` hangs up after the queued sends.
    """

    def __init__(self, handshake: dict):
        self.handshake = handshake
        self.port: int = 0
        self.received = bytearray()  # client→server bytes (v2 control messages)
        self._server: asyncio.Server | None = None
        self._queue: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self) -> "ScriptedServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._queue.put_nowait(_CLOSE)  # release a handler parked on the queue
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        inbox = asyncio.ensure_future(self._collect_client_bytes(reader))
        try:
            blob = json.dumps(self.handshake).encode("utf-8")
            writer.write(struct.pack("<I", len(blob)) + blob)
            await writer.drain()
            while True:
                item = await self._queue.get()
                if item is _CLOSE:
                    break
                writer.write(item)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            inbox.cancel()
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _collect_client_bytes(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    return
                self.received += chunk
        except (ConnectionError, asyncio.CancelledError):
            pass

    def control_messages(self) -> list[dict]:
        """Everything the client sent, parsed as [u32 LE length][JSON] messages."""
        out, offset = [], 0
        while offset + 4 <= len(self.received):
            (length,) = struct.unpack_from("<I", self.received, offset)
            if offset + 4 + length > len(self.received):
                break
            out.append(json.loads(bytes(self.received[offset + 4 : offset + 4 + length])))
            offset += 4 + length
        return out

    def send(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def close_client(self) -> None:
        self._queue.put_nowait(_CLOSE)


def async_test(coro):
    """Run an ``async def`` test to completion on a fresh event loop."""

    @functools.wraps(coro)
    def runner():
        asyncio.run(asyncio.wait_for(coro(), timeout=15.0))

    return runner


async def eventually(predicate, *, timeout: float = 5.0, what: str = "condition") -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.005)


async def collect(stream, n: int) -> list:
    out = []
    async for record in stream:
        out.append(record)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@async_test
async def test_connect_info_and_clock():
    async with ScriptedServer(imu_handshake()) as server:
        phone = await irtsp.connect_async("127.0.0.1", imu_port=server.port)
        try:
            info = phone.info
            assert info is not None
            assert info.protocol == "irtsp-imu"
            assert info.version == 1
            assert info.record_bytes == 64
            assert info.attitude_enabled is True
            assert info.streams["imu"] is True
            assert info.streams["gnss"] is False
            assert info.rate_hz == 100
            assert info.video_url == "rtsp://192.168.1.24:8554/live"
            assert info.video_codec == "h264"
            assert info.video_clock_rate == 90000

            clock = phone.clock
            assert isinstance(clock, StreamClock)
            assert clock.host_anchor == HOST_ANCHOR
            assert clock.wall_anchor == WALL_ANCHOR
            assert clock.to_unix(HOST_ANCHOR + 2.5) == pytest.approx(WALL_ANCHOR + 2.5)
            assert clock.to_host(clock.to_unix(7.0)) == pytest.approx(7.0)
        finally:
            await phone.aclose()


@async_test
async def test_async_for_imu_decodes_records():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            stream = phone.imu  # subscribe BEFORE the server emits anything
            half_sqrt2 = math.sqrt(0.5)
            server.send(imu_record(0, host_ts=1.0, unix_ts=WALL_ANCHOR + 1.0,
                                   quat=(0.0, 0.0, half_sqrt2, half_sqrt2)))
            server.send(imu_record(1, host_ts=1.01, unix_ts=WALL_ANCHOR + 1.01))
            server.send(imu_record(2, quat=(0.0, 0.0, 0.0, 0.0)))  # attitude off

            got = await collect(stream, 3)
            assert [r.seq for r in got] == [0, 1, 2]
            assert all(isinstance(r, IMU) for r in got)

            first = got[0]
            # f64 timestamps survive bit-exactly.
            assert first.host_ts == 1.0
            assert first.unix_ts == WALL_ANCHOR + 1.0
            # gyro rides the wire as-is (rad/s), values chosen f32-exact.
            assert first.gyro == Vec3(0.5, -1.25, 2.0)
            # accel: wire g -> SI m/s^2 via standard gravity.
            assert first.accel.x == pytest.approx(0.25 * STANDARD_GRAVITY)
            assert first.accel.y == pytest.approx(-0.5 * STANDARD_GRAVITY)
            assert first.accel.z == pytest.approx(-STANDARD_GRAVITY)
            assert first.accel_g.z == pytest.approx(-1.0)
            # quaternion arrives xyzw (f32 precision).
            assert first.quat is not None
            assert first.quat.z == pytest.approx(half_sqrt2, rel=1e-6)
            assert first.quat.w == pytest.approx(half_sqrt2, rel=1e-6)
            # zeroed quat slots decode to None.
            assert got[2].quat is None
            # no drops -> no gaps.
            assert [r.gap for r in got] == [0, 0, 0]


@async_test
async def test_multi_subscriber_fanout():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            s1 = phone.imu
            s2 = phone.stream(IMU)
            s3 = phone.odometry  # unfiltered — sees everything too
            for seq in range(3):
                server.send(imu_record(seq))

            g1 = await collect(s1, 3)
            g2 = await collect(s2, 3)
            g3 = await collect(s3, 3)
            assert [r.seq for r in g1] == [0, 1, 2]
            assert [r.seq for r in g2] == [0, 1, 2]
            assert [r.seq for r in g3] == [0, 1, 2]
            # Fan-out shares the record objects; consumers don't get copies.
            assert g1[0] is g2[0]
            assert g1[0] is g3[0]


@async_test
async def test_gap_detection():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            stream = phone.imu
            for seq in (0, 1, 5):  # seqs 2..4 went missing
                server.send(imu_record(seq))
            got = await collect(stream, 3)
            assert [r.seq for r in got] == [0, 1, 5]
            assert [r.gap for r in got] == [0, 0, 3]


@async_test
async def test_gap_detection_across_seq_wraparound():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            stream = phone.imu
            for seq in (65534, 65535, 0, 2):  # wraps 65535 -> 0, then drops seq 1
                server.send(imu_record(seq))
            got = await collect(stream, 4)
            assert [r.seq for r in got] == [65534, 65535, 0, 2]
            assert [r.gap for r in got] == [0, 0, 0, 1]


@async_test
async def test_drop_oldest_on_tiny_buffer():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            slow = phone.stream(IMU, buffer=2)  # a consumer that never keeps up
            for seq in range(5):
                server.send(imu_record(seq))
            # Wait (without consuming) until all 5 have been dispatched.
            await eventually(
                lambda: (r := phone.latest(IMU)) is not None and r.seq == 4,
                what="all 5 records dispatched",
            )
            assert slow.dropped == 3  # 0, 1, 2 fell off the front
            got = await collect(slow, 2)
            assert [r.seq for r in got] == [3, 4]  # newest survive


@async_test
async def test_latest():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            assert phone.latest(IMU) is None
            for seq in range(3):
                server.send(imu_record(seq, host_ts=1.0 + seq))
            await eventually(
                lambda: (r := phone.latest(IMU)) is not None and r.seq == 2,
                what="latest IMU to reach seq 2",
            )
            latest = phone.latest(IMU)
            assert isinstance(latest, IMU)
            assert latest.host_ts == 3.0
            assert phone.latest(Pose) is None  # never sent


@async_test
async def test_aclose_ends_iteration():
    async with ScriptedServer(imu_handshake()) as server:
        phone = await irtsp.connect_async("127.0.0.1", imu_port=server.port)
        stream = phone.imu
        server.send(imu_record(0))
        got = await collect(stream, 1)
        assert got[0].seq == 0

        await phone.aclose()
        assert phone.closed
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        with pytest.raises(StopAsyncIteration):  # stays ended
            await anext(stream)
        await phone.aclose()  # idempotent


@async_test
async def test_stream_created_after_close_ends_immediately():
    async with ScriptedServer(imu_handshake()) as server:
        phone = await irtsp.connect_async("127.0.0.1", imu_port=server.port)
        await phone.aclose()
        late = phone.imu
        with pytest.raises(StopAsyncIteration):
            await anext(late)


@async_test
async def test_async_context_manager():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            assert not phone.closed
            stream = phone.imu
            server.send(imu_record(7))
            got = await collect(stream, 1)
            assert got[0].seq == 7
        assert phone.closed  # __aexit__ closed the session
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


@async_test
async def test_server_close_ends_iterators():
    async with ScriptedServer(imu_handshake()) as server:
        phone = await irtsp.connect_async("127.0.0.1", imu_port=server.port)
        try:
            stream = phone.imu
            server.send(imu_record(0))
            server.send(imu_record(1))
            server.close_client()  # phone hangs up after the two records

            got = [record async for record in stream]  # runs to exhaustion
            assert [r.seq for r in got] == [0, 1]
            assert phone.closed
        finally:
            await phone.aclose()


@async_test
async def test_depth_channel():
    row0 = [0.25, 0.5, 1.0, 1.5]
    row1 = [2.0, 4.0, 0.75, 8.0]  # all f16-exact
    async with ScriptedServer(imu_handshake()) as imu_srv:
        async with ScriptedServer(depth_handshake()) as depth_srv:
            phone = await irtsp.connect_async(
                "127.0.0.1", imu_port=imu_srv.port, depth=True, depth_port=depth_srv.port
            )
            try:
                assert phone.depth_info is not None
                assert phone.depth_info.protocol == "irtsp-depth"
                # Both channels carry the SAME session anchors.
                assert phone.depth_info.clock.host_anchor == HOST_ANCHOR
                assert phone.depth_info.clock.wall_anchor == WALL_ANCHOR

                stream = phone.depth
                depth_srv.send(
                    depth_message(3, width=4, height=2, values=row0 + row1,
                                  host_ts=2.0, unix_ts=WALL_ANCHOR + 2.0)
                )
                depth_srv.send(
                    depth_message(4, width=4, height=2, values=row0 + row1)
                )

                frames = await collect(stream, 2)
                frame = frames[0]
                assert isinstance(frame, DepthFrame)
                assert (frame.width, frame.height) == (4, 2)
                assert frame.seq == 3
                assert frame.host_ts == 2.0
                assert frame.unix_ts == WALL_ANCHOR + 2.0
                # Row-major sample layout, meters.
                assert frame.at(0, 0) == 0.25
                assert frame.at(3, 0) == 1.5
                assert frame.at(0, 1) == 2.0
                assert frame.at(2, 1) == 0.75

                import numpy as np

                meters = frame.meters
                assert meters.shape == (2, 4)
                assert meters.dtype == np.float32
                assert meters.tolist() == [row0, row1]

                assert frames[1].seq == 4
                assert frames[1].gap == 0  # consecutive depth seqs -> no gap
            finally:
                await phone.aclose()


@async_test
async def test_handshake_v2_accepted():
    async with ScriptedServer(imu_handshake_v2()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            assert phone.info is not None
            assert phone.info.version == 2
            assert phone.info.emission["intrinsics"] == "state"
            assert phone.info.state_channels["keyframe_interval_s"] == 10
            # a v2 session streams exactly like v1
            stream = phone.imu
            server.send(imu_record(0))
            got = await collect(stream, 1)
            assert got[0].seq == 0


@async_test
async def test_depth_compression_negotiated_and_decoded():
    async with ScriptedServer(imu_handshake()) as imu_srv:
        async with ScriptedServer(depth_handshake_v2()) as depth_srv:
            phone = await irtsp.connect_async(
                "127.0.0.1", imu_port=imu_srv.port, depth=True,
                depth_port=depth_srv.port, depth_compression="zlib",
            )
            try:
                assert phone.depth_codec == "zlib"
                # the exact wire bytes: [u32 LE length][UTF-8 JSON]
                await eventually(
                    lambda: depth_srv.control_messages() == [{"compression": "zlib"}],
                    what="the compression opt-in to arrive",
                )
                # a compressed frame and a raw fallback frame both decode —
                # decoding follows per-frame flags, never the negotiated codec
                values = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
                stream = phone.depth
                depth_srv.send(depth_message(1, width=3, height=2, values=values,
                                             compress=True))
                depth_srv.send(depth_message(2, width=3, height=2, values=values))
                frames = await collect(stream, 2)
                assert frames[0].at(2, 1) == 4.0
                assert frames[0].data == frames[1].data
            finally:
                await phone.aclose()


@async_test
async def test_depth_compression_not_offered_to_v1_server():
    async with ScriptedServer(imu_handshake()) as imu_srv:
        async with ScriptedServer(depth_handshake()) as depth_srv:  # v1: no key
            phone = await irtsp.connect_async(
                "127.0.0.1", imu_port=imu_srv.port, depth=True,
                depth_port=depth_srv.port, depth_compression="zlib",
            )
            try:
                assert phone.depth_codec is None
                await asyncio.sleep(0.2)
                assert depth_srv.received == b""  # nothing sent, ever
            finally:
                await phone.aclose()


@async_test
async def test_depth_property_requires_flag():
    async with ScriptedServer(imu_handshake()) as server:
        async with await irtsp.connect_async("127.0.0.1", imu_port=server.port) as phone:
            with pytest.raises(RuntimeError):
                phone.depth


# When open() fails mid-handshake, AsyncSession leaves its already-opened
# connection for the GC (never aclose()d) — harmless here, but noisy under
# -W error::ResourceWarning, so silence that one warning for this test.
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@async_test
async def test_implausible_handshake_length_raises():
    async def bogus(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(struct.pack("<I", 0x7FFF_FFFF))  # way over the 64 MiB cap
        await writer.drain()

    server = await asyncio.start_server(bogus, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(ProtocolError):
            await irtsp.connect_async("127.0.0.1", imu_port=port)
    finally:
        server.close()
        await server.wait_closed()
