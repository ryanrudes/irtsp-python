"""asyncio flavor of the client — same shapes, ``async`` everywhere.

::

    import asyncio, irtsp

    async def main():
        async with await irtsp.connect_async("192.168.1.24") as phone:
            async for imu in phone.imu:
                print(imu.gyro, imu.accel)

    asyncio.run(main())

This is a native asyncio implementation (not a thread wrapper): one reader task
per channel, fanning records out to independent bounded queues with the same
drop-oldest-per-consumer policy as the sync :class:`~irtsp.Session`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from .clock import StreamClock
from .records import (
    Altitude,
    DepthFrame,
    GNSS,
    Heading,
    IMU,
    Intrinsics,
    Pose,
    Record,
)
from .session import DEFAULT_BUFFER, Handshake, _check_handshake
from .wire import (
    MAX_MESSAGE,
    RECORD_SIZE,
    ProtocolError,
    decode_depth_frame,
    decode_record,
)

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import Device

__all__ = ["connect_async", "AsyncSession", "AsyncRecordStream"]

log = logging.getLogger("irtsp.aio")

R = TypeVar("R", bound=Record)

_EOS: Any = object()  # end-of-stream sentinel


async def _recv_handshake(reader: asyncio.StreamReader) -> Handshake:
    (length,) = struct.unpack("<I", await reader.readexactly(4))
    if not 0 < length <= MAX_MESSAGE:
        raise ProtocolError(f"implausible handshake length {length}")
    return Handshake(json.loads(await reader.readexactly(length)))


class AsyncRecordStream(Generic[R]):
    """Async-iterable, filtered view of the session's records."""

    def __init__(self, session: "AsyncSession", types: tuple[type, ...], maxsize: int):
        self._session = session
        self.types = types
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0
        self._ended = False

    @property
    def dropped(self) -> int:
        """Records this consumer lost because it wasn't keeping up."""
        return self._dropped

    def _put(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover — racy but harmless
                pass
            self._queue.put_nowait(item)

    def pop_all(self) -> list[R]:
        """Drain everything buffered right now, without blocking."""
        out: list[R] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return out
            if item is _EOS:
                self._ended = True
                self._queue.put_nowait(_EOS)  # keep the sentinel for __anext__
                return out
            out.append(item)

    def close(self) -> None:
        """Unsubscribe. Any pending ``async for`` ends."""
        self._session._unsubscribe(self)
        self._put(_EOS)

    def __aiter__(self) -> "AsyncRecordStream[R]":
        return self

    async def __anext__(self) -> R:
        if self._ended:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _EOS:
            self._ended = True
            raise StopAsyncIteration
        return item


class AsyncSession:
    """A live async connection. Prefer :func:`irtsp.connect_async`."""

    def __init__(self, host: str, *, imu_port: int = 8555, depth: bool = False,
                 depth_port: int = 8556, buffer: int = DEFAULT_BUFFER):
        self.host = host
        self.imu_port = imu_port
        self.depth_enabled = depth
        self.depth_port = depth_port
        self._buffer = buffer

        self.info: Handshake | None = None
        self.depth_info: Handshake | None = None

        self._subs: list[AsyncRecordStream[Any]] = []
        self._callbacks: list[tuple[tuple[type, ...], Callable[[Any], None]]] = []
        self._latest: dict[type, Record] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._writers: list[asyncio.StreamWriter] = []
        self._closed = False

    # ------------------------------------------------------------------ setup

    async def open(self) -> "AsyncSession":
        try:
            reader, writer = await asyncio.open_connection(self.host, self.imu_port)
            self._writers.append(writer)
            info = await _recv_handshake(reader)
            _check_handshake(info, "irtsp-imu")
            self.info = info
            self._tasks.append(
                asyncio.create_task(self._record_loop(reader), name="irtsp-odometry")
            )

            if self.depth_enabled:
                dreader, dwriter = await asyncio.open_connection(self.host, self.depth_port)
                self._writers.append(dwriter)
                depth_info = await _recv_handshake(dreader)
                _check_handshake(depth_info, "irtsp-depth")
                self.depth_info = depth_info
                self._tasks.append(
                    asyncio.create_task(self._depth_loop(dreader), name="irtsp-depth")
                )
        except BaseException:
            await self.aclose()  # tear down whatever already started
            raise
        return self

    # ------------------------------------------------------------ reader tasks

    async def _record_loop(self, reader: asyncio.StreamReader) -> None:
        last_seq: int | None = None
        try:
            while True:
                buf = await reader.readexactly(RECORD_SIZE)
                record = decode_record(buf)
                # Server replays the latest Intrinsics (stale seq) to late
                # joiners — don't let it baseline the gap tracker.
                if last_seq is None and isinstance(record, Intrinsics):
                    self._dispatch(record, None)
                    continue
                last_seq = self._dispatch(record, last_seq)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except ProtocolError as e:
            log.error("odometry stream desynced: %s", e)
        finally:
            await self.aclose()

    async def _depth_loop(self, reader: asyncio.StreamReader) -> None:
        last_seq: int | None = None
        try:
            while True:
                (length,) = struct.unpack("<I", await reader.readexactly(4))
                if not 0 < length <= MAX_MESSAGE:
                    raise ProtocolError(f"implausible depth frame length {length}")
                frame = decode_depth_frame(await reader.readexactly(length))
                last_seq = self._dispatch(frame, last_seq)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except ProtocolError as e:
            log.error("depth stream desynced: %s", e)
        finally:
            await self.aclose()

    def _dispatch(self, record: Record, last_seq: int | None) -> int:
        if last_seq is not None:
            gap = (record.seq - last_seq - 1) & 0xFFFF
            if gap:
                record = replace(record, gap=gap)
        self._latest[type(record)] = record
        for sub in self._subs:
            if isinstance(record, sub.types):
                sub._put(record)
        for types, fn in self._callbacks:
            if isinstance(record, types):
                try:
                    fn(record)
                except Exception:
                    log.exception("irtsp callback %r raised", fn)
        return record.seq

    # -------------------------------------------------------------- consuming

    def stream(self, *types: type[R], buffer: int | None = None) -> AsyncRecordStream[R]:
        """A new independent async iterator over the given record types."""
        sub: AsyncRecordStream[R] = AsyncRecordStream(
            self, types or (Record,), buffer or self._buffer
        )
        if self._closed:
            sub._put(_EOS)
        else:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: AsyncRecordStream[Any]) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    @property
    def odometry(self) -> AsyncRecordStream[Record]:
        return self.stream()

    @property
    def imu(self) -> AsyncRecordStream[IMU]:
        return self.stream(IMU)

    @property
    def intrinsics(self) -> AsyncRecordStream[Intrinsics]:
        return self.stream(Intrinsics)

    @property
    def gnss(self) -> AsyncRecordStream[GNSS]:
        return self.stream(GNSS)

    @property
    def altitude(self) -> AsyncRecordStream[Altitude]:
        return self.stream(Altitude)

    @property
    def heading(self) -> AsyncRecordStream[Heading]:
        return self.stream(Heading)

    @property
    def pose(self) -> AsyncRecordStream[Pose]:
        return self.stream(Pose)

    @property
    def depth(self) -> AsyncRecordStream[DepthFrame]:
        if not self.depth_enabled:
            raise RuntimeError("depth channel not enabled — connect_async(host, depth=True)")
        return self.stream(DepthFrame)

    def on(self, types: type[R] | tuple[type[R], ...], callback: Callable[[R], None]) -> None:
        """Call ``callback(record)`` for matching records (on the event loop)."""
        filt = types if isinstance(types, tuple) else (types,)
        self._callbacks.append((filt, callback))

    def latest(self, rtype: type[R]) -> R | None:
        """The most recent record of exactly ``rtype`` seen, or ``None``."""
        return self._latest.get(rtype)  # type: ignore[return-value]

    @property
    def clock(self) -> StreamClock:
        if self.info is None:
            raise RuntimeError("session is not open")
        return self.info.clock

    # ---------------------------------------------------------------- teardown

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Close connections and end all iterators. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for writer in self._writers:
            try:
                writer.close()
            except Exception:  # pragma: no cover
                pass
        for sub in self._subs:
            sub._put(_EOS)
        # Cancel and await the reader tasks — except the current one: the
        # loops call aclose() from their own finally, and awaiting yourself
        # deadlocks.
        current = asyncio.current_task()
        others = [t for t in self._tasks if t is not current]
        for task in others:
            task.cancel()
        if others:
            await asyncio.gather(*others, return_exceptions=True)
        for writer in self._writers:
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover — transport already gone
                pass

    async def __aenter__(self) -> "AsyncSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:  # pragma: no cover
        state = "closed" if self._closed else "live"
        return f"<irtsp.AsyncSession {self.host}:{self.imu_port} [{state}]>"


async def connect_async(
    target: "str | Device",
    *,
    imu_port: int = 8555,
    depth: bool = False,
    depth_port: int = 8556,
    buffer: int = DEFAULT_BUFFER,
) -> AsyncSession:
    """Async twin of :func:`irtsp.connect` (odometry + depth channels)."""
    host = target
    if not isinstance(target, str):
        host = target.host
        ports = getattr(target, "ports", {})
        imu_port = ports.get("imu", imu_port)
        depth_port = ports.get("depth", depth_port)
    return await AsyncSession(
        host,  # type: ignore[arg-type]
        imu_port=imu_port, depth=depth, depth_port=depth_port, buffer=buffer
    ).open()
