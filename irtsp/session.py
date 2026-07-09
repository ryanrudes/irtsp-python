"""The live connection: :func:`connect`, :class:`Session`, and record streams.

The 3-line happy path::

    import irtsp

    with irtsp.connect("192.168.1.24") as phone:
        for imu in phone.imu:
            print(imu.gyro, imu.accel)

A :class:`Session` owns one background reader per channel (odometry always,
depth when asked). Records fan out to any number of independent consumers —
filtered iterators (``phone.imu``, ``phone.gnss``, …) and/or callbacks
(:meth:`Session.on`) — each with its own bounded buffer. A slow consumer drops
its *own* oldest records (counted on :attr:`RecordStream.dropped`) and never
stalls capture or the other consumers, mirroring the server's fire-and-forget
design.

Note that each access to a stream property (``phone.imu`` etc.) returns a
**fresh, independent subscription** — iterate the same object, not the
property, if you want one continuous stream. Abandoned streams are garbage
collected; close them explicitly (or use them as context managers) to
unsubscribe deterministically.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, TypeVar

from .clock import StreamClock
from .records import (
    Altitude,
    DepthFrame,
    GNSS,
    Heading,
    IMU,
    Intrinsics,
    Pose,
    RawAccel,
    RawGyro,
    Record,
)
from .wire import (
    RECORD_SIZE,
    ConnectionClosed,
    ProtocolError,
    decode_depth_frame,
    decode_record,
    read_exact,
    recv_handshake,
    recv_length_prefixed,
)

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import Device

__all__ = ["connect", "Session", "Handshake", "RecordStream"]

log = logging.getLogger("irtsp")

R = TypeVar("R", bound=Record)

#: Records buffered per consumer before its oldest are dropped.
DEFAULT_BUFFER = 8192


class Handshake:
    """The server's JSON handshake, with the useful bits as attributes.

    Everything the server sent is preserved in :attr:`raw` — this class never
    hides fields it doesn't know about.
    """

    def __init__(self, raw: Mapping[str, Any]):
        self.raw: dict[str, Any] = dict(raw)
        self.protocol: str = str(raw.get("protocol", ""))
        self.version: int = int(raw.get("version", 1))
        self.record_bytes: int = int(raw.get("record_bytes", RECORD_SIZE))
        self.clock: StreamClock = StreamClock.from_handshake(raw)
        #: Which optional streams this session has enabled, e.g. ``{"gnss": True, ...}``.
        self.streams: dict[str, bool] = dict(raw.get("streams", {}))
        video = raw.get("video", {}) or {}
        self.video_url: str | None = video.get("rtsp_url")
        self.video_codec: str | None = video.get("codec")
        self.video_clock_rate: int | None = video.get("clock_rate")
        #: Requested IMU target rate. The **true** rate is lower on iPhone
        #: (fused motion caps ~100 Hz) — measure from ``host_ts`` deltas.
        self.rate_hz: float | None = raw.get("rate_hz")
        self.attitude_enabled: bool = raw.get("attitude", "none") != "none"

    def __repr__(self) -> str:  # pragma: no cover
        on = ", ".join(k for k, v in sorted(self.streams.items()) if v) or "-"
        return f"<Handshake {self.protocol} v{self.version} streams: {on}>"


def _check_handshake(info: Handshake, expected_protocol: str) -> None:
    """Fail fast (and clearly) when connected to the wrong port or an
    incompatible future server, instead of desyncing into garbage."""
    if info.protocol and info.protocol != expected_protocol:
        raise ProtocolError(
            f"expected an {expected_protocol!r} handshake but got {info.protocol!r} — "
            "connected to the wrong port?"
        )
    if expected_protocol == "irtsp-imu" and info.record_bytes != RECORD_SIZE:
        raise ProtocolError(
            f"server uses {info.record_bytes}-byte records; this client only "
            f"understands {RECORD_SIZE} — upgrade the irtsp package"
        )
    if info.version > 1:
        log.warning(
            "server speaks %s v%d, this client knows v1 — continuing, unknown "
            "record types will surface as irtsp.Unknown",
            info.protocol or expected_protocol,
            info.version,
        )


class RecordStream(Iterator[R]):
    """An independent, filtered, iterable view of a session's records.

    Each stream has its own bounded buffer: iterate it (blocking), or call
    :meth:`pop_all` for a non-blocking drain. Create as many as you like — they
    don't compete. Iteration ends when the stream or its session is closed.
    Works as a context manager (``with phone.imu as imu: ...``) to unsubscribe
    deterministically.
    """

    def __init__(self, session: "Session", types: tuple[type, ...], maxlen: int):
        self._session = session
        self.types = types
        self._buf: list[R] = []
        self._maxlen = maxlen
        self._cond = threading.Condition()
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        """Records this consumer lost because it wasn't keeping up."""
        return self._dropped

    def _put(self, record: R) -> None:
        with self._cond:
            if len(self._buf) >= self._maxlen:
                del self._buf[0]
                self._dropped += 1
            self._buf.append(record)
            self._cond.notify()

    def pop_all(self) -> list[R]:
        """Drain everything buffered right now, without blocking."""
        with self._cond:
            out, self._buf = self._buf, []
            return out

    def close(self) -> None:
        """Unsubscribe. Any blocked iteration raises ``StopIteration``."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._session._unsubscribe(self)

    def __enter__(self) -> "RecordStream[R]":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> "RecordStream[R]":
        return self

    def __next__(self) -> R:
        while True:
            with self._cond:
                if self._buf:
                    return self._buf.pop(0)
                if self._closed or self._session.closed:
                    raise StopIteration
                self._cond.wait(timeout=0.25)


class Session:
    """A live connection to one iRTSP phone. Prefer :func:`irtsp.connect`."""

    def __init__(
        self,
        host: str,
        *,
        imu_port: int = 8555,
        depth: bool = False,
        depth_port: int = 8556,
        video: bool = False,
        video_url: str | None = None,
        video_auth: tuple[str, str] | None = None,
        timeout: float = 5.0,
        reconnect: bool = False,
        buffer: int = DEFAULT_BUFFER,
    ):
        self.host = host
        self.imu_port = imu_port
        self.depth_enabled = depth
        self.depth_port = depth_port
        self.video_enabled = video
        self._video_url_override = video_url
        self._video_auth = video_auth
        self.timeout = timeout
        self.reconnect = reconnect
        self._buffer = buffer

        self.info: Handshake | None = None  #: odometry-channel handshake
        self.depth_info: Handshake | None = None  #: depth-channel handshake

        self._lock = threading.Lock()
        self._latest_cond = threading.Condition(self._lock)
        # Weak: an abandoned stream (e.g. a discarded `phone.imu` access) is
        # unsubscribed by the garbage collector instead of leaking forever.
        self._subs: "weakref.WeakSet[RecordStream[Any]]" = weakref.WeakSet()
        self._callbacks: list[tuple[tuple[type, ...], Callable[[Any], None]]] = []
        self._latest: dict[type, Record] = {}
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: list[socket.socket] = []
        self._video_lock = threading.Lock()
        self._video_stream = None

    # ------------------------------------------------------------------ setup

    def open(self) -> "Session":
        """Connect and start the background readers. :func:`connect` calls this."""
        try:
            sock = self._dial(self.imu_port)
            info = Handshake(recv_handshake(sock))
            _check_handshake(info, "irtsp-imu")
            self.info = info
            self._spawn(self._record_loop, sock, "irtsp-odometry")

            if self.depth_enabled:
                dsock = self._dial(self.depth_port)
                depth_info = Handshake(recv_handshake(dsock))
                _check_handshake(depth_info, "irtsp-depth")
                self.depth_info = depth_info
                self._spawn(self._depth_loop, dsock, "irtsp-depth")
        except BaseException:
            self.close()  # tear down whatever already started
            raise

        log.info("connected to %s (odometry:%d%s)", self.host, self.imu_port,
                 f", depth:{self.depth_port}" if self.depth_enabled else "")
        return self

    def _dial(self, port: int) -> socket.socket:
        sock = socket.create_connection((self.host, port), timeout=self.timeout)
        sock.settimeout(None)  # readers block; close() unblocks them
        with self._lock:
            if self._closed.is_set():  # closed while we were connecting
                try:
                    sock.close()
                finally:
                    pass
                raise ConnectionClosed("session closed")
            self._sockets.append(sock)
        return sock

    def _forget(self, sock: socket.socket) -> None:
        """Close a socket and drop it from the registry (no unbounded growth)."""
        with self._lock:
            if sock in self._sockets:
                self._sockets.remove(sock)
        try:
            sock.close()
        except OSError:
            pass

    def _spawn(self, target: Callable[..., None], sock: socket.socket, name: str) -> None:
        t = threading.Thread(target=target, args=(sock,), name=name, daemon=True)
        self._threads.append(t)
        t.start()

    # ------------------------------------------------------------ reader loops

    def _record_loop(self, sock: socket.socket) -> None:
        last_seq: int | None = None
        try:
            while not self._closed.is_set():
                try:
                    record = decode_record(read_exact(sock, RECORD_SIZE))
                except (OSError, ConnectionClosed, ProtocolError) as e:
                    if isinstance(e, ProtocolError):
                        log.error("odometry stream desynced: %s", e)
                    sock2 = self._maybe_reconnect(sock, self.imu_port, is_depth=False)
                    if sock2 is None:
                        return
                    sock, last_seq = sock2, None
                    continue
                # Late joiners: the server replays the latest Intrinsics record
                # with its ORIGINAL (stale) seq — don't let it baseline the gap
                # tracker or the first live record shows a huge bogus gap.
                if last_seq is None and isinstance(record, Intrinsics):
                    self._dispatch(record)
                    continue
                last_seq = self._track_gap(record, last_seq)
        finally:
            # Belt and suspenders: no reader exit may leave consumers hanging.
            if not self._closed.is_set():
                self.close()

    def _depth_loop(self, sock: socket.socket) -> None:
        last_seq: int | None = None
        try:
            while not self._closed.is_set():
                try:
                    frame = decode_depth_frame(recv_length_prefixed(sock))
                except (OSError, ConnectionClosed, ProtocolError) as e:
                    if isinstance(e, ProtocolError):
                        log.error("depth stream desynced: %s", e)
                    sock2 = self._maybe_reconnect(sock, self.depth_port, is_depth=True)
                    if sock2 is None:
                        return
                    sock, last_seq = sock2, None
                    continue
                last_seq = self._track_gap(frame, last_seq)
        finally:
            if not self._closed.is_set():
                self.close()

    def _track_gap(self, record: Record, last_seq: int | None) -> int:
        """Detect dropped records via the per-channel wire sequence number."""
        if last_seq is not None:
            gap = (record.seq - last_seq - 1) & 0xFFFF
            if gap:
                record = replace(record, gap=gap)
        self._dispatch(record)
        return record.seq

    def _maybe_reconnect(
        self, dead: socket.socket, port: int, *, is_depth: bool
    ) -> socket.socket | None:
        """Handle a dropped connection: reconnect (if asked) or shut the session down."""
        self._forget(dead)
        if self._closed.is_set():
            return None
        if not self.reconnect:
            log.warning("connection to %s:%d lost; closing session", self.host, port)
            self.close()
            return None

        delay = 0.5
        while not self._closed.is_set():
            sock: socket.socket | None = None
            try:
                sock = self._dial(port)
                handshake = Handshake(recv_handshake(sock))
                _check_handshake(handshake, "irtsp-depth" if is_depth else "irtsp-imu")
                if is_depth:
                    self.depth_info = handshake
                else:
                    self.info = handshake  # fresh anchors if the app restarted its stream
                log.info("reconnected to %s:%d", self.host, port)
                return sock
            except (OSError, ProtocolError) as e:
                if sock is not None:
                    self._forget(sock)
                log.debug("reconnect to %s:%d failed (%s); retrying in %.1fs",
                          self.host, port, e, delay)
                if self._closed.wait(delay):
                    return None
                delay = min(delay * 2, 5.0)
        return None

    def _dispatch(self, record: Record) -> None:
        with self._lock:
            self._latest[type(record)] = record
            self._latest_cond.notify_all()
            subs = [s for s in self._subs if isinstance(record, s.types)]
            callbacks = [fn for types, fn in self._callbacks if isinstance(record, types)]
        for sub in subs:
            sub._put(record)
        for fn in callbacks:
            try:
                fn(record)
            except Exception:  # a bad callback must not kill the reader
                log.exception("irtsp callback %r raised", fn)

    # -------------------------------------------------------------- consuming

    def stream(self, *types: type[R], buffer: int | None = None) -> RecordStream[R]:
        """A new independent iterator over the given record types (all, if none given)."""
        filt: tuple[type, ...] = types or (Record,)
        sub: RecordStream[R] = RecordStream(self, filt, buffer or self._buffer)
        with self._lock:
            self._subs.add(sub)
        return sub

    def _unsubscribe(self, sub: RecordStream[Any]) -> None:
        with self._lock:
            self._subs.discard(sub)

    # Each property access creates a fresh, independent subscription.
    @property
    def odometry(self) -> RecordStream[Record]:
        """Every record on the odometry channel (and depth frames, if enabled)."""
        return self.stream()

    @property
    def imu(self) -> RecordStream[IMU]:
        return self.stream(IMU)

    @property
    def raw_gyro(self) -> RecordStream[RawGyro]:
        return self.stream(RawGyro)

    @property
    def raw_accel(self) -> RecordStream[RawAccel]:
        return self.stream(RawAccel)

    @property
    def intrinsics(self) -> RecordStream[Intrinsics]:
        return self.stream(Intrinsics)

    @property
    def gnss(self) -> RecordStream[GNSS]:
        return self.stream(GNSS)

    @property
    def altitude(self) -> RecordStream[Altitude]:
        return self.stream(Altitude)

    @property
    def heading(self) -> RecordStream[Heading]:
        return self.stream(Heading)

    @property
    def pose(self) -> RecordStream[Pose]:
        return self.stream(Pose)

    @property
    def depth(self) -> RecordStream[DepthFrame]:
        if not self.depth_enabled:
            raise RuntimeError("depth channel not enabled — connect(host, depth=True)")
        return self.stream(DepthFrame)

    def on(self, types: type[R] | tuple[type[R], ...], callback: Callable[[R], None]) -> None:
        """Call ``callback(record)`` for matching records (on the reader thread)."""
        filt = types if isinstance(types, tuple) else (types,)
        with self._lock:
            self._callbacks.append((filt, callback))

    def latest(self, rtype: type[R], *, wait: float | None = None) -> R | None:
        """The most recent record of **exactly** ``rtype`` seen, or ``None``.

        Lookup is by concrete class (``latest(irtsp.IMU)``), not by base class —
        ``latest(irtsp.Record)`` is always ``None``. With ``wait``, blocks up to
        that many seconds for one to arrive — handy for
        :class:`~irtsp.Intrinsics`, which the server replays on connect::

            k = phone.latest(irtsp.Intrinsics, wait=2.0)
        """
        deadline = None if wait is None else time.monotonic() + wait
        with self._lock:
            while True:
                record = self._latest.get(rtype)
                if record is not None or deadline is None:
                    return record  # type: ignore[return-value]
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed.is_set():
                    return None
                self._latest_cond.wait(remaining)

    # ------------------------------------------------------------------ video

    @property
    def clock(self) -> StreamClock:
        """The session's shared host↔wall clock anchor."""
        if self.info is None:
            raise RuntimeError("session is not open")
        return self.info.clock

    @property
    def video_url(self) -> str:
        """The RTSP URL for this phone's video, rebuilt against the host we dialed."""
        if self._video_url_override:
            return self._video_url_override
        from urllib.parse import urlsplit

        advertised = (self.info.video_url if self.info else None) or "rtsp://x:8554/live"
        parts = urlsplit(advertised)
        auth = f"{self._video_auth[0]}:{self._video_auth[1]}@" if self._video_auth else ""
        port = parts.port or 8554
        return f"rtsp://{auth}{self.host}:{port}{parts.path or '/live'}"

    @property
    def frames(self):
        """Live video frames (requires the ``video`` extra). EXPERIMENTAL."""
        from . import video as _video  # deferred: needs PyAV

        with self._video_lock:
            if self._closed.is_set():
                raise RuntimeError("session is closed")
            if self._video_stream is None:
                self._video_stream = _video.VideoStream(self.video_url)
            return self._video_stream

    def synced(self, **kwargs):
        """Time-aligned :class:`~irtsp.video.SyncedFrame` bundles. EXPERIMENTAL.

        See :func:`irtsp.video.synced` for options.
        """
        from . import video as _video

        return _video.synced(self, **kwargs)

    # ---------------------------------------------------------------- teardown

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def run(self) -> None:
        """Block until the session closes (Ctrl-C closes it cleanly)."""
        try:
            self._closed.wait()
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        """Stop the readers, close the sockets, and end all iterators. Idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()
        with self._lock:
            sockets, self._sockets = self._sockets, []
            subs = list(self._subs)
            self._latest_cond.notify_all()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        for sub in subs:
            with sub._cond:
                sub._cond.notify_all()
        with self._video_lock:
            video_stream, self._video_stream = self._video_stream, None
        if video_stream is not None:
            try:
                video_stream.close()
            except Exception:  # pragma: no cover
                pass
        current = threading.current_thread()
        for t in self._threads:
            if t is not current:
                t.join(timeout=2.0)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        state = "closed" if self.closed else "live"
        extras = (" +depth" if self.depth_enabled else "") + (" +video" if self.video_enabled else "")
        return f"<irtsp.Session {self.host}:{self.imu_port}{extras} [{state}]>"


def connect(
    target: "str | Device",
    *,
    imu_port: int = 8555,
    depth: bool = False,
    depth_port: int = 8556,
    video: bool = False,
    video_url: str | None = None,
    video_auth: tuple[str, str] | None = None,
    timeout: float = 5.0,
    reconnect: bool = False,
    buffer: int = DEFAULT_BUFFER,
) -> Session:
    """Connect to an iRTSP phone and return a live :class:`Session`.

    ``target`` is a hostname/IP (``"ryans-iphone.local"``, ``"192.168.1.24"``)
    or a :class:`~irtsp.discovery.Device` from :func:`irtsp.discover`.

    Args:
        depth: also open the LiDAR depth channel (``phone.depth``).
        video: enable ``phone.frames`` / ``phone.synced()`` (needs ``irtsp[video]``).
        video_url / video_auth: override or authenticate the RTSP URL.
        reconnect: transparently redial if the phone restarts its stream.
        buffer: per-consumer record buffer before oldest are dropped.
    """
    host = target
    if not isinstance(target, str):  # a discovery.Device
        host = target.host
        ports = getattr(target, "ports", {})
        imu_port = ports.get("imu", imu_port)
        depth_port = ports.get("depth", depth_port)

    return Session(
        host,  # type: ignore[arg-type]
        imu_port=imu_port,
        depth=depth,
        depth_port=depth_port,
        video=video,
        video_url=video_url,
        video_auth=video_auth,
        timeout=timeout,
        reconnect=reconnect,
        buffer=buffer,
    ).open()
