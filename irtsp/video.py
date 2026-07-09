"""EXPERIMENTAL — video frames and time-aligned VIO bundles.

Requires the ``video`` extra (PyAV + numpy)::

    pip install 'irtsp[video]'

Then the headline feature — every video frame delivered together with the
odometry that belongs to it, all on one clock::

    with irtsp.connect("192.168.1.24", video=True, depth=True) as phone:
        for f in phone.synced():
            f.image        # (H, W, 3) RGB ndarray
            f.timestamp    # unix seconds — same axis as every record below
            f.imu          # IMU records since the previous frame
            f.depth        # nearest DepthFrame or None
            f.pose         # Pose interpolated at f.timestamp, or None
            f.intrinsics   # camera matrix for this stream

How the video lands on the shared clock
---------------------------------------
RTP timestamps are relative ticks; the RTCP Sender Reports anchor them to the
sender's wall clock — and iRTSP builds those reports from the *same* clock
anchor that stamps every odometry record (integration guide §4). When FFmpeg
exposes that anchor (``start_time_realtime``), frame times here are exact.
When it doesn't, we fall back to anchoring the first frame at local receive
time: alignment is then approximate (network + decode latency, typically a few
tens of ms) and :attr:`VideoFrame.approx_clock` is ``True`` so you know.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .clock import interpolate_pose
from .records import DepthFrame, IMU, Intrinsics, Pose

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .session import Session

__all__ = ["VideoFrame", "VideoStream", "SyncedFrame", "synced"]

log = logging.getLogger("irtsp.video")

_AV_NOPTS = -9223372036854775808


def _require_av():
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "video support needs PyAV and numpy — pip install 'irtsp[video]'"
        ) from e
    return av


@dataclass(frozen=True)
class VideoFrame:
    """One decoded video frame with a wall-clock timestamp."""

    image: "np.ndarray"  #: (H, W, 3) RGB, uint8
    unix_ts: float  #: wall-clock seconds (the odometry ``unix_ts`` axis)
    pts: float  #: seconds since the video stream started
    approx_clock: bool  #: True if the wall anchor was estimated locally (see module docs)


class VideoStream:
    """Iterate decoded :class:`VideoFrame` s from the phone's RTSP video."""

    def __init__(self, url: str, *, transport: str = "tcp", timeout: float = 5.0):
        av = _require_av()
        self.url = url
        self._container = av.open(
            url,
            options={
                "rtsp_transport": transport,
                # FFmpeg >= 5 renamed the RTSP socket timeout 'stimeout' -> 'timeout' (µs).
                "timeout": str(int(timeout * 1_000_000)),
            },
            timeout=timeout,  # PyAV-level open/read guard, FFmpeg-version independent
        )
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        anchor_us = getattr(self._container, "start_time_realtime", None)
        if anchor_us is not None and anchor_us not in (0, _AV_NOPTS):
            self._anchor: float | None = anchor_us / 1_000_000.0
            self._approx = False
        else:
            self._anchor = None  # resolved at the first frame (local clock)
            self._approx = True
            log.warning(
                "FFmpeg did not expose the RTCP wall-clock anchor; video timestamps "
                "will be approximate (local-clock anchored)."
            )

    def __iter__(self) -> Iterator[VideoFrame]:
        time_base = float(self._stream.time_base)
        for frame in self._container.decode(self._stream):
            if frame.pts is None:  # pragma: no cover — rare decoder hiccup
                continue
            pts = frame.pts * time_base
            if self._anchor is None:
                self._anchor = time.time() - pts
            yield VideoFrame(
                image=frame.to_ndarray(format="rgb24"),
                unix_ts=self._anchor + pts,
                pts=pts,
                approx_clock=self._approx,
            )

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:  # pragma: no cover
            pass


@dataclass(frozen=True)
class SyncedFrame:
    """A video frame plus everything the phone knew at that moment, on one clock."""

    image: "np.ndarray"  #: (H, W, 3) RGB
    timestamp: float  #: unix seconds — directly comparable to any record's ``unix_ts``
    imu: list[IMU] = field(default_factory=list)  #: samples since the previous frame
    pose: Pose | None = None  #: interpolated at ``timestamp`` (SLERP), if pose is streaming
    depth: DepthFrame | None = None  #: nearest depth map within tolerance, if streaming
    intrinsics: Intrinsics | None = None  #: latest camera matrix
    approx_clock: bool = False  #: see :class:`VideoFrame`


def synced(
    session: "Session",
    *,
    depth_tolerance: float = 0.15,
    pose_window: float = 0.5,
    warmup: float = 0.3,
) -> Iterator[SyncedFrame]:
    """Yield time-aligned :class:`SyncedFrame` bundles from a video-enabled session.

    Args:
        depth_tolerance: max |Δt| (s) for a depth map to count as "this frame's".
        pose_window: how long (s) to retain poses for bracketing/interpolation.
        warmup: initial delay before the first frame, letting odometry buffers fill.
    """
    imu_stream = session.stream(IMU)
    pose_stream = session.stream(Pose)
    depth_stream = session.stream(DepthFrame) if session.depth_enabled else None

    time.sleep(warmup)

    imu_backlog: list[IMU] = []
    poses: list[Pose] = []
    depths: list[DepthFrame] = []
    last_t: float | None = None

    try:
        for vf in session.frames:
            t = vf.unix_ts

            # IMU since the previous frame (ready for pre-integration).
            imu_backlog.extend(imu_stream.pop_all())
            frame_imu = [r for r in imu_backlog if r.unix_ts <= t]
            imu_backlog = [r for r in imu_backlog if r.unix_ts > t]
            if last_t is not None:
                frame_imu = [r for r in frame_imu if r.unix_ts > last_t]

            # Pose interpolated at t from the bracketing samples.
            poses.extend(pose_stream.pop_all())
            poses = [p for p in poses if p.unix_ts >= t - pose_window]
            pose = _pose_at(poses, t, pose_window)

            # Nearest depth frame within tolerance.
            depth = None
            if depth_stream is not None:
                depths.extend(depth_stream.pop_all())
                depths = [d for d in depths if d.unix_ts >= t - 2 * depth_tolerance]
                if depths:
                    best = min(depths, key=lambda d: abs(d.unix_ts - t))
                    if abs(best.unix_ts - t) <= depth_tolerance:
                        depth = best

            yield SyncedFrame(
                image=vf.image,
                timestamp=t,
                imu=frame_imu,
                pose=pose,
                depth=depth,
                intrinsics=session.latest(Intrinsics),
                approx_clock=vf.approx_clock,
            )
            last_t = t
    finally:
        imu_stream.close()
        pose_stream.close()
        if depth_stream is not None:
            depth_stream.close()


def _pose_at(poses: list[Pose], t: float, window: float) -> Pose | None:
    """Interpolate the pose at time ``t`` from a time-ordered-ish buffer."""
    if not poses:
        return None
    before = [p for p in poses if p.unix_ts <= t]
    after = [p for p in poses if p.unix_ts > t]
    if before and after:
        return interpolate_pose(max(before, key=lambda p: p.unix_ts),
                                min(after, key=lambda p: p.unix_ts), t)
    nearest = max(before, key=lambda p: p.unix_ts) if before else min(after, key=lambda p: p.unix_ts)
    return nearest if abs(nearest.unix_ts - t) <= window else None
