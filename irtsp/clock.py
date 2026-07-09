"""The shared clock, and time interpolation helpers.

iRTSP captures **one** anchor pair per streaming session — a host-clock reading
and the Unix wall time at the same instant — and every stream (video RTP/RTCP,
IMU, GPS, pose, depth) derives its timestamps from it. That is why the streams
need no cross-correlation to align: they were never on different clocks.

:class:`StreamClock` is that anchor, parsed straight out of the handshake, and
converts between the two axes. See the integration guide §3–4 for the full
story of how the video's RTCP Sender Reports land on the same ``unix_ts`` axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .records import Pose, Quat, Tracking, Vec3

__all__ = ["StreamClock", "slerp", "interpolate_pose"]


@dataclass(frozen=True, slots=True)
class StreamClock:
    """The session's host↔wall anchor: ``unix = wall_anchor + (host − host_anchor)``."""

    host_anchor: float  #: host-clock seconds (mach_absolute_time) at anchor instant
    wall_anchor: float  #: Unix wall seconds at the same instant

    @classmethod
    def from_handshake(cls, handshake: Mapping[str, Any]) -> "StreamClock":
        clock = handshake.get("clock", {})
        return cls(
            host_anchor=float(clock.get("host_anchor", 0.0)),
            wall_anchor=float(clock.get("wall_anchor", 0.0)),
        )

    def to_unix(self, host_ts: float) -> float:
        """Map a host-clock timestamp onto the wall clock (the RTCP-SR NTP axis)."""
        return self.wall_anchor + (host_ts - self.host_anchor)

    def to_host(self, unix_ts: float) -> float:
        """Map a wall-clock timestamp back onto the host clock (the video-PTS axis)."""
        return self.host_anchor + (unix_ts - self.wall_anchor)


def slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two unit quaternions, ``t ∈ [0, 1]``.

    Takes the shortest path, and degrades gracefully to normalized lerp when the
    quaternions are nearly parallel.
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:  # shortest path
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot

    if dot > 0.9995:  # nearly parallel — nlerp avoids division blow-up
        q = Quat(
            ax + t * (bx - ax),
            ay + t * (by - ay),
            az + t * (bz - az),
            aw + t * (bw - aw),
        )
        return q.normalized()

    theta0 = math.acos(min(dot, 1.0))
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return Quat(ax * s0 + bx * s1, ay * s0 + by * s1, az * s0 + bz * s1, aw * s0 + bw * s1)


def interpolate_pose(a: Pose, b: Pose, unix_ts: float) -> Pose:
    """The pose at ``unix_ts``, interpolated between two bracketing samples.

    Position is interpolated linearly and orientation spherically; ``tracking``
    reports the *worse* of the two endpoints so degraded tracking is never
    hidden by interpolation. ``unix_ts`` outside ``[a, b]`` is clamped.
    """
    if b.unix_ts <= a.unix_ts:
        return a
    t = max(0.0, min(1.0, (unix_ts - a.unix_ts) / (b.unix_ts - a.unix_ts)))
    position = Vec3(
        a.position.x + t * (b.position.x - a.position.x),
        a.position.y + t * (b.position.y - a.position.y),
        a.position.z + t * (b.position.z - a.position.z),
    )
    return Pose(
        host_ts=a.host_ts + t * (b.host_ts - a.host_ts),
        unix_ts=a.unix_ts + t * (b.unix_ts - a.unix_ts),
        seq=b.seq,
        position=position,
        orientation=slerp(a.orientation, b.orientation, t),
        tracking=Tracking(min(a.tracking, b.tracking)),
    )
