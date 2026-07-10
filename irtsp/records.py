"""Typed, unit-aware records for every iRTSP stream.

Every sample iRTSP sends becomes one small, frozen, pattern-matchable object here.
Units are SI by default (acceleration in m/s², pressure in Pa, meters, seconds);
where the wire uses something else the conversion happens at decode time, and the
wire's native unit is always available one property away (e.g. ``imu.accel_g``).

Angles that are conventionally spoken in degrees (latitude/longitude, compass
headings, GNSS course) keep degrees, with explicit ``*_deg`` names and ``*_rad``
properties so nothing is ever ambiguous.

Two timestamps ride on every record (see the integration guide, §3):

* ``host_ts`` — seconds on the phone's monotonic host clock. Same axis as the
  video's presentation timestamps, CoreMotion, ARKit, and the depth frames.
* ``unix_ts`` — wall-clock seconds. Same axis as the video's RTCP Sender-Report
  NTP timeline, so odometry and RTP video align with **no offset to estimate**.

Wire format reference:
https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md
"""

from __future__ import annotations

import math
import numbers
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, NamedTuple

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = [
    "STANDARD_GRAVITY",
    "Vec3",
    "Quat",
    "Tracking",
    "Record",
    "IMU",
    "RawGyro",
    "RawAccel",
    "Intrinsics",
    "GNSS",
    "Altitude",
    "Heading",
    "Pose",
    "DepthFrame",
    "Unknown",
]

#: Standard gravity (m/s² per g), used to convert the wire's g-units to SI.
STANDARD_GRAVITY = 9.80665


class Vec3(NamedTuple):
    """A 3-vector that behaves like maths, not like a tuple.

    Unpacks (``x, y, z = v``), indexes, iterates, and converts to numpy with
    ``np.asarray(v)`` — but ``+``, ``-``, ``*``, ``/`` are element-wise/scalar
    vector operations rather than tuple concatenation.
    """

    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":  # type: ignore[override]
        return Vec3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __radd__(self, other) -> "Vec3":  # supports sum(vectors) and tuple + Vec3
        if other == 0:  # sum()'s integer start value
            return self
        return Vec3(other[0] + self.x, other[1] + self.y, other[2] + self.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __rsub__(self, other) -> "Vec3":
        return Vec3(other[0] - self.x, other[1] - self.y, other[2] - self.z)

    def __mul__(self, k: float) -> "Vec3":  # type: ignore[override]
        if not isinstance(k, numbers.Real):  # Vec3 * Vec3 is ambiguous — use .dot/.cross
            return NotImplemented
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__  # type: ignore[assignment]

    def __truediv__(self, k: float) -> "Vec3":
        if not isinstance(k, numbers.Real):
            return NotImplemented
        return Vec3(self.x / k, self.y / k, self.z / k)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    @property
    def magnitude(self) -> float:
        """Euclidean length."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other[0] + self.y * other[1] + self.z * other[2]

    def cross(self, other: "Vec3") -> "Vec3":
        ox, oy, oz = other[0], other[1], other[2]
        return Vec3(
            self.y * oz - self.z * oy,
            self.z * ox - self.x * oz,
            self.x * oy - self.y * ox,
        )


class Quat(NamedTuple):
    """A unit quaternion stored ``(x, y, z, w)`` — the same order as the wire.

    Represents an attitude/orientation that rotates vectors from the sensor's
    body frame into the world/reference frame (CoreMotion & ARKit convention).
    """

    x: float
    y: float
    z: float
    w: float

    @property
    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalized(self) -> "Quat":
        n = self.norm
        if n == 0.0:
            raise ValueError("cannot normalize a zero quaternion")
        return Quat(self.x / n, self.y / n, self.z / n, self.w / n)

    def conjugate(self) -> "Quat":
        return Quat(-self.x, -self.y, -self.z, self.w)

    def __mul__(self, other: "Quat") -> "Quat":  # type: ignore[override]
        """Hamilton product — ``a * b`` composes rotations (apply ``b``, then ``a``)."""
        ax, ay, az, aw = self
        bx, by, bz, bw = other
        return Quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    def rotate(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion (body → world for an attitude)."""
        qv = Vec3(self.x, self.y, self.z)
        t = qv.cross(v) * 2.0
        return v + t * self.w + qv.cross(t)


class Tracking(IntEnum):
    """ARKit world-tracking quality for a :class:`Pose`."""

    NONE = 0
    LIMITED = 1
    NORMAL = 2


@dataclass(frozen=True, kw_only=True, slots=True)
class Record:
    """Base for every decoded sample. See the module docstring for the two clocks."""

    host_ts: float  #: seconds on the phone's monotonic host clock (video-PTS axis)
    unix_ts: float  #: wall-clock seconds (RTP RTCP-SR NTP axis)
    seq: int  #: per-channel wire sequence number (wraps at 65536)
    gap: int = 0  #: records lost immediately before this one (0 = none dropped)

    @property
    def time(self) -> datetime:
        """``unix_ts`` as a timezone-aware UTC :class:`~datetime.datetime`."""
        return datetime.fromtimestamp(self.unix_ts, tz=timezone.utc)


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class IMU(Record):
    """Fused device-motion sample (wire type 1) — the main inertial stream, ≤ ~100 Hz."""

    __match_args__: ClassVar[tuple[str, ...]] = ("gyro", "accel", "quat")

    gyro: Vec3  #: angular velocity, rad/s (body frame: X-right, Y-up, Z-out-of-screen)
    accel: Vec3  #: specific force, m/s² — gravity **included**; face-up at rest ≈ (0, 0, −9.81)
    quat: Quat | None  #: attitude (body → world, ``xArbitraryZVertical``), or None if not streamed

    @property
    def accel_g(self) -> Vec3:
        """Acceleration in the wire's native g units."""
        return self.accel / STANDARD_GRAVITY


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class RawGyro(Record):
    """Raw (unfused) gyroscope sample (wire type 2), only in raw sensor mode."""

    __match_args__: ClassVar[tuple[str, ...]] = ("gyro",)

    gyro: Vec3  #: angular velocity, rad/s


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class RawAccel(Record):
    """Raw (unfused) accelerometer sample (wire type 3), only in raw sensor mode."""

    __match_args__: ClassVar[tuple[str, ...]] = ("accel",)

    accel: Vec3  #: specific force, m/s²

    @property
    def accel_g(self) -> Vec3:
        return self.accel / STANDARD_GRAVITY


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class Intrinsics(Record):
    """Pinhole camera intrinsics (wire type 5), in pixels at (``width`` × ``height``).

    Sent when the projection changes (zoom/lens switch) and replayed to late
    joiners, so :meth:`irtsp.Session.latest` almost always has one.
    No lens-distortion model — the video is rectilinear.
    """

    __match_args__: ClassVar[tuple[str, ...]] = ("fx", "fy", "cx", "cy")

    fx: float  #: focal length x, px
    fy: float  #: focal length y, px
    cx: float  #: principal point x, px
    cy: float  #: principal point y, px
    width: int  #: resolution the intrinsics are expressed at
    height: int

    @property
    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        """The 3×3 camera matrix **K** as nested tuples (``np.array(intr.matrix)`` works)."""
        return (
            (self.fx, 0.0, self.cx),
            (0.0, self.fy, self.cy),
            (0.0, 0.0, 1.0),
        )

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """These intrinsics re-expressed for another resolution (e.g. the depth map's)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            host_ts=self.host_ts, unix_ts=self.unix_ts, seq=self.seq, gap=self.gap,
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height,
        )


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class GNSS(Record):
    """GNSS fix (wire type 6), ~1 Hz.

    ``lat``/``lon``/``altitude`` are always plain floats. The accuracy, speed
    and course fields — which CoreLocation marks invalid with a negative value
    on the wire — arrive here as ``None`` instead (altitude's validity is
    indicated by ``v_accuracy``)."""

    __match_args__: ClassVar[tuple[str, ...]] = ("lat", "lon", "altitude")

    lat: float  #: degrees
    lon: float  #: degrees
    altitude: float  #: meters above sea level (validity indicated by ``v_accuracy``)
    h_accuracy: float | None  #: horizontal accuracy radius, m
    v_accuracy: float | None  #: vertical accuracy, m
    speed: float | None  #: ground speed, m/s
    course_deg: float | None  #: direction of travel, degrees clockwise from true north
    speed_accuracy: float | None  #: m/s

    @property
    def course_rad(self) -> float | None:
        return math.radians(self.course_deg) if self.course_deg is not None else None


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class Altitude(Record):
    """Barometric altitude sample (wire type 7), ~1 Hz."""

    __match_args__: ClassVar[tuple[str, ...]] = ("relative_altitude", "pressure")

    relative_altitude: float  #: meters relative to stream start
    pressure: float  #: atmospheric pressure, Pa (SI)

    @property
    def pressure_kpa(self) -> float:
        """Pressure in the wire's native kPa."""
        return self.pressure / 1000.0

    @property
    def pressure_hpa(self) -> float:
        """Pressure in hPa (= millibar), the meteorological convention."""
        return self.pressure / 100.0


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class Heading(Record):
    """Compass heading (wire type 8), event-driven. Invalid values are ``None``."""

    __match_args__: ClassVar[tuple[str, ...]] = ("true_deg", "magnetic_deg")

    true_deg: float | None  #: degrees clockwise from true north
    magnetic_deg: float  #: degrees clockwise from magnetic north
    accuracy_deg: float | None  #: maximum deviation, degrees

    @property
    def true_rad(self) -> float | None:
        return math.radians(self.true_deg) if self.true_deg is not None else None

    @property
    def magnetic_rad(self) -> float:
        return math.radians(self.magnetic_deg)


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class Pose(Record):
    """ARKit 6-DOF world pose (wire type 9); rate matches the AR camera (30-60 Hz).

    The world frame is gravity-aligned with its origin at session start.
    ``host_ts`` is the ``ARFrame`` timestamp — the same axis as the video PTS,
    so poses line up with video frames directly.
    """

    __match_args__: ClassVar[tuple[str, ...]] = ("position", "orientation", "tracking")

    position: Vec3  #: meters, world frame
    orientation: Quat  #: body → world
    tracking: Tracking
    #: True on the first pose after an ARKit session interruption / relocalization —
    #: the world frame may have jumped discontinuously; re-anchor any registration.
    #: (Wire flags bit0; always False from app versions before 1.1.)
    discontinuity: bool = False

    def transform(self, point: Vec3) -> Vec3:
        """Map a point from the device's body frame into the world frame."""
        return self.orientation.rotate(point) + self.position


@dataclass(frozen=True, kw_only=True, match_args=False)  # no slots: cached_property below
class DepthFrame(Record):
    """One LiDAR metric depth map (wire type 10, its own channel), ≤ 30 Hz.

    ``data`` holds ``width × height`` IEEE-754 half floats, row-major, each the
    distance from the camera in **meters**. Use :attr:`meters` for a numpy array
    or :meth:`at` for single pixels with no dependencies.
    """

    __match_args__: ClassVar[tuple[str, ...]] = ("width", "height")

    width: int
    height: int
    data: bytes  #: raw half-float samples (2 bytes/px, little-endian)

    @cached_property
    def meters(self) -> "np.ndarray":
        """The depth map as a float32 numpy array of shape ``(height, width)``."""
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DepthFrame.meters needs numpy — pip install 'irtsp[numpy]' "
                "(or use DepthFrame.at(x, y) which is dependency-free)"
            ) from e
        return (
            np.frombuffer(self.data, dtype=np.float16)
            .reshape(self.height, self.width)
            .astype(np.float32)
        )

    def at(self, x: int, y: int) -> float:
        """Depth at pixel ``(x, y)`` in meters — pure stdlib, no numpy needed."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"({x}, {y}) outside {self.width}x{self.height} depth map")
        return struct.unpack_from("<e", self.data, (y * self.width + x) * 2)[0]

    def point_cloud(self, intrinsics: Intrinsics, *, stride: int = 1) -> "np.ndarray":
        """Back-project to an ``(N, 3)`` float32 point cloud in the camera frame.

        ``intrinsics`` may be at any resolution (e.g. the video's) — it is
        rescaled to this depth map automatically. Non-finite depths are dropped.
        +X right, +Y down, +Z forward (standard pinhole camera frame).
        """
        try:
            import numpy as np  # noqa: F811 — shadows the TYPE_CHECKING import
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DepthFrame.point_cloud needs numpy — pip install 'irtsp[numpy]'"
            ) from e

        k = intrinsics.scaled(self.width, self.height)
        z = self.meters[::stride, ::stride]
        ys, xs = np.mgrid[0 : self.height : stride, 0 : self.width : stride]
        x = (xs - k.cx) * z / k.fx
        y = (ys - k.cy) * z / k.fy
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        return pts[np.isfinite(pts).all(axis=1)].astype(np.float32)


@dataclass(frozen=True, kw_only=True, slots=True, match_args=False)
class Unknown(Record):
    """A record type this library version doesn't know — kept, never dropped."""

    __match_args__: ClassVar[tuple[str, ...]] = ("type_id",)

    type_id: int
    payload: bytes  #: the 40 payload bytes (offsets 24..64 of the wire record)
