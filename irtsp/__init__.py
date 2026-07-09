"""irtsp — friendly Python client for iRTSP phones.

Read your iPhone's camera + IMU / GPS / LiDAR-depth / ARKit-pose streams as
typed, unit-aware (SI) records — all on one shared clock, so video and
odometry align with **no time offset to estimate**.

The 3-line tour::

    import irtsp

    with irtsp.connect("192.168.1.24") as phone:
        for imu in phone.imu:
            print(imu.gyro, imu.accel)     # rad/s, m/s²

Pattern-match the whole odometry channel::

    with irtsp.connect(phone_ip, depth=True) as phone:
        for rec in phone.odometry:
            match rec:
                case irtsp.IMU(gyro=g):            ...
                case irtsp.GNSS() as fix:          ...
                case irtsp.Pose(position=p):       ...
                case irtsp.DepthFrame() as d:      ...

Extras: ``irtsp[numpy]`` (depth arrays), ``irtsp[discovery]`` (Bonjour),
``irtsp[video]`` (frames + ``synced()`` bundles, experimental).

Wire format & synchronization deep-dive:
https://github.com/ryanrudes/irtsp-support/blob/main/INTEGRATION.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .clock import StreamClock, interpolate_pose, slerp
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
from .session import Handshake, RecordStream, Session, connect
from .wire import ConnectionClosed, ProtocolError, RecordType

if TYPE_CHECKING:  # pragma: no cover
    from .aio import AsyncRecordStream, AsyncSession
    from .discovery import Device

__version__ = "0.1.0"

# Names resolved lazily (PEP 562) so `import irtsp` stays dependency-free and
# fast, while `irtsp.Device` / `irtsp.AsyncSession` still work at runtime.
_LAZY = {
    "Device": ("irtsp.discovery", "Device"),
    "AsyncSession": ("irtsp.aio", "AsyncSession"),
    "AsyncRecordStream": ("irtsp.aio", "AsyncRecordStream"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'irtsp' has no attribute {name!r}")


def __dir__() -> "list[str]":
    return sorted(list(globals()) + list(_LAZY))

__all__ = [
    # connecting
    "connect",
    "connect_async",
    "discover",
    "Session",
    "AsyncSession",
    "AsyncRecordStream",
    "Device",
    "Handshake",
    "RecordStream",
    # records
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
    # value types & constants
    "Vec3",
    "Quat",
    "Tracking",
    "STANDARD_GRAVITY",
    # clock
    "StreamClock",
    "slerp",
    "interpolate_pose",
    # protocol
    "RecordType",
    "ProtocolError",
    "ConnectionClosed",
    "__version__",
]


def discover(timeout: float = 2.0) -> "list[Device]":
    """Find iRTSP phones on the local network (needs ``irtsp[discovery]``)."""
    from .discovery import discover as _discover

    return _discover(timeout=timeout)


async def connect_async(target: "str | Any", **kwargs: Any) -> "AsyncSession":
    """Async twin of :func:`connect` — see :mod:`irtsp.aio`."""
    from .aio import connect_async as _connect_async

    return await _connect_async(target, **kwargs)
