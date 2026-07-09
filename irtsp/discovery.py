"""Find iRTSP phones on the local network via Bonjour/mDNS.

Requires the ``discovery`` extra::

    pip install 'irtsp[discovery]'

Then::

    import irtsp

    for device in irtsp.discover():
        print(device.name, device.host, device.ports)

    phone = irtsp.discover()[0].connect()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session

__all__ = ["Device", "discover"]

#: Bonjour service type → channel name.
_SERVICE_CHANNELS = {
    "_rtsp._tcp.local.": "video",
    "_irtsp-imu._tcp.local.": "imu",
    "_irtsp-depth._tcp.local.": "depth",
}


@dataclass
class Device:
    """One discovered iRTSP phone."""

    name: str  #: the service name the phone advertises
    host: str  #: best address to connect to (IPv4 preferred)
    addresses: list[str] = field(default_factory=list)
    ports: dict[str, int] = field(default_factory=dict)  #: e.g. ``{"video": 8554, "imu": 8555}``
    properties: dict[str, str] = field(default_factory=dict)

    def connect(self, **kwargs: Any) -> "Session":
        """Open a session to this phone — same options as :func:`irtsp.connect`."""
        from .session import connect

        return connect(self, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        ports = ", ".join(f"{k}:{v}" for k, v in sorted(self.ports.items()))
        return f"<irtsp.Device {self.name!r} {self.host} ({ports})>"


def discover(timeout: float = 2.0) -> list[Device]:
    """Browse the local network for iRTSP phones for ``timeout`` seconds.

    Only devices advertising the iRTSP odometry channel (``_irtsp-imu._tcp``)
    are returned — a plain RTSP camera that isn't iRTSP won't show up.
    """
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except ImportError as e:
        raise ImportError(
            "irtsp.discover() needs zeroconf — pip install 'irtsp[discovery]' "
            "(or connect by IP: irtsp.connect('192.168.1.24'))"
        ) from e

    found: dict[str, Device] = {}

    class _Listener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name, timeout=int(timeout * 1000))
            if info is None:
                return
            service_name = name.removesuffix("." + type_)
            addresses = [addr for addr in info.parsed_scoped_addresses() if ":" not in addr] or list(
                info.parsed_scoped_addresses()
            )
            if not addresses:
                return
            device = found.setdefault(
                service_name, Device(name=service_name, host=addresses[0], addresses=addresses)
            )
            device.ports[_SERVICE_CHANNELS[type_]] = info.port or 0
            for k, v in (info.properties or {}).items():
                try:
                    device.properties[k.decode()] = v.decode() if isinstance(v, bytes) else str(v)
                except Exception:  # pragma: no cover — malformed TXT records
                    pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            self.add_service(zc, type_, name)

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

    zc = Zeroconf()
    try:
        listener = _Listener()
        browsers = [ServiceBrowser(zc, t, listener) for t in _SERVICE_CHANNELS]
        time.sleep(timeout)
        del browsers
    finally:
        zc.close()

    # A phone must expose the odometry channel to count as an iRTSP device.
    return sorted(
        (d for d in found.values() if "imu" in d.ports),
        key=lambda d: d.name,
    )
