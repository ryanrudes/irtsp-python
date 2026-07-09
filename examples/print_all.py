"""Print every record an iRTSP phone sends, one line per sample.

Connects to the phone's odometry channel (plus the LiDAR depth channel with
--depth), pattern-matches each record type, and prints a formatted line for
it. Also demonstrates the two loss counters:

* ``record.gap``      -- records the *wire* lost right before this one
* ``stream.dropped``  -- records *this consumer* dropped for falling behind

Stdlib only. Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse

import irtsp


def fmt3(v: irtsp.Vec3) -> str:
    return f"({v.x:+8.3f}, {v.y:+8.3f}, {v.z:+8.3f})"


def describe(rec: irtsp.Record) -> str:
    match rec:
        case irtsp.IMU(gyro=g, accel=a, quat=q):
            quat = f"  quat=({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})" if q else ""
            return f"IMU        gyro={fmt3(g)} rad/s  accel={fmt3(a)} m/s^2{quat}"
        case irtsp.RawGyro(gyro=g):
            return f"RawGyro    {fmt3(g)} rad/s"
        case irtsp.RawAccel(accel=a):
            return f"RawAccel   {fmt3(a)} m/s^2"
        case irtsp.Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy):
            return (f"Intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
                    f" @ {rec.width}x{rec.height}px")
        case irtsp.GNSS(lat=lat, lon=lon, altitude=alt):
            acc = f" +/-{rec.h_accuracy:.0f}m" if rec.h_accuracy is not None else ""
            spd = f"  speed={rec.speed:.1f}m/s" if rec.speed is not None else ""
            return f"GNSS       {lat:+.6f}, {lon:+.6f}  alt={alt:.1f}m{acc}{spd}"
        case irtsp.Altitude(relative_altitude=rel, pressure=_):
            return f"Altitude   rel={rel:+.2f}m  pressure={rec.pressure_kpa:.3f} kPa"
        case irtsp.Heading(true_deg=t, magnetic_deg=m):
            true = f"{t:.1f}" if t is not None else "n/a"
            return f"Heading    true={true} deg  magnetic={m:.1f} deg"
        case irtsp.Pose(position=p, tracking=t):
            return f"Pose       pos={fmt3(p)} m  tracking={t.name}"
        case irtsp.DepthFrame(width=w, height=h):
            return f"DepthFrame {w}x{h}px  center={rec.at(w // 2, h // 2):.2f} m"
        case irtsp.Unknown(type_id=t):
            return f"Unknown    type_id={t} ({len(rec.payload)} payload bytes)"
        case _:
            return repr(rec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="phone hostname or IP, e.g. 192.168.1.24")
    parser.add_argument("--depth", action="store_true",
                        help="also open the LiDAR depth channel")
    args = parser.parse_args()

    with irtsp.connect(args.host, depth=args.depth) as phone:
        stream = phone.odometry  # one subscription over every record type
        print(f"connected: {phone.info}")
        try:
            for n, rec in enumerate(stream, start=1):
                if rec.gap:  # the wire lost records right before this one
                    print(f"!! wire gap: {rec.gap} record(s) lost before seq={rec.seq}")
                print(f"[{rec.host_ts:12.4f}] {describe(rec)}")
                if n % 500 == 0 and stream.dropped:
                    print(f"!! consumer fell behind: {stream.dropped} record(s) dropped")
        except KeyboardInterrupt:
            pass
        print(f"\nbye -- consumer dropped {stream.dropped} record(s) total")


if __name__ == "__main__":
    main()
