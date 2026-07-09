"""Record the fused IMU stream from an iRTSP phone to a CSV file.

Columns: host_ts, unix_ts, gyro_x/y/z (rad/s), accel_x/y/z (m/s^2, gravity
included), quat_x/y/z/w (empty when attitude streaming is off).

Stdlib only. Stops after --seconds, or earlier on Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import irtsp

HEADER = [
    "host_ts", "unix_ts",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="phone hostname or IP, e.g. 192.168.1.24")
    parser.add_argument("-o", "--output", default="imu.csv",
                        help="CSV path (default: %(default)s)")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="how long to record (default: %(default)ss)")
    args = parser.parse_args()

    rows = 0
    with irtsp.connect(args.host) as phone, \
            open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        deadline = time.monotonic() + args.seconds
        print(f"recording IMU from {args.host} for {args.seconds:g}s -> {args.output}")
        try:
            for imu in phone.imu:
                q = imu.quat
                writer.writerow([
                    f"{imu.host_ts:.6f}", f"{imu.unix_ts:.6f}",
                    imu.gyro.x, imu.gyro.y, imu.gyro.z,
                    imu.accel.x, imu.accel.y, imu.accel.z,
                    *((q.x, q.y, q.z, q.w) if q is not None else ("", "", "", "")),
                ])
                rows += 1
                if time.monotonic() >= deadline:
                    break
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
    print(f"wrote {rows} rows to {args.output}")


if __name__ == "__main__":
    main()
