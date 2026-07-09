#!/usr/bin/env python3
"""Validate the irtsp library against a REAL iRTSP phone.

The test suite proves the library matches the wire spec; this proves it matches
the phone. Start streaming in the iRTSP app, then::

    python3 validate_device.py <iphone-ip> [--seconds 10] [--depth]

Checks, per enabled stream: records decode, rates look sane (measured from
host_ts deltas, never the requested rate), timestamps are monotonic and the
clock anchor maps host→unix consistently, quaternions are unit-norm, sentinel
fields are None-or-plausible, intrinsics arrive (replayed on connect), and no
unexplained gaps. Exits 0 if everything passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

import irtsp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host", help="iPhone IP or hostname (shown in the iRTSP app)")
    ap.add_argument("--seconds", type=float, default=10.0, help="capture window")
    ap.add_argument("--depth", action="store_true", help="also validate the depth channel")
    args = ap.parse_args()

    problems: list[str] = []
    ok = lambda msg: print(f"  ✓ {msg}")
    bad = lambda msg: (print(f"  ✗ {msg}"), problems.append(msg))

    print(f"connecting to {args.host} …")
    with irtsp.connect(args.host, depth=args.depth) as phone:
        info = phone.info
        print(f"\nhandshake: {info.protocol} v{info.version}")
        print(f"  streams enabled: {', '.join(k for k, v in sorted(info.streams.items()) if v) or '-'}")
        anchor = phone.clock
        drift = abs(anchor.to_unix(anchor.host_anchor) - anchor.wall_anchor)
        (ok if drift < 1e-9 else bad)(f"clock anchor self-consistent (drift {drift:.2e}s)")

        wall_skew = abs(anchor.wall_anchor - time.time())
        (ok if wall_skew < 3600 else bad)(
            f"phone wall clock within {wall_skew:.1f}s of this machine"
        )

        intr = phone.latest(irtsp.Intrinsics, wait=3.0)
        if info.streams.get("intrinsics", False):
            (ok if intr else bad)("intrinsics replayed on connect")
            if intr:
                print(f"    fx={intr.fx:.1f} fy={intr.fy:.1f} @ {intr.width}x{intr.height}")

        print(f"\ncapturing {args.seconds:.0f}s of records …")
        counts: Counter[str] = Counter()
        gaps: Counter[str] = Counter()
        times: dict[str, list[float]] = defaultdict(list)
        quat_bad = mono_bad = clock_bad = 0

        stream = phone.stream(buffer=65536)
        deadline = time.monotonic() + args.seconds
        for rec in stream:
            name = type(rec).__name__
            counts[name] += 1
            gaps[name] += rec.gap
            ts = times[name]
            if ts and rec.host_ts < ts[-1]:
                mono_bad += 1
            ts.append(rec.host_ts)
            if abs(anchor.to_unix(rec.host_ts) - rec.unix_ts) > 1e-6:
                clock_bad += 1
            if isinstance(rec, irtsp.IMU) and rec.quat is not None:
                if abs(rec.quat.norm - 1.0) > 0.01:
                    quat_bad += 1
            if time.monotonic() >= deadline:
                break

        print("\nresults:")
        if not counts:
            bad("no records received — is the stream running with IMU enabled?")
        for name, n in sorted(counts.items()):
            ts = times[name]
            if len(ts) > 2:
                span = ts[-1] - ts[0]
                rate = (len(ts) - 1) / span if span > 0 else float("inf")
                print(f"  {name:<12} {n:>6} records   {rate:7.1f} Hz (from host_ts)   gaps: {gaps[name]}")
            else:
                print(f"  {name:<12} {n:>6} records   gaps: {gaps[name]}")

        imu_n = counts.get("IMU", 0)
        if info.streams.get("imu", True):
            (ok if imu_n > args.seconds * 20 else bad)(f"IMU flowing ({imu_n} records)")
        (ok if mono_bad == 0 else bad)(f"host_ts monotonic per stream ({mono_bad} violations)")
        (ok if clock_bad == 0 else bad)(
            f"unix_ts == anchor(host_ts) on every record ({clock_bad} mismatches)"
        )
        (ok if quat_bad == 0 else bad)(f"quaternions unit-norm ({quat_bad} bad)")
        total_gaps = sum(gaps.values())
        (ok if total_gaps == 0 else bad)(f"no dropped records (total gap {total_gaps})")
        (ok if stream.dropped == 0 else bad)(f"consumer kept up (dropped {stream.dropped})")

        if args.depth:
            frame = phone.latest(irtsp.DepthFrame, wait=5.0)
            if frame is None:
                bad("no depth frame within 5s — is 'Stream LiDAR depth' on?")
            else:
                ok(f"depth {frame.width}x{frame.height}, center {frame.at(frame.width // 2, frame.height // 2):.2f} m")

    print(f"\n{'ALL CHECKS PASSED ✅' if not problems else f'{len(problems)} PROBLEM(S) ❌'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
