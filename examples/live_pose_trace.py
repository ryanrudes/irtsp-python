"""Live ARKit pose trace: distance travelled + current position at 1 Hz.

Subscribes to Pose records with a callback (Session.on), integrates the path
length as samples arrive, and a small printer thread reports once a second
while Session.run() blocks the main thread until Ctrl-C.

Stdlib only. The phone must be streaming in AR pose mode.
"""

from __future__ import annotations

import argparse
import threading
import time

import irtsp


class Trace:
    """Thread-safe running trajectory: total path length + last pose."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples = 0
        self.distance = 0.0  # meters, sum of consecutive position deltas
        self.last: irtsp.Pose | None = None

    def add(self, pose: irtsp.Pose) -> None:  # runs on the reader thread
        with self._lock:
            if self.last is not None:
                self.distance += (pose.position - self.last.position).magnitude
            self.last = pose
            self.samples += 1

    def snapshot(self) -> tuple[int, float, "irtsp.Pose | None"]:
        with self._lock:
            return self.samples, self.distance, self.last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="phone hostname or IP, e.g. 192.168.1.24")
    args = parser.parse_args()

    trace = Trace()

    with irtsp.connect(args.host) as phone:
        phone.on(irtsp.Pose, trace.add)

        def report() -> None:
            while not phone.closed:
                time.sleep(1.0)
                n, dist, pose = trace.snapshot()
                if pose is None:
                    print("waiting for poses (is the app in AR pose mode?) ...")
                    continue
                p = pose.position
                print(f"poses={n:6d}  travelled={dist:8.2f} m  "
                      f"pos=({p.x:+7.3f}, {p.y:+7.3f}, {p.z:+7.3f}) m  "
                      f"tracking={pose.tracking.name}")

        threading.Thread(target=report, name="pose-report", daemon=True).start()
        print(f"connected to {args.host} -- Ctrl-C to stop")
        phone.run()  # blocks until Ctrl-C, which closes the session cleanly

    n, dist, _ = trace.snapshot()
    print(f"\ndone: {n} poses, {dist:.2f} m travelled")


if __name__ == "__main__":
    main()
