"""Grab one LiDAR depth frame from an iRTSP phone and save it.

Connects with the depth channel enabled, waits for a single DepthFrame plus
the latest camera Intrinsics, saves the depth map as a PNG (if matplotlib is
installed) or a .npy file (numpy only), and prints the shape of the
back-projected point cloud.

Requires numpy (pip install "irtsp[numpy]"); matplotlib is optional.
"""

from __future__ import annotations

import argparse
import sys

import irtsp


def save(depth, path_stem: str) -> str:
    """Save as PNG when matplotlib is importable, else fall back to .npy."""
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = path_stem + ".png"
        plt.imsave(path, np.nan_to_num(depth, nan=0.0, posinf=0.0), cmap="viridis")
    except ImportError:
        path = path_stem + ".npy"
        np.save(path, depth)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="phone hostname or IP, e.g. 192.168.1.24")
    parser.add_argument("-o", "--output", default="depth",
                        help="output path stem, extension added (default: %(default)s)")
    parser.add_argument("--stride", type=int, default=1,
                        help="point-cloud subsampling stride (default: %(default)s)")
    args = parser.parse_args()

    try:
        import numpy  # noqa: F401 -- DepthFrame.meters / point_cloud need it
    except ImportError:
        sys.exit('this example needs numpy: pip install "irtsp[numpy]"')

    try:
        with irtsp.connect(args.host, depth=True) as phone:
            print("waiting for a depth frame (is the app in LiDAR depth mode?) ...")
            frame = next(phone.depth)
            k = phone.latest(irtsp.Intrinsics, wait=2.0)  # replayed on connect
    except StopIteration:
        sys.exit("connection closed before a depth frame arrived")
    except KeyboardInterrupt:
        sys.exit("interrupted before a depth frame arrived")

    depth = frame.meters  # (height, width) float32, meters
    print(f"depth frame: {frame.width}x{frame.height}, "
          f"center pixel = {frame.at(frame.width // 2, frame.height // 2):.3f} m")
    print(f"saved {save(depth, args.output)}")

    if k is None:
        print("no Intrinsics arrived within 2 s -- skipping the point cloud")
    else:
        cloud = frame.point_cloud(k, stride=args.stride)
        print(f"point cloud: {cloud.shape} float32 (camera frame, meters)")


if __name__ == "__main__":
    main()
