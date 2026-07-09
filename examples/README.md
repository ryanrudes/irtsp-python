# irtsp examples

Runnable scripts, smallest first. Each takes the phone's hostname or IP as
its first argument (find it with `irtsp.discover()` or in the iRTSP app) and
exits cleanly on Ctrl-C.

| Script | What it shows | Needs |
| --- | --- | --- |
| [`print_all.py`](print_all.py) | every record type, pattern-matched and printed; wire gaps (`record.gap`) and consumer drops (`stream.dropped`) | stdlib |
| [`imu_to_csv.py`](imu_to_csv.py) | the IMU stream to CSV (SI units) with a `--seconds` duration | stdlib |
| [`live_pose_trace.py`](live_pose_trace.py) | callbacks + `Session.run()`: distance travelled and current position at 1 Hz | stdlib |
| [`depth_snapshot.py`](depth_snapshot.py) | one LiDAR depth frame to PNG/`.npy`, plus the point-cloud shape | numpy (matplotlib optional) |

```bash
python print_all.py 192.168.1.24 --depth
python imu_to_csv.py ryans-iphone.local --seconds 30 -o imu.csv
python live_pose_trace.py 192.168.1.24
python depth_snapshot.py 192.168.1.24 -o depth
```
