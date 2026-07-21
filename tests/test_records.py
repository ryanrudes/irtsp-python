"""Tests for the typed record layer: Vec3/Quat math, Intrinsics, DepthFrame,
Record.time, and structural pattern matching via each record's __match_args__."""

from __future__ import annotations

import math
import struct
from datetime import datetime, timezone

import pytest

from irtsp.records import (
    GNSS,
    IMU,
    STANDARD_GRAVITY,
    Altitude,
    Camera,
    CameraFormat,
    CapturePath,
    DepthFrame,
    Heading,
    Intrinsics,
    Pose,
    PTSConvention,
    PTSProvenance,
    Quat,
    RawAccel,
    RawGyro,
    ReadoutDirection,
    ReadoutProvenance,
    Record,
    SyncModel,
    SyncState,
    Tracking,
    Unknown,
    Vec3,
)

COMMON = {"host_ts": 12.5, "unix_ts": 1700000000.5, "seq": 3}


def approx_vec(v: Vec3, expected: tuple[float, float, float], tol: float = 1e-12) -> None:
    assert v.x == pytest.approx(expected[0], abs=tol)
    assert v.y == pytest.approx(expected[1], abs=tol)
    assert v.z == pytest.approx(expected[2], abs=tol)


# --------------------------------------------------------------------------- #
# Vec3
# --------------------------------------------------------------------------- #


def test_vec3_add_is_elementwise_not_tuple_concat() -> None:
    v = Vec3(1.0, 2.0, 3.0) + Vec3(4.0, 5.0, 6.0)
    assert isinstance(v, Vec3)
    assert len(v) == 3  # tuple concat would make 6
    assert v == Vec3(5.0, 7.0, 9.0)


def test_vec3_sub_neg_div() -> None:
    assert Vec3(5.0, 7.0, 9.0) - Vec3(4.0, 5.0, 6.0) == Vec3(1.0, 2.0, 3.0)
    assert -Vec3(1.0, -2.0, 3.0) == Vec3(-1.0, 2.0, -3.0)
    assert Vec3(2.0, 4.0, 8.0) / 2.0 == Vec3(1.0, 2.0, 4.0)


def test_vec3_scalar_mul_both_sides() -> None:
    v = Vec3(1.0, -2.0, 0.5)
    assert v * 2.0 == Vec3(2.0, -4.0, 1.0)
    assert 2.0 * v == Vec3(2.0, -4.0, 1.0)
    # tuple semantics would have repeated the elements instead
    assert v * 2.0 != (1.0, -2.0, 0.5, 1.0, -2.0, 0.5)


def test_vec3_dot() -> None:
    assert Vec3(1.0, 2.0, 3.0).dot(Vec3(4.0, 5.0, 6.0)) == 32.0
    assert Vec3(1.0, 0.0, 0.0).dot(Vec3(0.0, 1.0, 0.0)) == 0.0


def test_vec3_cross_right_handed_basis() -> None:
    x, y, z = Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)
    assert x.cross(y) == z
    assert y.cross(z) == x
    assert z.cross(x) == y
    # antisymmetry
    assert y.cross(x) == -z


def test_vec3_magnitude() -> None:
    assert Vec3(3.0, 4.0, 0.0).magnitude == 5.0
    assert Vec3(0.0, 0.0, 0.0).magnitude == 0.0
    assert Vec3(1.0, 1.0, 1.0).magnitude == pytest.approx(math.sqrt(3.0))


def test_vec3_still_unpacks_and_indexes_like_a_tuple() -> None:
    v = Vec3(1.0, 2.0, 3.0)
    x, y, z = v
    assert (x, y, z) == (1.0, 2.0, 3.0)
    assert v[0] == 1.0 and v[2] == 3.0
    assert list(v) == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Quat
# --------------------------------------------------------------------------- #

S2 = math.sqrt(0.5)
QZ90 = Quat(0.0, 0.0, S2, S2)  # 90° about +Z
QX90 = Quat(S2, 0.0, 0.0, S2)  # 90° about +X
IDENTITY = Quat(0.0, 0.0, 0.0, 1.0)


def test_quat_norm_and_normalized() -> None:
    q = Quat(0.0, 0.0, 3.0, 4.0)
    assert q.norm == 5.0
    assert q.normalized() == Quat(0.0, 0.0, 0.6, 0.8)
    assert Quat(0.0, 0.0, 0.0, 1.0).normalized() == IDENTITY


def test_quat_normalize_zero_raises() -> None:
    with pytest.raises(ValueError):
        Quat(0.0, 0.0, 0.0, 0.0).normalized()


def test_quat_conjugate() -> None:
    assert Quat(1.0, -2.0, 3.0, 4.0).conjugate() == Quat(-1.0, 2.0, -3.0, 4.0)


def test_quat_hamilton_basis_identities() -> None:
    i = Quat(1.0, 0.0, 0.0, 0.0)
    j = Quat(0.0, 1.0, 0.0, 0.0)
    k = Quat(0.0, 0.0, 1.0, 0.0)
    assert i * i == Quat(0.0, 0.0, 0.0, -1.0)  # i² = -1
    assert j * j == Quat(0.0, 0.0, 0.0, -1.0)
    assert k * k == Quat(0.0, 0.0, 0.0, -1.0)
    assert i * j == k
    assert j * k == i
    assert k * i == j
    assert j * i == Quat(0.0, 0.0, -1.0, 0.0)  # anticommutes


def test_quat_identity_is_multiplicative_identity() -> None:
    q = Quat(0.5, -0.5, 0.5, 0.5)
    assert IDENTITY * q == q
    assert q * IDENTITY == q


def test_quat_times_conjugate_is_identity_for_unit_quat() -> None:
    p = QZ90 * QZ90.conjugate()
    assert p.x == pytest.approx(0.0, abs=1e-12)
    assert p.y == pytest.approx(0.0, abs=1e-12)
    assert p.z == pytest.approx(0.0, abs=1e-12)
    assert p.w == pytest.approx(1.0, abs=1e-12)


def test_quat_rotate_90deg_about_z() -> None:
    approx_vec(QZ90.rotate(Vec3(1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
    approx_vec(QZ90.rotate(Vec3(0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0))
    approx_vec(QZ90.rotate(Vec3(0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))  # axis fixed


def test_quat_rotate_90deg_about_x() -> None:
    approx_vec(QX90.rotate(Vec3(0.0, 1.0, 0.0)), (0.0, 0.0, 1.0))
    approx_vec(QX90.rotate(Vec3(0.0, 0.0, 1.0)), (0.0, -1.0, 0.0))


def test_quat_composition_applies_right_factor_first() -> None:
    # a * b applies b then a; two 90° Z turns = 180° about Z
    q180 = QZ90 * QZ90
    approx_vec(q180.rotate(Vec3(1.0, 0.0, 0.0)), (-1.0, 0.0, 0.0))
    # composed rotation == sequential rotations
    v = Vec3(0.25, -1.5, 2.0)
    composed = (QX90 * QZ90).rotate(v)
    sequential = QX90.rotate(QZ90.rotate(v))
    approx_vec(composed, tuple(sequential))


def test_quat_rotate_preserves_length() -> None:
    v = Vec3(1.0, 2.0, -3.0)
    assert QZ90.rotate(v).magnitude == pytest.approx(v.magnitude)


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #


def _intrinsics() -> Intrinsics:
    return Intrinsics(fx=1000.0, fy=1200.0, cx=640.0, cy=360.0,
                      width=1280, height=720, **COMMON)


def test_intrinsics_matrix() -> None:
    assert _intrinsics().matrix == (
        (1000.0, 0.0, 640.0),
        (0.0, 1200.0, 360.0),
        (0.0, 0.0, 1.0),
    )


def test_intrinsics_scaled_halves_everything_at_half_resolution() -> None:
    k = _intrinsics().scaled(640, 360)
    assert k.fx == 500.0
    assert k.fy == 600.0
    assert k.cx == 320.0
    assert k.cy == 180.0
    assert k.width == 640 and k.height == 360
    # provenance rides along
    assert k.host_ts == COMMON["host_ts"]
    assert k.unix_ts == COMMON["unix_ts"]
    assert k.seq == COMMON["seq"]
    assert k.gap == 0


def test_intrinsics_scaled_anisotropic() -> None:
    k = _intrinsics().scaled(320, 720)  # x by 1/4, y unchanged
    assert k.fx == 250.0 and k.cx == 160.0
    assert k.fy == 1200.0 and k.cy == 360.0


def test_intrinsics_scaled_to_own_size_is_a_fixed_point() -> None:
    k0 = _intrinsics()
    k = k0.scaled(k0.width, k0.height)
    assert (k.fx, k.fy, k.cx, k.cy, k.width, k.height) == (
        k0.fx, k0.fy, k0.cx, k0.cy, k0.width, k0.height)


# --------------------------------------------------------------------------- #
# DepthFrame
# --------------------------------------------------------------------------- #


def _depth_frame(width: int, height: int, values: list[float]) -> DepthFrame:
    data = struct.pack(f"<{width * height}e", *values)
    return DepthFrame(width=width, height=height, data=data, **COMMON)


def test_depth_at_reads_row_major_half_floats() -> None:
    frame = _depth_frame(3, 2, [1.0, 2.0, 0.5, 4.0, 0.25, 8.0])
    assert frame.at(0, 0) == 1.0
    assert frame.at(1, 0) == 2.0
    assert frame.at(2, 0) == 0.5
    assert frame.at(0, 1) == 4.0
    assert frame.at(2, 1) == 8.0


def test_depth_at_bounds_checked() -> None:
    frame = _depth_frame(3, 2, [1.0] * 6)
    for x, y in [(3, 0), (0, 2), (-1, 0), (0, -1)]:
        with pytest.raises(IndexError):
            frame.at(x, y)


def test_depth_at_agrees_with_numpy_meters() -> None:
    np = pytest.importorskip("numpy")
    values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    frame = _depth_frame(4, 3, values)
    m = frame.meters
    assert m.shape == (3, 4)
    assert m.dtype == np.float32
    for y in range(frame.height):
        for x in range(frame.width):
            assert frame.at(x, y) == m[y, x]  # exact: both decode binary16


def test_depth_point_cloud_flat_map_back_projection() -> None:
    np = pytest.importorskip("numpy")
    width, height, depth = 8, 6, 2.0  # 2.0 m is exact in binary16
    frame = _depth_frame(width, height, [depth] * (width * height))
    # intrinsics at 2x the depth resolution: must be rescaled internally
    intr = Intrinsics(fx=8.0, fy=8.0, cx=8.0, cy=6.0, width=16, height=12, **COMMON)
    pts = frame.point_cloud(intr)

    assert pts.shape == (width * height, 3)
    assert pts.dtype == np.float32
    # flat map: back-projected z is exactly the depth everywhere
    assert np.all(pts[:, 2] == depth)

    # after scaling to 8x6: fx=fy=4, cx=4, cy=3. Row-major flattening puts
    # pixel (x, y) at index y*width + x.
    def px(x: int, y: int) -> "np.ndarray":
        return pts[y * width + x]

    # x = (u - cx) * z / fx ; y = (v - cy) * z / fy
    assert px(0, 0) == pytest.approx([(0 - 4) * depth / 4, (0 - 3) * depth / 4, depth])
    assert px(7, 5) == pytest.approx([(7 - 4) * depth / 4, (5 - 3) * depth / 4, depth])
    # sign convention: +X right, +Y down
    assert px(0, 0)[0] < 0 and px(7, 0)[0] > 0     # left of center vs right
    assert px(4, 0)[1] < 0 and px(4, 5)[1] > 0     # above center vs below
    assert px(4, 3)[0] == 0.0 and px(4, 3)[1] == 0.0  # exactly at principal point


def test_depth_point_cloud_stride_and_nonfinite_drop() -> None:
    np = pytest.importorskip("numpy")
    width, height = 8, 6
    values = [2.0] * (width * height)
    values[0] = math.inf  # LiDAR no-return
    frame = _depth_frame(width, height, values)
    intr = Intrinsics(fx=4.0, fy=4.0, cx=4.0, cy=3.0, width=8, height=6, **COMMON)

    pts = frame.point_cloud(intr)
    assert pts.shape == (width * height - 1, 3)  # inf pixel dropped
    assert np.isfinite(pts).all()

    strided = frame.point_cloud(intr, stride=2)
    # rows 0,2,4 x cols 0,2,4,6 = 12 pixels, minus the inf at (0,0)
    assert strided.shape == (11, 3)
    assert np.all(strided[:, 2] == 2.0)


# --------------------------------------------------------------------------- #
# SyncModel
# --------------------------------------------------------------------------- #


def _sync(**over) -> SyncModel:
    kw = dict(offset_ns=123456789, skew_ppm=24.7, epoch_host_ns=98765432100,
              residual_ns=84000.0, state=SyncState.CONVERGED, sample_count=42)
    kw.update(over)
    return SyncModel(**kw, **COMMON)


def test_sync_model_leader_time_matches_formula() -> None:
    m = _sync(offset_ns=1_000_000, skew_ppm=25.0, epoch_host_ns=2_000_000_000)
    host_ns = 2_000_500_000
    expected = host_ns + 1_000_000 + 25.0 * 1e-6 * (host_ns - 2_000_000_000)
    assert m.leader_time(host_ns) == expected


def test_sync_model_leader_time_at_epoch_is_pure_offset() -> None:
    # At host_ns == epoch the skew term vanishes, leaving exactly the offset.
    m = _sync(offset_ns=-4200, epoch_host_ns=5_000_000_000)
    assert m.leader_time(5_000_000_000) == 5_000_000_000 - 4200


def test_sync_model_offset_only_ignores_skew() -> None:
    m = _sync(skew_ppm=0.0, offset_ns=777, epoch_host_ns=0, state=SyncState.OFFSET_ONLY)
    assert m.leader_time(999_999) == 999_999 + 777


def test_sync_model_is_frozen() -> None:
    m = _sync()
    with pytest.raises(AttributeError):
        m.offset_ns = 0  # type: ignore[misc]


def test_sync_state_wire_values() -> None:
    assert (SyncState.NOT_CONVERGED, SyncState.OFFSET_ONLY, SyncState.CONVERGED) == (0, 1, 2)


def test_match_sync_model() -> None:
    match _sync():
        case SyncModel(offset_ns, skew_ppm, state):
            assert offset_ns == 123456789
            assert skew_ppm == 24.7
            assert state is SyncState.CONVERGED
        case _:
            pytest.fail("SyncModel did not match")


# --------------------------------------------------------------------------- #
# Record.time
# --------------------------------------------------------------------------- #


def test_record_time_is_aware_utc_datetime() -> None:
    rec = RawGyro(gyro=Vec3(0.0, 0.0, 0.0), **COMMON)
    assert rec.time == datetime(2023, 11, 14, 22, 13, 20, 500000, tzinfo=timezone.utc)
    assert rec.time.tzinfo == timezone.utc
    assert rec.time.timestamp() == COMMON["unix_ts"]


def test_record_gap_defaults_to_zero_and_is_settable() -> None:
    assert RawGyro(gyro=Vec3(0, 0, 0), **COMMON).gap == 0
    assert RawGyro(gyro=Vec3(0, 0, 0), gap=4, **COMMON).gap == 4


def test_records_are_frozen() -> None:
    rec = RawGyro(gyro=Vec3(0, 0, 0), **COMMON)
    with pytest.raises(AttributeError):
        rec.seq = 99  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Structural pattern matching (__match_args__)
# --------------------------------------------------------------------------- #


def test_match_imu() -> None:
    rec: Record = IMU(gyro=Vec3(1, 2, 3), accel=Vec3(0, 0, -STANDARD_GRAVITY),
                      quat=Quat(0, 0, 0, 1), **COMMON)
    match rec:
        case IMU(gyro, accel, quat):
            assert gyro == Vec3(1, 2, 3)
            assert accel == Vec3(0, 0, -STANDARD_GRAVITY)
            assert quat == Quat(0, 0, 0, 1)
        case _:
            pytest.fail("IMU did not match IMU(gyro, accel, quat)")


def test_match_raw_gyro_and_accel() -> None:
    match RawGyro(gyro=Vec3(4, 5, 6), **COMMON):
        case RawGyro(g):
            assert g == Vec3(4, 5, 6)
        case _:
            pytest.fail("RawGyro did not match")
    match RawAccel(accel=Vec3(7, 8, 9), **COMMON):
        case RawAccel(a):
            assert a == Vec3(7, 8, 9)
        case _:
            pytest.fail("RawAccel did not match")


def test_match_intrinsics() -> None:
    match _intrinsics():
        case Intrinsics(fx, fy, cx, cy):
            assert (fx, fy, cx, cy) == (1000.0, 1200.0, 640.0, 360.0)
        case _:
            pytest.fail("Intrinsics did not match")


def test_match_gnss() -> None:
    rec = GNSS(lat=37.5, lon=-122.25, altitude=10.0, h_accuracy=3.0,
               v_accuracy=None, speed=None, course_deg=None,
               speed_accuracy=None, **COMMON)
    match rec:
        case GNSS(lat, lon, altitude):
            assert (lat, lon, altitude) == (37.5, -122.25, 10.0)
        case _:
            pytest.fail("GNSS did not match")


def test_match_altitude() -> None:
    match Altitude(relative_altitude=1.5, pressure=101250.0, **COMMON):
        case Altitude(rel, pressure):
            assert rel == 1.5 and pressure == 101250.0
        case _:
            pytest.fail("Altitude did not match")


def test_match_heading() -> None:
    match Heading(true_deg=350.0, magnetic_deg=337.5, accuracy_deg=None, **COMMON):
        case Heading(true_deg, magnetic_deg):
            assert true_deg == 350.0 and magnetic_deg == 337.5
        case _:
            pytest.fail("Heading did not match")


def test_match_pose() -> None:
    rec = Pose(position=Vec3(1, 2, 3), orientation=Quat(0, 0, 0, 1),
               tracking=Tracking.NORMAL, **COMMON)
    match rec:
        case Pose(position, orientation, tracking):
            assert position == Vec3(1, 2, 3)
            assert orientation == Quat(0, 0, 0, 1)
            assert tracking is Tracking.NORMAL
        case _:
            pytest.fail("Pose did not match")


def test_match_depth_frame() -> None:
    match _depth_frame(3, 2, [1.0] * 6):
        case DepthFrame(width, height):
            assert (width, height) == (3, 2)
        case _:
            pytest.fail("DepthFrame did not match")


def _camera_format(**over) -> CameraFormat:
    kw = dict(format_id=0x1234ABCD, width=1920, height=1440, fps=30.0,
              readout_time_s=0.015625, camera=Camera.BACK_WIDE,
              capture_path=CapturePath.AVCAPTURE,
              readout_direction=ReadoutDirection.POS_Y,
              pts_convention=PTSConvention.FIRST_ROW_START,
              pts_provenance=PTSProvenance.DOCUMENTED,
              readout_provenance=ReadoutProvenance.PROBED,
              binned=False, cropped=False)
    kw.update(over)
    return CameraFormat(**kw, **COMMON)


def test_camera_format_enum_wire_values() -> None:
    assert (Camera.UNKNOWN, Camera.BACK_WIDE, Camera.BACK_ULTRAWIDE,
            Camera.BACK_TELE, Camera.FRONT, Camera.BACK_LIDAR) == (0, 1, 2, 3, 4, 5)
    assert (CapturePath.AVCAPTURE, CapturePath.ARKIT) == (0, 1)
    assert (ReadoutDirection.UNKNOWN, ReadoutDirection.POS_Y, ReadoutDirection.NEG_Y,
            ReadoutDirection.POS_X, ReadoutDirection.NEG_X) == (0, 1, 2, 3, 4)
    assert (PTSConvention.UNKNOWN, PTSConvention.FIRST_ROW_START, PTSConvention.FRAME_CENTER,
            PTSConvention.LAST_ROW_END, PTSConvention.EXPOSURE_START) == (0, 1, 2, 3, 4)
    assert (PTSProvenance.UNKNOWN, PTSProvenance.DOCUMENTED, PTSProvenance.MEASURED) == (0, 1, 2)
    assert (ReadoutProvenance.ABSENT, ReadoutProvenance.PROBED) == (0, 1)


def test_camera_format_snapshot_defaults_false_and_is_frozen() -> None:
    fmt = _camera_format()
    assert fmt.snapshot is False
    with pytest.raises(AttributeError):
        fmt.format_id = 0  # type: ignore[misc]


def test_match_camera_format() -> None:
    match _camera_format(readout_time_s=None):
        case CameraFormat(format_id, width, height, fps):
            assert format_id == 0x1234ABCD
            assert (width, height) == (1920, 1440)
            assert fps == 30.0
        case _:
            pytest.fail("CameraFormat did not match")


def test_match_unknown() -> None:
    match Unknown(type_id=42, payload=b"\x00" * 40, **COMMON):
        case Unknown(type_id):
            assert type_id == 42
        case _:
            pytest.fail("Unknown did not match")


def test_match_dispatch_over_mixed_stream() -> None:
    """The advertised usage: one match statement fanning out a record stream."""
    records: list[Record] = [
        IMU(gyro=Vec3(0, 0, 0), accel=Vec3(0, 0, -STANDARD_GRAVITY),
            quat=None, **COMMON),
        Pose(position=Vec3(1, 0, 0), orientation=Quat(0, 0, 0, 1),
             tracking=Tracking.LIMITED, **COMMON),
        Unknown(type_id=42, payload=b"\x00" * 40, **COMMON),
    ]
    seen: list[str] = []
    for rec in records:
        match rec:
            case IMU(_, accel, None):
                seen.append(f"imu:{accel.z:.2f}")
            case Pose(_, _, Tracking.LIMITED):
                seen.append("pose:limited")
            case Unknown(type_id):
                seen.append(f"unknown:{type_id}")
            case _:
                pytest.fail(f"unexpected fallthrough for {rec!r}")
    assert seen == ["imu:-9.81", "pose:limited", "unknown:42"]


def test_pose_transform_body_to_world() -> None:
    pose = Pose(position=Vec3(10.0, 0.0, 0.0), orientation=QZ90,
                tracking=Tracking.NORMAL, **COMMON)
    approx_vec(pose.transform(Vec3(1.0, 0.0, 0.0)), (10.0, 1.0, 0.0))
