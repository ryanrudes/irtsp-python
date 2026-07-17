"""Tests for irtsp.clock: StreamClock, slerp, and interpolate_pose.

Ground truth: Sources/Motion/StreamClock.swift (one host/wall anchor pair per
session; unix = wall_anchor + (host - host_anchor)) and INTEGRATION.md §3-4.
"""

from __future__ import annotations

import logging
import math

import pytest

from irtsp import (
    KNOWN_TIMEBASES,
    Pose,
    Quat,
    StreamClock,
    Tracking,
    Vec3,
    interpolate_pose,
    slerp,
)

HOST_ANCHOR = 12_345.678
WALL_ANCHOR = 1_752_000_000.25


def quat_axis_angle(axis: tuple[float, float, float], angle: float) -> Quat:
    """Unit quaternion for a rotation of ``angle`` radians about ``axis``."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    s = math.sin(angle / 2.0) / n
    return Quat(x * s, y * s, z * s, math.cos(angle / 2.0))


def quat_about_z(angle: float) -> Quat:
    return quat_axis_angle((0.0, 0.0, 1.0), angle)


def assert_quat_close(got: Quat, want: Quat, *, abs_tol: float = 1e-9) -> None:
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=abs_tol)


IDENTITY = Quat(0.0, 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# StreamClock
# --------------------------------------------------------------------------- #


class TestStreamClock:
    def test_anchor_maps_exactly(self):
        clock = StreamClock(host_anchor=HOST_ANCHOR, wall_anchor=WALL_ANCHOR)
        assert clock.to_unix(HOST_ANCHOR) == WALL_ANCHOR
        assert clock.to_host(WALL_ANCHOR) == HOST_ANCHOR

    @pytest.mark.parametrize(
        "host_ts", [0.0, 1.0, 12_345.678, 99_999.125, -3.25, 8.64e4]
    )
    def test_to_unix_to_host_round_trip(self, host_ts):
        clock = StreamClock(host_anchor=HOST_ANCHOR, wall_anchor=WALL_ANCHOR)
        assert clock.to_host(clock.to_unix(host_ts)) == pytest.approx(host_ts, abs=1e-6)

    @pytest.mark.parametrize(
        "unix_ts", [WALL_ANCHOR, WALL_ANCHOR + 0.001, WALL_ANCHOR - 42.5, 1.7e9]
    )
    def test_to_host_to_unix_round_trip(self, unix_ts):
        clock = StreamClock(host_anchor=HOST_ANCHOR, wall_anchor=WALL_ANCHOR)
        assert clock.to_unix(clock.to_host(unix_ts)) == pytest.approx(unix_ts, abs=1e-6)

    def test_offsets_are_preserved(self):
        """The mapping is a pure shift: intervals survive the conversion."""
        clock = StreamClock(host_anchor=HOST_ANCHOR, wall_anchor=WALL_ANCHOR)
        for dt in (0.001, 1 / 60, 1.0, 3600.0):
            assert clock.to_unix(100.0 + dt) - clock.to_unix(100.0) == pytest.approx(
                dt, abs=1e-6
            )
            assert clock.to_host(WALL_ANCHOR + dt) - clock.to_host(
                WALL_ANCHOR
            ) == pytest.approx(dt, abs=1e-6)

    def test_from_handshake_real_shape(self):
        """The exact 'clock' object IMUStreamServer.swift puts in the handshake."""
        handshake = {
            "protocol": "irtsp-imu",
            "version": 1,
            "clock": {
                "timebase": "mach_absolute_time_seconds",
                "host_anchor": HOST_ANCHOR,
                "wall_anchor": WALL_ANCHOR,
                "rtcp_sync": "unix_ts matches RTP RTCP SR NTP timeline",
            },
        }
        clock = StreamClock.from_handshake(handshake)
        assert clock.host_anchor == HOST_ANCHOR
        assert clock.wall_anchor == WALL_ANCHOR
        assert clock.to_unix(HOST_ANCHOR + 1.5) == WALL_ANCHOR + 1.5

    def test_from_handshake_missing_clock_defaults_to_identity(self):
        clock = StreamClock.from_handshake({})
        assert clock.host_anchor == 0.0
        assert clock.wall_anchor == 0.0
        assert clock.to_unix(123.5) == 123.5
        assert clock.to_host(123.5) == 123.5

    def test_from_handshake_reports_ios_timebase(self):
        clock = StreamClock.from_handshake(
            {"clock": {"timebase": "mach_absolute_time_seconds",
                       "host_anchor": HOST_ANCHOR, "wall_anchor": WALL_ANCHOR}}
        )
        assert clock.timebase == "mach_absolute_time_seconds"

    def test_from_handshake_accepts_android_timebase(self):
        """The Android server's honest timebase is accepted, not rejected, and the
        anchor math is identical — the conversion never depended on the timebase."""
        clock = StreamClock.from_handshake(
            {"clock": {"timebase": "android_elapsed_realtime_seconds",
                       "host_anchor": HOST_ANCHOR, "wall_anchor": WALL_ANCHOR}}
        )
        assert clock.timebase == "android_elapsed_realtime_seconds"
        assert clock.timebase in KNOWN_TIMEBASES
        assert clock.to_unix(HOST_ANCHOR + 1.5) == WALL_ANCHOR + 1.5

    def test_absent_timebase_defaults_to_mach_for_backward_compat(self):
        """A handshake predating the timebase field is iOS/mach by construction."""
        clock = StreamClock.from_handshake(
            {"clock": {"host_anchor": HOST_ANCHOR, "wall_anchor": WALL_ANCHOR}}
        )
        assert clock.timebase == "mach_absolute_time_seconds"

    def test_unrecognized_timebase_warns_but_still_converts(self, caplog):
        """An unknown timebase is non-fatal — the anchor math is timebase-agnostic —
        but it is surfaced, because it means a platform newer than this client."""
        with caplog.at_level(logging.WARNING, logger="irtsp"):
            clock = StreamClock.from_handshake(
                {"clock": {"timebase": "some_future_platform_clock",
                           "host_anchor": HOST_ANCHOR, "wall_anchor": WALL_ANCHOR}}
            )
        assert clock.timebase == "some_future_platform_clock"
        assert clock.timebase not in KNOWN_TIMEBASES
        assert clock.to_unix(HOST_ANCHOR + 2.0) == WALL_ANCHOR + 2.0  # still correct
        assert any("unrecognized clock timebase" in r.message for r in caplog.records)

    def test_default_constructed_clock_is_mach(self):
        """The dataclass default keeps existing StreamClock(...) call sites valid."""
        clock = StreamClock(host_anchor=HOST_ANCHOR, wall_anchor=WALL_ANCHOR)
        assert clock.timebase == "mach_absolute_time_seconds"


# --------------------------------------------------------------------------- #
# slerp
# --------------------------------------------------------------------------- #


class TestSlerp:
    def test_endpoints(self):
        a = IDENTITY
        b = quat_axis_angle((1.0, 2.0, 3.0), math.radians(120.0))
        assert_quat_close(slerp(a, b, 0.0), a, abs_tol=1e-12)
        assert_quat_close(slerp(a, b, 1.0), b, abs_tol=1e-12)

    def test_unit_norm_along_path(self):
        a = IDENTITY
        b = quat_axis_angle((1.0, 2.0, 3.0), math.radians(170.0))
        for i in range(11):
            t = i / 10.0
            q = slerp(a, b, t)
            assert q.norm == pytest.approx(1.0, abs=1e-9)

    def test_known_midpoint(self):
        """identity -> 90 deg about z: midpoint is 45 deg about z."""
        mid = slerp(IDENTITY, quat_about_z(math.pi / 2), 0.5)
        assert_quat_close(mid, quat_about_z(math.pi / 4))
        quarter = slerp(IDENTITY, quat_about_z(math.pi / 2), 0.25)
        assert_quat_close(quarter, quat_about_z(math.pi / 8))

    def test_shortest_path_when_dot_negative(self):
        """A negated endpoint (same rotation, dot < 0) must not flip the path."""
        a = IDENTITY
        b = quat_about_z(math.pi / 2)
        b_neg = Quat(-b.x, -b.y, -b.z, -b.w)
        assert a.x * b_neg.x + a.y * b_neg.y + a.z * b_neg.z + a.w * b_neg.w < 0

        # Same interior points as with the un-negated b...
        for t in (0.25, 0.5, 0.75):
            assert_quat_close(slerp(a, b_neg, t), slerp(a, b, t))
        # ...so t=1 lands on -b_neg == b (the flipped representative).
        assert_quat_close(slerp(a, b_neg, 1.0), b)
        # And the norm stays unit along the flipped path too.
        for i in range(11):
            assert slerp(a, b_neg, i / 10.0).norm == pytest.approx(1.0, abs=1e-9)

    def test_near_parallel_is_stable(self):
        """Nearly-identical quaternions must not blow up (nlerp fallback)."""
        a = IDENTITY
        b = Quat(1e-9, 0.0, 0.0, 1.0).normalized()
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = slerp(a, b, t)
            assert all(math.isfinite(c) for c in q)
            assert q.norm == pytest.approx(1.0, abs=1e-12)
            assert_quat_close(q, a, abs_tol=1e-6)

    def test_identical_quats(self):
        a = quat_axis_angle((3.0, -1.0, 2.0), 1.234)
        assert_quat_close(slerp(a, a, 0.5), a, abs_tol=1e-12)

    def test_small_angle_nlerp_matches_geodesic(self):
        """In the nlerp regime (dot > 0.9995) the result still tracks the arc."""
        eps = 0.02  # rotation angle, rad -> quaternion-space angle 0.01 rad
        a = IDENTITY
        b = quat_about_z(eps)
        dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
        assert dot > 0.9995  # confirms we exercise the nlerp branch
        assert_quat_close(slerp(a, b, 0.5), quat_about_z(eps / 2))
        assert_quat_close(slerp(a, b, 0.25), quat_about_z(eps / 4), abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# interpolate_pose
# --------------------------------------------------------------------------- #


def make_pose(
    *,
    host_ts: float,
    unix_ts: float,
    seq: int,
    position: Vec3,
    orientation: Quat,
    tracking: Tracking,
) -> Pose:
    return Pose(
        host_ts=host_ts,
        unix_ts=unix_ts,
        seq=seq,
        position=position,
        orientation=orientation,
        tracking=tracking,
    )


class TestInterpolatePose:
    def setup_method(self):
        self.a = make_pose(
            host_ts=100.0,
            unix_ts=WALL_ANCHOR + 10.0,
            seq=1,
            position=Vec3(0.0, 0.0, 0.0),
            orientation=IDENTITY,
            tracking=Tracking.NORMAL,
        )
        self.b = make_pose(
            host_ts=102.0,
            unix_ts=WALL_ANCHOR + 12.0,
            seq=2,
            position=Vec3(2.0, 4.0, -6.0),
            orientation=quat_about_z(math.pi / 2),
            tracking=Tracking.LIMITED,
        )

    def test_midpoint(self):
        mid = interpolate_pose(self.a, self.b, WALL_ANCHOR + 11.0)
        assert mid.position == Vec3(1.0, 2.0, -3.0)  # linear position lerp
        assert mid.host_ts == pytest.approx(101.0)
        assert mid.unix_ts == pytest.approx(WALL_ANCHOR + 11.0)
        assert_quat_close(mid.orientation, quat_about_z(math.pi / 4))  # slerp'd
        assert mid.seq == self.b.seq

    def test_tracking_is_worse_of_endpoints(self):
        mid = interpolate_pose(self.a, self.b, WALL_ANCHOR + 11.0)
        assert mid.tracking is Tracking.LIMITED  # NORMAL + LIMITED -> LIMITED

        worse_a = make_pose(
            host_ts=self.a.host_ts,
            unix_ts=self.a.unix_ts,
            seq=self.a.seq,
            position=self.a.position,
            orientation=self.a.orientation,
            tracking=Tracking.NONE,
        )
        assert interpolate_pose(worse_a, self.b, WALL_ANCHOR + 11.0).tracking is (
            Tracking.NONE
        )

        good_b = make_pose(
            host_ts=self.b.host_ts,
            unix_ts=self.b.unix_ts,
            seq=self.b.seq,
            position=self.b.position,
            orientation=self.b.orientation,
            tracking=Tracking.NORMAL,
        )
        assert interpolate_pose(self.a, good_b, WALL_ANCHOR + 11.0).tracking is (
            Tracking.NORMAL
        )

    def test_clamps_below_range(self):
        early = interpolate_pose(self.a, self.b, WALL_ANCHOR + 5.0)
        assert early.position == self.a.position
        assert early.unix_ts == pytest.approx(self.a.unix_ts)
        assert early.host_ts == pytest.approx(self.a.host_ts)
        assert_quat_close(early.orientation, self.a.orientation, abs_tol=1e-12)

    def test_clamps_above_range(self):
        late = interpolate_pose(self.a, self.b, WALL_ANCHOR + 1e6)
        assert late.position == self.b.position
        assert late.unix_ts == pytest.approx(self.b.unix_ts)
        assert late.host_ts == pytest.approx(self.b.host_ts)
        assert_quat_close(late.orientation, self.b.orientation, abs_tol=1e-12)

    def test_degenerate_equal_timestamps_returns_a(self):
        b_same_time = make_pose(
            host_ts=self.a.host_ts,
            unix_ts=self.a.unix_ts,  # b.unix_ts == a.unix_ts
            seq=3,
            position=Vec3(9.0, 9.0, 9.0),
            orientation=quat_about_z(1.0),
            tracking=Tracking.NONE,
        )
        assert interpolate_pose(self.a, b_same_time, WALL_ANCHOR + 10.5) == self.a

    def test_degenerate_reversed_timestamps_returns_a(self):
        assert interpolate_pose(self.b, self.a, WALL_ANCHOR + 11.0) == self.b
