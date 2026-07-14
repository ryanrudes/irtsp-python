"""Cross-platform wire compatibility: the Android server's handshake, parsed by this client.

The fixture ``tests/fixtures/android_handshake.json`` is the VERBATIM JSON emitted by the
Kotlin port's ``Handshake.json(...)`` (iRTSP repo, ``Android/core`` module) for a default
Android session — captured from the Kotlin source of truth, not hand-written here. The point
of this test is that a client which works against the iOS app works IDENTICALLY against the
Android app: same 64-byte records, same clock model, same ``streams`` contract. The only
honest differences are (a) the clock timebase names the real Android host clock, and (b) the
phone declares ``streams.depth = false`` because no Android device can source metric depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from irtsp import KNOWN_TIMEBASES
from irtsp.session import Handshake

FIXTURE = Path(__file__).parent / "fixtures" / "android_handshake.json"


@pytest.fixture(scope="module")
def android_raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def android(android_raw) -> Handshake:
    return Handshake(android_raw)


def test_it_is_an_irtsp_imu_handshake(android: Handshake):
    assert android.protocol == "irtsp-imu"
    assert android.version == 1
    # The 64-byte stride is the whole parser's premise; an Android phone must agree byte-for-byte.
    assert android.record_bytes == 64


def test_clock_timebase_is_the_android_one_and_recognized(android: Handshake):
    assert android.clock.timebase == "android_elapsed_realtime_seconds"
    assert android.clock.timebase in KNOWN_TIMEBASES  # not a warning path


def test_clock_anchor_converts_identically_to_ios(android: Handshake):
    """The conversion is anchor arithmetic and does not know or care about the platform."""
    c = android.clock
    assert c.host_anchor == pytest.approx(123456.789012)
    assert c.wall_anchor == pytest.approx(1751990400.123456)
    # A host timestamp 2.5 s after the anchor lands 2.5 s after the wall anchor — same as iOS.
    assert c.to_unix(c.host_anchor + 2.5) == pytest.approx(c.wall_anchor + 2.5)
    assert c.to_host(c.wall_anchor + 2.5) == pytest.approx(c.host_anchor + 2.5)


def test_streams_parse_as_a_dict_including_the_additive_depth_key(android: Handshake):
    # streams is read as a dict, so the Android-only `depth` key is additive and safe — it does
    # not disturb any stream the iOS client already knew about.
    assert android.streams["imu"] is True
    assert android.streams["intrinsics"] is True
    assert android.streams["pose"] is True
    assert android.streams["gnss"] is False


def test_the_phone_honestly_denies_depth_in_band(android: Handshake):
    """No Android device can source metric depth, so it says so in the handshake rather than
    advertising a channel that would then refuse the connection."""
    assert android.streams["depth"] is False


def test_video_descriptor_matches(android: Handshake):
    assert android.video_url == "rtsp://10.0.0.5:8554/live"
    assert android.video_codec == "H264"
    assert android.video_clock_rate == 90000


def test_nothing_the_client_does_not_understand_is_lost(android: Handshake, android_raw: dict):
    # Handshake preserves the full raw dict; forward-compat by construction.
    assert android.raw == android_raw
