"""End-to-end tests for :mod:`irtsp.audio` against ``tests/mockserver.MockAudioPhone``.

Every test speaks the real wire protocol over loopback TCP: the RTSP handshake
(OPTIONS/DESCRIBE/SETUP/PLAY, Basic and Digest auth), then ``$``-framed
interleaved RTP and compound RTCP Sender Reports, exactly as the iRTSP app's
RTSP server frames them.

The point of this reader is that it does *not* smooth over the phone's capture
gaps, so most of what is asserted here is arithmetic: which block a packet lands
in, what its anchor is, and how many frames are missing in front of it.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):  # import irtsp + mockserver uninstalled
    if _p not in sys.path:
        sys.path.insert(0, _p)

import irtsp
from irtsp.audio import ntp_to_unix, parse_rtp, parse_sdp, parse_sender_reports
from irtsp.wire import ProtocolError
from mockserver import MockAudioPhone, rtp_packet, sender_report

#: TOC byte for a 20 ms hybrid Opus frame, one frame per packet (960 @ 48 kHz).
OPUS_TOC_20MS = bytes([0x78])

#: A minimal AAC payload: the 4-byte AU header section, then the access unit.
AAC_AU = bytes([0x00, 0x10, 0x02, 0x08]) + b"\xde\xad\xbe\xef"


# --------------------------------------------------------------------- helpers


def connect(phone: MockAudioPhone, **kwargs) -> irtsp.AudioStream:
    """Open a stream against a mock phone."""
    return irtsp.audio_stream(
        "127.0.0.1", port=phone.port, path=phone.path, timeout=2.0, **kwargs
    )


def blocks(audio, phone: MockAudioPhone, timeout: float = 3.0) -> list:
    """Every block the stream yields once the phone stops sending.

    The final, still-open block is flushed at end of stream, so this returns the
    whole capture — nothing is left waiting for a packet that never comes.
    """
    phone.end_stream()
    out: list = []
    t = threading.Thread(target=lambda: out.extend(audio), daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"stream did not end within {timeout}s (got {len(out)} blocks)"
    return out


# -------------------------------------------------------------------- fixtures


@pytest.fixture
def mic(request):
    """Factory for mock RTSP phones; takes any :class:`MockAudioPhone` option."""

    def make(**kwargs) -> MockAudioPhone:
        phone = MockAudioPhone(**kwargs).start()
        request.addfinalizer(phone.close)
        return phone

    return make


# ------------------------------------------------------------------------ L16


def test_l16_mono_decodes_big_endian_samples(mic):
    phone = mic(codec="l16", sample_rate=48000, channels=1,
                start_seq=100, start_timestamp=7000)
    audio = connect(phone)
    assert (audio.codec, audio.sample_rate, audio.channels) == ("l16", 48000, 1)
    assert audio.control_url == f"rtsp://127.0.0.1:{phone.port}/live/trackID=1"

    phone.emit_l16([0, 1, -1, 32767, -32768])
    phone.emit_l16([5, 6, 7, 8, 9])
    (block,) = blocks(audio, phone)

    assert (block.codec, block.sample_rate, block.channels) == ("l16", 48000, 1)
    assert block.rtp_timestamp == 7000
    assert block.rtp_timestamp_raw == 7000  # no rollover yet: the two agree
    assert block.seq_start == 100
    assert block.n_packets == 2
    assert (block.gap_frames, block.lost_packets) == (0, 0)
    assert block.n_frames == 10
    assert block.duration == pytest.approx(10 / 48000)
    assert block.packets == (
        b"\x00\x00\x00\x01\xff\xff\x7f\xff\x80\x00",  # big-endian on the wire
        b"\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09",
    )
    samples = block.samples
    assert samples.dtype == np.int16
    assert samples.dtype.byteorder in ("=", "|")  # native order, not big-endian
    assert samples.shape == (10, 1)
    assert samples[:, 0].tolist() == [0, 1, -1, 32767, -32768, 5, 6, 7, 8, 9]


def test_l16_stereo_deinterleaves_channels(mic):
    phone = mic(codec="l16", sample_rate=44100, channels=2)
    audio = connect(phone)
    phone.emit_l16([1, -1, 2, -2, 3, -3])  # 3 frames of (left, right)
    (block,) = blocks(audio, phone)

    assert block.channels == 2
    assert block.sample_rate == 44100
    assert block.n_frames == 3
    assert block.samples.shape == (3, 2)
    assert block.samples[:, 0].tolist() == [1, 2, 3]
    assert block.samples[:, 1].tolist() == [-1, -2, -3]


# ----------------------------------------------------------- the capture gap


def test_capture_gap_seam_reports_the_exact_hole(mic):
    """The headline: a marker + timestamp step is a *measured* hole, not a guess."""
    phone = mic(codec="l16", channels=1, start_timestamp=1000)
    audio = connect(phone)

    for _ in range(3):
        phone.emit_l16([7] * 480)
    # The mic stalled for 1234 frames; the app re-anchored and marked the seam.
    phone.emit_l16([9] * 480, gap_frames=1234, marker=True)
    phone.emit_l16([9] * 480)
    first, second = blocks(audio, phone)

    assert first.n_packets == 3
    assert first.rtp_timestamp == 1000
    assert (first.gap_frames, first.lost_packets) == (0, 0)
    assert first.marker is False

    assert second.n_packets == 2
    assert second.gap_frames == 1234  # exactly the frames that were never captured
    assert second.lost_packets == 0  # and not a packet was lost doing it
    assert second.marker is True
    # The new anchor is the old one plus the audio we got plus the hole.
    assert second.rtp_timestamp == 1000 + 3 * 480 + 1234
    assert second.samples[0, 0] == 9


def test_block_anchors_chain_through_frames_and_gaps(mic):
    phone = mic(codec="l16", channels=1, start_timestamp=500)
    audio = connect(phone)
    phone.emit_l16([0] * 240)
    phone.emit_l16([0] * 240, gap_frames=17, marker=True)
    phone.emit_l16([0] * 240)
    phone.emit_l16([0] * 240, gap_frames=99, marker=True)
    phone.emit_l16([0] * 240)
    got = blocks(audio, phone)

    assert [b.gap_frames for b in got] == [0, 17, 99]
    assert [b.n_frames for b in got] == [240, 480, 480]
    anchor = 500
    for block in got:
        anchor += block.gap_frames
        assert block.rtp_timestamp == anchor
        anchor += block.n_frames


def test_marker_without_a_hole_still_seams_but_gap_is_zero(mic):
    phone = mic(codec="l16", channels=1, start_timestamp=0)
    audio = connect(phone)
    phone.emit_l16([0] * 480)
    phone.emit_l16([0] * 480, marker=True)  # a seam the app flagged with no lost time
    first, second = blocks(audio, phone)

    assert second.marker is True
    assert second.gap_frames == 0
    assert second.rtp_timestamp == first.rtp_timestamp + first.n_frames


# ------------------------------------------------------------- transit loss


def test_transit_loss_is_never_reported_as_a_capture_gap(mic):
    phone = mic(codec="l16", channels=1, start_seq=10, start_timestamp=0)
    audio = connect(phone)
    phone.emit_l16([0] * 480)
    phone.emit_l16([0] * 480)
    # Two packets the phone never managed to send: the sequence jumps, and so
    # does the timestamp — but no audio was *missed*, only lost.
    phone.emit_l16([0] * 480, lost=2)
    phone.emit_l16([0] * 480)
    first, second = blocks(audio, phone)

    assert first.n_packets == 2
    assert (first.gap_frames, first.lost_packets) == (0, 0)
    assert second.lost_packets == 2
    assert second.gap_frames == 0  # loss is not a hole in the capture
    assert second.marker is False
    assert second.seq_start == 14  # 10, 11, [12, 13 lost], 14
    assert second.rtp_timestamp == 4 * 480


def test_sequence_wraparound_is_not_a_discontinuity(mic):
    phone = mic(codec="l16", channels=1, start_seq=65534)
    audio = connect(phone)
    for _ in range(4):  # 65534, 65535, 0, 1
        phone.emit_l16([0] * 480)
    (block,) = blocks(audio, phone)

    assert block.seq_start == 65534
    assert block.n_packets == 4  # the u16 rollover kept the run contiguous
    assert block.lost_packets == 0


# ------------------------------------------- the 32-bit RTP timestamp rollover


def test_timestamp_wraparound_keeps_anchors_monotonic(mic):
    """RFC 3550 randomizes the RTP base, so the rollover is not a rare event."""
    phone = mic(codec="l16", channels=1, start_timestamp=2**32 - 960)
    audio = connect(phone)
    for _ in range(4):  # crosses the 32-bit rollover mid-block
        phone.emit_l16([0] * 480)
    phone.emit_l16([0] * 480, marker=True)  # a seam on the far side of the wrap
    phone.emit_l16([0] * 480)
    first, second = blocks(audio, phone)

    assert first.rtp_timestamp == 2**32 - 960
    assert first.n_packets == 4  # the wrap itself is not a seam
    assert first.gap_frames == 0
    # Unwrapped: strictly forwards, never a 24.85-hour leap backwards.
    assert second.rtp_timestamp == 2**32 + 960
    assert second.rtp_timestamp > first.rtp_timestamp
    assert second.gap_frames == 0
    # ...while the wire value is exactly what the phone put on it.
    assert second.rtp_timestamp_raw == 960


def test_wrap_coinciding_with_a_capture_gap_still_measures_the_gap(mic):
    phone = mic(codec="l16", channels=1, start_timestamp=2**32 - 480)
    audio = connect(phone)
    phone.emit_l16([0] * 480)  # ends exactly on the rollover
    phone.emit_l16([0] * 480, gap_frames=777, marker=True)
    phone.emit_l16([0] * 480)
    first, second = blocks(audio, phone)

    assert first.rtp_timestamp == 2**32 - 480
    assert second.gap_frames == 777  # not 2**32 ± 777
    assert second.rtp_timestamp == 2**32 + 777
    assert second.rtp_timestamp_raw == 777


# ------------------------------------------------------------------- RTCP SR


def test_sender_reports_pair_rtp_ticks_with_wall_time(mic):
    phone = mic(codec="l16", channels=1, start_timestamp=5000)
    audio = connect(phone)
    phone.emit_l16([0] * 480)
    phone.emit_sr(ntp_unix=1_700_000_000.5, rtp_timestamp=5480,
                  packet_count=1, octet_count=960)
    phone.emit_l16([0] * 480)
    phone.emit_sr(ntp_unix=1_700_000_005.25, rtp_timestamp=5960, packet_count=2)
    phone.emit_l16([0] * 480)
    (block,) = blocks(audio, phone)

    assert block.n_packets == 3  # RTCP between packets never breaks a block
    assert len(audio.sender_reports) == 2
    rtp, ntp = audio.sender_reports[0]  # it unpacks like the pair it is
    assert rtp == 5480
    assert ntp == pytest.approx(1_700_000_000.5, abs=1e-6)
    assert audio.sender_reports[1].rtp_timestamp == 5960
    assert audio.sender_reports[1].ntp_unix == pytest.approx(1_700_000_005.25, abs=1e-6)
    # The pairs live on the same axis as the block anchors, so this is meaningful:
    seconds = (rtp - block.rtp_timestamp) / block.sample_rate
    assert ntp - seconds == pytest.approx(1_700_000_000.5 - 480 / 48000, abs=1e-6)


def test_sender_report_after_a_wrap_lands_on_the_block_axis(mic):
    phone = mic(codec="l16", channels=1, start_timestamp=2**32 - 480)
    audio = connect(phone)
    phone.emit_l16([0] * 480)
    phone.emit_l16([0] * 480)  # wrapped: raw timestamp is now 0
    phone.emit_sr(ntp_unix=1_700_000_000.0, rtp_timestamp=480)
    phone.emit_l16([0] * 480)
    blocks(audio, phone)

    (report,) = audio.sender_reports
    assert report.rtp_timestamp == 2**32 + 480  # unwrapped like the blocks


def test_sender_report_parsing_walks_the_compound_packet():
    packet = sender_report(
        ssrc=0x1A2B3C4D, ntp_unix=1_700_000_123.25, rtp_timestamp=987654,
        packet_count=42, octet_count=4242,
    )
    assert len(packet) > 28  # SR + SDES, as the app sends it
    (report,) = parse_sender_reports(packet)  # the SDES is ignored, not misread
    assert report.rtp_timestamp == 987654
    assert report.ntp_unix == pytest.approx(1_700_000_123.25, abs=1e-6)
    # and an SR on its own parses identically
    alone = sender_report(ssrc=1, ntp_unix=1.5, rtp_timestamp=7, with_sdes=False)
    assert parse_sender_reports(alone)[0].rtp_timestamp == 7
    assert ntp_to_unix((2_208_988_800 + 10) << 32 | 2**31) == pytest.approx(10.5)


# ------------------------------------------------------------------ AAC / Opus


def test_aac_markers_are_not_capture_gaps(mic):
    phone = mic(codec="aac", sample_rate=44100, channels=1)
    audio = connect(phone)
    assert audio.codec == "aac"
    assert "config=1190" in audio.fmtp
    for _ in range(3):
        # Every unfragmented AAC packet is marked — it means end-of-access-unit.
        phone.emit_rtp(AAC_AU, frames=1024, marker=True)
    (block,) = blocks(audio, phone)

    assert block.n_packets == 3  # not one block per marker
    assert block.gap_frames == 0
    assert block.samples is None  # compressed frames, not samples
    assert block.packets == (AAC_AU,) * 3
    assert block.n_frames is None  # the AU length isn't on the wire
    assert block.duration is None
    assert block.sample_rate == 44100


def test_aac_still_splits_on_lost_packets(mic):
    phone = mic(codec="aac", sample_rate=44100, channels=1, start_seq=7)
    audio = connect(phone)
    phone.emit_rtp(AAC_AU, frames=1024, marker=True)
    phone.emit_rtp(AAC_AU, frames=1024, marker=True, lost=1)
    first, second = blocks(audio, phone)

    assert first.n_packets == 1
    assert second.lost_packets == 1
    assert second.gap_frames == 0
    assert second.seq_start == 9


def test_opus_packets_are_exposed_and_gaps_measured(mic):
    phone = mic(codec="opus", channels=2)
    audio = connect(phone)
    assert (audio.codec, audio.sample_rate, audio.channels) == ("opus", 48000, 2)

    packet = OPUS_TOC_20MS + b"\x01\x02\x03"
    phone.emit_rtp(packet, frames=960)
    phone.emit_rtp(packet, frames=960)
    phone.emit_rtp(packet, frames=960, gap_frames=480, marker=True)
    phone.emit_rtp(packet, frames=960)
    first, second = blocks(audio, phone)

    assert first.samples is None
    assert first.packets == (packet, packet)
    assert first.n_frames == 1920  # read back from the Opus TOC byte
    assert second.gap_frames == 480  # so the hole is measurable for Opus too
    assert second.marker is True
    assert second.rtp_timestamp == first.rtp_timestamp + 1920 + 480


def test_opus_channels_come_from_sprop_stereo(mic):
    # An Opus rtpmap always reads opus/48000/2, whatever the phone captured.
    phone = mic(codec="opus", channels=1)
    audio = connect(phone)
    assert "sprop-stereo=0" in audio.fmtp
    assert audio.channels == 1
    audio.close()


# ------------------------------------------------------------------------ auth


def test_digest_auth_handshake(mic):
    phone = mic(auth="digest", username="alice", password="s3cret")
    audio = connect(phone, username="alice", password="s3cret")
    phone.emit_l16([1, 2, 3, 4])
    (block,) = blocks(audio, phone)

    assert block.n_frames == 4
    assert phone.challenges_sent == 1  # challenged once, then satisfied every time
    assert [m for m, _ in phone.requests][:5] == [
        "OPTIONS", "DESCRIBE", "DESCRIBE", "SETUP", "PLAY",
    ]
    sent = [h for h in phone.authorizations if h]
    assert sent and all(h.startswith("Digest ") for h in sent)
    # Digest was offered alongside Basic; the password must not have gone plain.
    assert not any(h.startswith("Basic") for h in sent)


def test_basic_auth_handshake(mic):
    import base64

    phone = mic(auth="basic", username="bob", password="hunter2")
    audio = connect(phone, username="bob", password="hunter2")
    phone.emit_l16([1, 2])
    (block,) = blocks(audio, phone)

    assert block.n_frames == 2
    sent = [h for h in phone.authorizations if h]
    assert sent and all(h.startswith("Basic ") for h in sent)
    assert base64.b64decode(sent[0].split()[1]) == b"bob:hunter2"


def test_wrong_password_fails_loudly(mic):
    phone = mic(auth="digest", username="alice", password="s3cret")
    with pytest.raises(ProtocolError, match="wrong username/password"):
        connect(phone, username="alice", password="nope")


def test_missing_credentials_say_so(mic):
    phone = mic(auth="digest")
    with pytest.raises(ProtocolError, match="credentials"):
        connect(phone)


def test_credentials_may_ride_in_the_url(mic):
    phone = mic(auth="basic", username="bob", password="hunter2")
    audio = irtsp.audio_stream(
        f"rtsp://bob:hunter2@127.0.0.1:{phone.port}/live", timeout=2.0
    )
    phone.emit_l16([1, 2])
    (block,) = blocks(audio, phone)
    assert block.n_frames == 2


# -------------------------------------------------------------------- teardown


def test_close_tears_the_session_down(mic):
    phone = mic()
    audio = connect(phone)
    phone.emit_l16([0] * 480)
    audio.close()

    assert phone.torn_down.wait(2.0)
    assert audio.closed
    assert ("TEARDOWN", audio.url) in phone.requests
    audio.close()  # idempotent


def test_context_manager(mic):
    phone = mic()
    with connect(phone) as audio:
        assert not audio.closed
        assert audio.session_id == phone.session_id
    assert audio.closed


# ------------------------------------------------------------- parsing details


def test_an_rtsp_response_between_frames_is_stepped_over(mic):
    phone = mic(codec="l16", channels=1)
    audio = connect(phone)
    phone.emit_l16([1, 2])
    # A late RTSP response (the phone answering a keepalive) lands between two
    # interleaved frames; body and all, it must not desync the reader.
    phone.send_raw(
        b"RTSP/1.0 200 OK\r\nCSeq: 9\r\nSession: 0BADCAFE\r\n"
        b"Content-Length: 4\r\n\r\nping"
    )
    phone.emit_l16([3, 4])
    (block,) = blocks(audio, phone)

    assert block.n_packets == 2
    assert block.samples[:, 0].tolist() == [1, 2, 3, 4]


def test_parse_sdp_finds_the_audio_track_past_the_video_one(mic):
    phone = mic(codec="l16", sample_rate=48000, channels=2, video=True)
    base = phone.content_base
    media = parse_sdp(phone.sdp(), base)

    assert media.payload_type == 97
    assert media.codec == "l16"
    assert media.sample_rate == 48000
    assert media.channels == 2
    # relative control, resolved against the Content-Base's trailing slash
    assert media.control == base.rstrip("/") + "/trackID=1"


def test_parse_sdp_without_an_audio_track_says_so():
    sdp = "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=control:trackID=0\r\n"
    with pytest.raises(ProtocolError, match="no audio track"):
        parse_sdp(sdp, "rtsp://127.0.0.1:8554/live/")


def test_parse_sdp_defaults_mono_when_the_rtpmap_omits_channels():
    sdp = "v=0\r\nm=audio 0 RTP/AVP 97\r\na=rtpmap:97 L16/48000\r\na=control:trackID=1\r\n"
    media = parse_sdp(sdp, "rtsp://127.0.0.1:8554/live/")
    assert media.channels == 1


def test_parse_rtp_skips_csrcs_extension_and_padding():
    packet = rtp_packet(
        b"\x01\x02\x03\x04", seq=9, timestamp=1234, ssrc=0xDEADBEEF, marker=True,
        csrcs=(1, 2), extension=b"\xaa" * 8, padding=3,
    )
    parsed = parse_rtp(packet)

    assert parsed.payload == b"\x01\x02\x03\x04"  # nothing but the codec bytes
    assert parsed.seq == 9
    assert parsed.timestamp == 1234
    assert parsed.ssrc == 0xDEADBEEF
    assert parsed.marker is True
    assert parsed.payload_type == 97


def test_parse_rtp_rejects_what_is_not_rtp():
    with pytest.raises(ProtocolError, match="RTP version"):
        parse_rtp(b"\x00" * 16)
    with pytest.raises(ProtocolError, match="needs"):
        parse_rtp(b"\x80\x61\x00")
