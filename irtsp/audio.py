"""Raw RTP audio — the microphone track with its capture gaps left visible.

The usual way to get audio off the phone is ``ffmpeg -i rtsp://… out.wav``, and
for *listening* that is fine. For **measurement** it quietly lies: a WAV file is
a dense array of samples, so all you can do when you read it back is assume
sample *n* was captured at ``anchor + n / rate``. iRTSP deliberately breaks that
assumption when it has to — when the microphone capture stalls, the app
re-anchors the RTP timestamp to the *true* capture time instead of pretending
the missing audio happened. That leaves a **real hole in the timeline**, and the
first packet after it carries the RTP **marker bit**. FFmpeg papers the hole
over (silence, or a resampler nudge); this reader hands it to you.

So the deal here is: no decode, no resample, no concealment. You get the RTP
timestamp of every contiguous block and the exact size of every hole between
them::

    import irtsp

    with irtsp.audio_stream("192.168.1.24") as audio:
        for block in audio:
            if block.gap_frames:
                print(f"{block.gap_frames} frames of capture missing before this block")
            x = block.samples          # (n, channels) int16 — needs numpy
            t0 = block.rtp_timestamp   # RTP ticks of x[0]: this block's anchor

Landing that on wall-clock time is the consumer's call, and everything needed is
here: :attr:`AudioStream.sender_reports` is the raw ``(rtp_timestamp, ntp_unix)``
pairs from the phone's RTCP Sender Reports, built from the *same* session anchor
that stamps every odometry record (integration guide §4). Fit them however your
rig wants to — this module never interpolates on your behalf.

Scope, deliberately narrow: the **audio track only**, over **TCP-interleaved**
RTSP only. TCP removes transit loss and makes framing deterministic, which is
the whole point — a sequence discontinuity then means the *phone* dropped
packets, never the network, and it is reported as :attr:`AudioBlock.lost_packets`
and never confused with :attr:`AudioBlock.gap_frames`. Those two have different
causes and different remedies, so they are never summed together.

One thing the reader insists on doing for you: the wire's RTP timestamp is 32
bits and RFC 3550 requires it to *start at a random value*, so the rollover is
not "once every 24.85 hours" — it is uniformly distributed, and a stream whose
random base lands high can wrap seconds in. Unhandled, that presents as the
clock jumping ~24.85 hours backwards while every other field looks perfectly
healthy, which is the worst possible failure for a measurement rig. So
:attr:`AudioBlock.rtp_timestamp` is **unwrapped** to a monotonic 64-bit count
(the raw wire value stays available as :attr:`AudioBlock.rtp_timestamp_raw`),
and all the gap arithmetic below runs on the unwrapped values — the reader knows
more about stream continuity than a consumer downstream of it ever can.

L16 (uncompressed big-endian int16, RFC 3551) is the path this is built for and
the one that decodes to samples. AAC and Opus stream fine — their payloads come
back in :attr:`AudioBlock.packets` for you to feed a decoder — but note that an
AAC marker bit means *end of access unit*, not a capture gap, and is never read
as one.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import socket
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterator, NamedTuple
from urllib.parse import urlsplit

from .wire import ConnectionClosed, ProtocolError

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = [
    "audio_stream",
    "AudioStream",
    "AudioBlock",
    "SenderReport",
    "AUDIO_PAYLOAD_TYPE",
]

log = logging.getLogger("irtsp.audio")

#: RTP payload type iRTSP uses for audio, whatever the codec (video is 96).
AUDIO_PAYLOAD_TYPE = 97

#: Seconds between the NTP epoch (1900-01-01) and the Unix epoch (1970-01-01).
NTP_UNIX_DELTA = 2_208_988_800

#: rtpmap encoding name → the short codec name this module reports.
_CODECS = {"l16": "l16", "mpeg4-generic": "aac", "opus": "opus"}

_RTP_HEADER = 12
_RTCP_SR = 200  # RTCP packet type of a Sender Report


class SenderReport(NamedTuple):
    """One RTCP Sender Report, as the pair that anchors RTP time to the wall.

    Unpacks like the tuple it is (``rtp, unix = sr``). The phone emits one
    0.15 s after PLAY and every ~5 s after that, so a minute of audio gives you
    a dozen anchors to fit a line through — that fit is yours to make.
    """

    #: RTP ticks, unwrapped onto the *same* axis as :attr:`AudioBlock.rtp_timestamp`
    #: so the two can be subtracted directly, rollover or not.
    rtp_timestamp: int
    ntp_unix: float  #: the wall-clock instant of that tick, unix seconds


class _Packet(NamedTuple):
    """One parsed RTP packet — header fields plus the payload we care about."""

    seq: int
    timestamp: int
    marker: bool
    payload_type: int
    ssrc: int
    payload: bytes


@dataclass(frozen=True, kw_only=True, match_args=False)  # no slots: cached_property below
class AudioBlock:
    """One **contiguous** run of audio: no hole, no loss, no seam inside it.

    A block is closed and a new one started the moment the wire says continuity
    broke — a marker bit (capture-gap seam), an RTP timestamp that doesn't
    continue the previous packet, or a sequence discontinuity. So within a block
    sample *n* really was captured at ``rtp_timestamp + n`` ticks; that is the
    guarantee this whole module exists to provide, and it is exactly the one a
    WAV file cannot make.

    Timestamps here are already unwrapped past the 32-bit RTP rollover — see
    :class:`AudioStream` for why that matters more than it sounds like it should.

    :attr:`gap_frames` and :attr:`lost_packets` describe what happened *before*
    this block, and they are never mixed: the first is capture the phone never
    got (a real hole in time), the second is capture the phone got but did not
    manage to send (audio you are missing, in a stretch of time that is not a
    hole). Both are 0 on the first block, which has nothing before it.
    """

    __match_args__ = ("codec", "rtp_timestamp", "gap_frames")

    packets: tuple[bytes, ...]  #: raw RTP payloads, in order — always populated
    codec: str  #: ``"l16"`` | ``"aac"`` | ``"opus"`` (or the rtpmap name, lowercased)
    sample_rate: int  #: frames/s, and also the RTP clock rate for every iRTSP codec
    channels: int  #: interleaved channel count
    #: RTP ticks of the **first sample of this block** — your anchor. **Unwrapped**:
    #: a monotonic 64-bit count that keeps rising past the wire's 32-bit rollover
    #: (see :class:`AudioStream`), so subtracting two blocks' anchors is always safe.
    rtp_timestamp: int
    rtp_timestamp_raw: int  #: the same instant as it appeared on the wire (32-bit)
    seq_start: int  #: RTP sequence number of the first packet
    n_packets: int  #: packets in this block
    gap_frames: int = 0  #: frames of capture missing immediately before this block
    lost_packets: int = 0  #: packets the phone skipped immediately before this block
    marker: bool = False  #: this block opened on a marker bit (a capture-gap seam)

    @cached_property
    def samples(self) -> "np.ndarray | None":
        """``(n, channels)`` int16 in native byte order — or ``None`` for AAC/Opus.

        L16 is big-endian on the wire (RFC 3551); the byte swap happens here, so
        what you get is a plain native-endian array you can hand to scipy. AAC
        and Opus are compressed frames, not samples: use :attr:`packets`.
        """
        if self.codec != "l16":
            return None
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AudioBlock.samples needs numpy — pip install 'irtsp[numpy]' "
                "(AudioBlock.packets is the raw big-endian payload, dependency-free)"
            ) from e
        raw = b"".join(self.packets)
        return np.frombuffer(raw, dtype=">i2").astype(np.int16).reshape(-1, self.channels)

    @property
    def n_frames(self) -> int | None:
        """Frames (per channel) in this block — ``None`` when the codec hides it.

        Exact for L16 and Opus (whose payloads declare their own frame size);
        ``None`` for AAC, where the frame count lives in the decoder config
        rather than on the wire.
        """
        total = 0
        for payload in self.packets:
            frames = _frames_in_payload(self.codec, payload, self.channels)
            if frames is None:
                return None
            total += frames
        return total

    @property
    def duration(self) -> float | None:
        """Length of this block in seconds, or ``None`` if :attr:`n_frames` is."""
        frames = self.n_frames
        return None if frames is None else frames / self.sample_rate

    def __repr__(self) -> str:  # pragma: no cover
        frames = self.n_frames
        extra = f" gap={self.gap_frames}" if self.gap_frames else ""
        extra += f" lost={self.lost_packets}" if self.lost_packets else ""
        return (
            f"<AudioBlock {self.codec} {self.sample_rate}Hz x{self.channels} "
            f"rtp={self.rtp_timestamp} frames={frames} packets={self.n_packets}{extra}>"
        )


# --------------------------------------------------------------------------- #
# Payload arithmetic
#
# Block assembly needs one number per packet: how many frames it carries, so the
# *next* packet's timestamp can be predicted and any shortfall named as a gap.
# --------------------------------------------------------------------------- #

#: Opus frame sizes in 48 kHz samples, by TOC config class (RFC 6716 §3.1).
_OPUS_SILK = (480, 960, 1920, 2880)  # 10 / 20 / 40 / 60 ms
_OPUS_HYBRID = (480, 960)  # 10 / 20 ms
_OPUS_CELT = (120, 240, 480, 960)  # 2.5 / 5 / 10 / 20 ms


def _opus_frames(payload: bytes) -> int | None:
    """Samples (at 48 kHz) in one Opus packet, read from its TOC byte."""
    if not payload:
        return None
    toc = payload[0]
    config, code = toc >> 3, toc & 0x03
    if config < 12:
        per_frame = _OPUS_SILK[config & 0x03]
    elif config < 16:
        per_frame = _OPUS_HYBRID[config & 0x01]
    else:
        per_frame = _OPUS_CELT[config & 0x03]
    if code == 0:
        count = 1
    elif code in (1, 2):
        count = 2
    else:  # code 3: an arbitrary frame count in the next byte's low 6 bits
        if len(payload) < 2:
            return None
        count = payload[1] & 0x3F
    return per_frame * count if count else None


def _frames_in_payload(codec: str, payload: bytes, channels: int) -> int | None:
    """Frames (per channel) in one packet, or ``None`` when the wire doesn't say.

    ``None`` disables the timestamp-continuity check for that packet — better a
    missed seam than a fabricated gap size.
    """
    if codec == "l16":
        stride = 2 * channels
        # The app always splits on whole frames; a partial frame means we are
        # reading the wrong bytes, so say so instead of rounding it away.
        if not payload or len(payload) % stride:
            raise ProtocolError(
                f"L16 payload of {len(payload)} bytes is not a whole number of "
                f"{channels}-channel frames — desynced?"
            )
        return len(payload) // stride
    if codec == "opus":
        return _opus_frames(payload)
    return None  # AAC: the AU length lives in the decoder config, not the wire


def parse_rtp(packet: bytes) -> _Packet:
    """Parse one RTP packet: fixed header, CSRCs, extension, padding.

    Everything the header says to skip is skipped properly — a CSRC list, a
    profile-specific header extension, and RFC 3550 padding — so
    :attr:`_Packet.payload` is exactly the codec bytes.
    """
    if len(packet) < _RTP_HEADER:
        raise ProtocolError(f"RTP packet needs ≥{_RTP_HEADER} bytes, got {len(packet)}")
    b0, b1 = packet[0], packet[1]
    version = b0 >> 6
    if version != 2:
        raise ProtocolError(f"RTP version {version}, expected 2 — desynced?")
    csrc_count = b0 & 0x0F
    offset = _RTP_HEADER + 4 * csrc_count
    if b0 & 0x10:  # extension header: 4-byte prelude + a u16 word count
        if len(packet) < offset + 4:
            raise ProtocolError("RTP header extension is truncated")
        (words,) = struct.unpack_from(">H", packet, offset + 2)
        offset += 4 + 4 * words
    end = len(packet)
    if b0 & 0x20:  # padding: the last byte counts the padding bytes, itself included
        pad = packet[-1]
        if not 0 < pad <= end - offset:
            raise ProtocolError(f"implausible RTP padding length {pad}")
        end -= pad
    if offset > end:
        raise ProtocolError("RTP header runs past the end of the packet")
    seq, timestamp, ssrc = struct.unpack_from(">HII", packet, 2)
    return _Packet(
        seq=seq,
        timestamp=timestamp,
        marker=bool(b1 & 0x80),
        payload_type=b1 & 0x7F,
        ssrc=ssrc,
        payload=bytes(packet[offset:end]),
    )


def parse_sender_reports(packet: bytes) -> list[SenderReport]:
    """Pull every Sender Report out of one (possibly compound) RTCP packet.

    RTCP arrives compound — an SR with an SDES glued behind it is the norm — so
    this walks the whole datagram by each sub-packet's own length field and
    ignores everything that isn't PT 200.
    """
    reports: list[SenderReport] = []
    offset = 0
    while offset + 4 <= len(packet):
        if packet[offset] >> 6 != 2:
            break  # not RTCP after all; don't guess at the rest
        payload_type = packet[offset + 1]
        (words,) = struct.unpack_from(">H", packet, offset + 2)
        length = 4 * (words + 1)
        if length <= 0 or offset + length > len(packet):
            break
        if payload_type == _RTCP_SR and length >= 28:
            ntp, rtp = struct.unpack_from(">QI", packet, offset + 8)
            reports.append(SenderReport(rtp_timestamp=rtp, ntp_unix=ntp_to_unix(ntp)))
        offset += length
    return reports


def ntp_to_unix(ntp: int) -> float:
    """Convert a 64-bit NTP timestamp (1900 epoch, 32.32 fixed point) to unix seconds."""
    return (ntp >> 32) - NTP_UNIX_DELTA + (ntp & 0xFFFFFFFF) / 2**32


# --------------------------------------------------------------------------- #
# The minimal RTSP client
#
# OPTIONS → DESCRIBE → SETUP → PLAY → (interleaved data) → TEARDOWN, with Basic
# and Digest auth. Only what the phone actually speaks; no UDP, no aggregate
# multi-track setup.
# --------------------------------------------------------------------------- #


class _Response(NamedTuple):
    """One RTSP response. Repeated headers are joined with ``\\n`` (the phone
    sends two ``WWW-Authenticate`` lines, Digest then Basic)."""

    status: int
    reason: str
    headers: dict[str, str]  # keys lowercased
    body: bytes


class _Media(NamedTuple):
    """The audio ``m=`` section of the SDP, reduced to what we need."""

    payload_type: int
    codec: str
    sample_rate: int
    channels: int
    control: str
    fmtp: str


def parse_sdp(sdp: str, base_url: str) -> _Media:
    """Find the audio media section and resolve its control URL.

    iRTSP puts video on ``trackID=0`` and audio on ``trackID=1`` (relative to a
    ``Content-Base`` of ``rtsp://host:port/live/``); nothing here assumes that,
    it reads whichever ``m=audio`` section the phone sent and resolves whatever
    ``a=control`` it carries. Session-level attributes — including the
    ``a=control:*`` above the first ``m=`` line — are deliberately not read as
    the audio track's.

    Every ``m=audio`` section is collected before one is chosen, rather than a
    single accumulator being reset on each new section. The reset version worked
    only while audio was the *last* media section: an ``m=video`` after it threw
    the captured audio away and the call failed with "no audio track", blaming
    the app for having audio disabled. Section order is not something a client
    may depend on, so it no longer decides the outcome.
    """
    sections: list[tuple[str | None, str, str]] = []  # (rtpmap, fmtp, control)
    in_audio = False
    rtpmap: str | None = None
    fmtp = ""
    control = ""
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("m="):
            if in_audio:
                sections.append((rtpmap, fmtp, control))
            in_audio = line[2:].startswith("audio")
            rtpmap, fmtp, control = None, "", ""
            continue
        if not in_audio or not line.startswith("a="):
            continue
        attribute = line[2:]
        if attribute.startswith("rtpmap:") and rtpmap is None:
            rtpmap = attribute[len("rtpmap:"):].strip()
        elif attribute.startswith("fmtp:"):
            fmtp = attribute[len("fmtp:"):].strip()
        elif attribute.startswith("control:"):
            control = attribute[len("control:"):].strip()
    if in_audio:
        sections.append((rtpmap, fmtp, control))

    # First audio section that carries an rtpmap. One without is unusable, so
    # skipping past it beats failing on it when a usable section follows.
    audio = next((section for section in sections if section[0] is not None), None)
    if audio is None:
        raise ProtocolError(
            "the phone's SDP has no audio track with an rtpmap — is audio enabled "
            "in the iRTSP app?"
        )
    rtpmap, fmtp, control = audio
    assert rtpmap is not None

    payload_text, _, encoding_text = rtpmap.partition(" ")
    parts = encoding_text.split("/")
    name = parts[0].strip().lower()
    try:
        payload_type = int(payload_text)
        sample_rate = int(parts[1])
    except (IndexError, ValueError) as e:
        raise ProtocolError(f"unparsable audio rtpmap {rtpmap!r}") from e
    # RFC 4566: the channel count is optional and defaults to 1.
    channels = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 1
    codec = _CODECS.get(name, name)
    if codec == "opus":
        # An Opus rtpmap always reads `opus/48000/2` whatever was captured
        # (RFC 7587 §7 — and the app hardcodes it); `sprop-stereo` is what
        # actually says whether the phone is sending two channels.
        channels = 2 if "sprop-stereo=1" in fmtp.replace(" ", "") else 1
    return _Media(
        payload_type=payload_type,
        codec=codec,
        sample_rate=sample_rate,
        channels=channels,
        control=resolve_url(base_url, control),
        fmtp=fmtp,
    )


def resolve_url(base: str, control: str) -> str:
    """Resolve an SDP ``a=control`` value against the session's base URL."""
    if not control or control == "*":
        return base
    if "://" in control:
        return control
    return base.rstrip("/") + "/" + control.lstrip("/")


def _digest_response(
    challenge: dict[str, str], method: str, url: str, username: str, password: str
) -> str:
    """Build a Digest ``Authorization`` value (MD5, with or without ``qop=auth``).

    Exactly what the app's ``DigestAuth.swift`` verifies: ``HA1 =
    MD5(user:realm:pass)``, ``HA2 = MD5(method:uri)``, and the response is the
    RFC 2069 ``MD5(HA1:nonce:HA2)``, because the phone's challenge carries no
    ``qop``. The ``qop=auth`` variant (nonce count + cnonce inside the digest)
    is here for any server that does ask for it — the app accepts that form too.
    """

    def md5(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    realm = challenge.get("realm", "")
    nonce = challenge.get("nonce", "")
    ha1 = md5(f"{username}:{realm}:{password}")
    ha2 = md5(f"{method}:{url}")
    fields = {
        "username": username,
        "realm": realm,
        "nonce": nonce,
        "uri": url,
    }
    qop = challenge.get("qop", "")
    qops = [q.strip() for q in qop.split(",") if q.strip()]
    if "auth" in qops:
        cnonce = os.urandom(8).hex()
        nc = "00000001"
        fields["response"] = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
        fields["qop"] = "auth"
        fields["nc"] = nc
        fields["cnonce"] = cnonce
    else:
        fields["response"] = md5(f"{ha1}:{nonce}:{ha2}")
    if "opaque" in challenge:
        fields["opaque"] = challenge["opaque"]
    unquoted = {"qop", "nc"}
    parts = [
        f"{k}={v}" if k in unquoted else f'{k}="{v}"' for k, v in fields.items()
    ]
    return "Digest " + ", ".join(parts)


def parse_challenge(header: str) -> tuple[str, dict[str, str]]:
    """Pick the strongest ``WWW-Authenticate`` challenge and split it up.

    The phone offers both schemes, on two header lines (Digest first, then
    Basic) — Digest wins, so a password never crosses the wire recoverably when
    the phone was willing to avoid it.
    """
    best: tuple[str, dict[str, str]] = ("", {})
    for line in header.split("\n"):
        scheme, _, rest = line.strip().partition(" ")
        params: dict[str, str] = {}
        for chunk in rest.split(","):
            key, sep, value = chunk.strip().partition("=")
            if sep:
                params[key.strip().lower()] = value.strip().strip('"')
        scheme = scheme.lower()
        if scheme == "digest":
            return scheme, params
        if scheme and not best[0]:
            best = (scheme, params)
    return best


class AudioStream:
    """A live audio-only RTSP session. Prefer :func:`irtsp.audio_stream`.

    Iterating the stream is what drives it: each :class:`AudioBlock` is assembled
    from packets read on demand, and the RTCP Sender Reports seen along the way
    accumulate in :attr:`sender_reports`. There is no background thread and no
    buffering behind your back — if you stop iterating, TCP back-pressure
    eventually reaches the phone.
    """

    def __init__(
        self,
        host: str,
        *,
        port: int = 8554,
        path: str = "live",
        username: str | None = None,
        password: str | None = None,
        timeout: float = 5.0,
        user_agent: str = "irtsp-python",
    ):
        if "://" in host:  # a full rtsp:// URL is accepted verbatim
            parts = urlsplit(host)
            username = username or parts.username
            password = password or parts.password
            port = parts.port or port
            path = (parts.path or "/live").lstrip("/")
            host = parts.hostname or host
        self.host = host
        self.port = port
        self.path = path.lstrip("/")
        self.url = f"rtsp://{self.host}:{self.port}/{self.path}"
        self.timeout = timeout
        self.user_agent = user_agent
        self._username = username
        self._password = password

        #: Every RTCP Sender Report received so far, oldest first.
        self.sender_reports: list[SenderReport] = []
        self.sdp: str = ""
        self.codec: str = ""
        self.sample_rate: int = 0
        self.channels: int = 0
        self.fmtp: str = ""
        self.control_url: str = ""
        self.session_id: str | None = None
        #: SSRC of the audio stream, once the first packet has been read.
        self.ssrc: int | None = None

        self._sock: socket.socket | None = None
        self._reader = None  # type: ignore[var-annotated]
        self._cseq = 0
        #: The last ``WWW-Authenticate`` challenge, as (scheme, params). Digest
        #: signs the method and URI, so the header is rebuilt per request.
        self._challenge: tuple[str, dict[str, str]] | None = None
        self._payload_type = AUDIO_PAYLOAD_TYPE
        self._rtp_channel = 0
        self._rtcp_channel = 1
        self._closed = False

        # Block assembly state. `_prev_ts` is unwrapped (64-bit); `_prev_raw_ts`
        # is the wire value it came from, and `_wraps` counts the rollovers.
        self._packets: list[bytes] = []
        self._block_ts = 0
        self._block_raw_ts = 0
        self._block_seq = 0
        self._block_gap = 0
        self._block_lost = 0
        self._block_marker = False
        self._prev_seq: int | None = None
        self._prev_ts: int | None = None
        self._prev_raw_ts: int | None = None
        self._prev_frames: int | None = None
        self._wraps = 0

    # ------------------------------------------------------------------ setup

    def open(self) -> "AudioStream":
        """Run the RTSP handshake through PLAY. :func:`audio_stream` calls this."""
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock = sock
            self._reader = sock.makefile("rb")

            # OPTIONS is never authenticated by the app, so this is purely a cheap
            # "is anything RTSP-shaped listening here?" before we ask for the SDP.
            self._request("OPTIONS", self.url)
            describe = self._request("DESCRIBE", self.url, {"Accept": "application/sdp"})
            self.sdp = describe.body.decode("utf-8", "replace")
            base = describe.headers.get("content-base") or self.url
            media = parse_sdp(self.sdp, base)
            self._payload_type = media.payload_type
            self.codec = media.codec
            self.sample_rate = media.sample_rate
            self.channels = media.channels
            self.fmtp = media.fmtp
            self.control_url = media.control

            setup = self._request(
                "SETUP",
                self.control_url,
                {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"},
            )
            self.session_id = _session_id(setup.headers.get("session", ""))
            self._rtp_channel, self._rtcp_channel = _interleaved(
                setup.headers.get("transport", "")
            )
            self._request("PLAY", base, {"Range": "npt=0.000-"})

            sock.settimeout(None)  # iteration blocks; close() unblocks it
        except BaseException:
            self.close()
            raise
        log.info(
            "audio: %s %s Hz x%d on %s (interleaved %d-%d)",
            self.codec, self.sample_rate, self.channels, self.url,
            self._rtp_channel, self._rtcp_channel,
        )
        return self

    def _request(
        self, method: str, url: str, headers: dict[str, str] | None = None
    ) -> _Response:
        """Send one RTSP request, transparently answering a 401 challenge."""
        response = self._send(method, url, headers)
        if response.status == 401 and self._username is not None:
            challenge = response.headers.get("www-authenticate")
            if challenge is None:
                raise ProtocolError("401 from the phone with no WWW-Authenticate header")
            self._challenge = parse_challenge(challenge)
            response = self._send(method, url, headers)
        if response.status != 200:
            hint = ""
            if response.status == 401:
                hint = (
                    " — wrong username/password?" if self._username
                    else " — this phone wants credentials (username=/password=)"
                )
            elif response.status == 503 and method == "DESCRIBE":
                # The phone withholds the SDP until its video encoder has produced
                # parameter sets, so "no audio yet" really means "not streaming yet".
                hint = " — is the phone actually streaming? (start the stream in the app)"
            raise ProtocolError(
                f"{method} {url} failed: {response.status} {response.reason}{hint}"
            )
        return response

    def _send(
        self, method: str, url: str, headers: dict[str, str] | None = None
    ) -> _Response:
        assert self._sock is not None
        self._cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {self._cseq}",
                 f"User-Agent: {self.user_agent}"]
        if self.session_id:
            lines.append(f"Session: {self.session_id}")
        authorization = self._authorization(method, url)
        if authorization:
            lines.append(f"Authorization: {authorization}")
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        self._sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
        return self._read_response()

    def _authorization(self, method: str, url: str) -> str | None:
        """The ``Authorization`` value for this request, per the last challenge."""
        if self._username is None:
            return None
        password = self._password or ""
        if self._challenge is None:
            return None  # nothing asked yet — don't leak credentials unprompted
        scheme, params = self._challenge
        if scheme == "basic":
            token = base64.b64encode(f"{self._username}:{password}".encode()).decode()
            return f"Basic {token}"
        if scheme == "digest":
            return _digest_response(params, method, url, self._username, password)
        raise ProtocolError(f"unsupported RTSP auth scheme {scheme!r}")

    def _read_response(self) -> _Response:
        """Read one RTSP response (status line, headers, optional body)."""
        status_line = self._readline()
        if not status_line.startswith("RTSP/"):
            raise ProtocolError(f"not an RTSP response: {status_line!r}")
        parts = status_line.split(" ", 2)
        try:
            status = int(parts[1])
        except (IndexError, ValueError) as e:
            raise ProtocolError(f"unparsable RTSP status line {status_line!r}") from e
        reason = parts[2] if len(parts) > 2 else ""
        headers: dict[str, str] = {}
        while True:
            line = self._readline()
            if not line:
                break
            key, _, value = line.partition(":")
            name, value = key.strip().lower(), value.strip()
            # Repeated headers accumulate — the phone's 401 carries one
            # WWW-Authenticate line per scheme it accepts.
            headers[name] = f"{headers[name]}\n{value}" if name in headers else value
        body = b""
        length = int(headers.get("content-length", 0) or 0)
        if length:
            body = self._read_exact(length)
        return _Response(status=status, reason=reason, headers=headers, body=body)

    def _readline(self) -> str:
        assert self._reader is not None
        line = self._reader.readline()
        if not line:
            raise ConnectionClosed("the phone closed the RTSP connection")
        return line.decode("utf-8", "replace").rstrip("\r\n")

    def _read_exact(self, n: int) -> bytes:
        assert self._reader is not None
        data = self._reader.read(n)
        if data is None or len(data) < n:
            raise ConnectionClosed("the phone closed the RTSP connection")
        return data

    # --------------------------------------------------------------- streaming

    def _read_interleaved(self) -> tuple[int, bytes]:
        """Read one ``$`` frame: channel + big-endian u16 length + payload.

        A late RTSP response (the phone answering a keepalive, say) can appear
        between frames; it is consumed and dropped rather than read as garbage.
        """
        while True:
            marker = self._read_exact(1)
            if marker == b"$":
                header = self._read_exact(3)
                (length,) = struct.unpack(">H", header[1:3])
                return header[0], self._read_exact(length)
            if marker != b"R":
                raise ProtocolError(
                    f"expected an interleaved frame, got byte {marker!r} — desynced"
                )
            # 'R' of "RTSP/1.0 ..." — consume the whole response (body included,
            # or the very next read would land mid-message) and carry on.
            self._readline()
            length = 0
            while True:
                line = self._readline()
                if not line:
                    break
                key, _, value = line.partition(":")
                if key.strip().lower() == "content-length":
                    length = int(value.strip() or 0)
            if length:
                self._read_exact(length)

    def __iter__(self) -> Iterator[AudioBlock]:
        """Yield contiguous :class:`AudioBlock` s until the stream ends."""
        while not self._closed:
            try:
                channel, payload = self._read_interleaved()
            except (ConnectionClosed, OSError):
                break
            if channel == self._rtcp_channel:
                for report in parse_sender_reports(payload):
                    self.sender_reports.append(
                        report._replace(rtp_timestamp=self._sr_on_axis(report.rtp_timestamp))
                    )
                continue
            if channel != self._rtp_channel:
                continue  # another track's data (video, if it were ever set up)
            packet = parse_rtp(payload)
            if packet.payload_type != self._payload_type:
                continue
            block = self._feed(packet)
            if block is not None:
                yield block
        tail = self._flush()
        if tail is not None:
            yield tail

    def _sr_on_axis(self, raw: int) -> int:
        """Put an SR's 32-bit RTP timestamp on the blocks' unwrapped axis.

        A Sender Report stamps roughly the last packet sent, so it belongs to
        whichever rollover epoch the audio is in — and an SR minted just either
        side of the boundary must not land a whole 32-bit space from the audio
        it describes.
        """
        candidate = self._wraps * 2**32 + raw
        if self._prev_ts is not None:
            if candidate - self._prev_ts > 2**31:
                candidate -= 2**32
            elif self._prev_ts - candidate > 2**31:
                candidate += 2**32
        return candidate

    def _unwrap(self, raw: int) -> int:
        """Lift a 32-bit wire timestamp onto the stream's monotonic 64-bit axis.

        A step that looks like more than half the 32-bit space is a rollover,
        not a jump — audio timestamps advance by a packet's worth of frames, so
        the two are never ambiguous in practice. Only the RTP timestamp is
        unwrapped this way; sequence numbers stay 16-bit modular, because
        counting *packets* is what they are for.
        """
        prev = self._prev_raw_ts
        if prev is not None:
            if raw < prev and prev - raw > 2**31:
                self._wraps += 1
            elif raw > prev and raw - prev > 2**31 and self._wraps:
                self._wraps -= 1  # a step backwards across the boundary
        return self._wraps * 2**32 + raw

    def _feed(self, packet: _Packet) -> AudioBlock | None:
        """Add one packet to the open block; return the block it just closed, if any."""
        closed: AudioBlock | None = None
        gap = lost = 0
        marker = False

        if self.ssrc is None:
            self.ssrc = packet.ssrc
        elif packet.ssrc != self.ssrc:
            # The phone restarted the stream: sequence and timestamp origins are
            # both new, so nothing about the seam is measurable. Say so, don't guess.
            log.warning("audio SSRC changed (%s → %s) — timeline restarted",
                        self.ssrc, packet.ssrc)
            self.ssrc = packet.ssrc
            closed = self._flush()
            self._wraps = 0
            self._prev_raw_ts = None

        timestamp = self._unwrap(packet.timestamp)

        if self._packets and closed is None:
            assert self._prev_seq is not None and self._prev_ts is not None
            lost = (packet.seq - self._prev_seq - 1) & 0xFFFF
            # Unwrapped, so a rollover here is a step of one packet like any
            # other — never a 2**32-sized phantom gap.
            step = timestamp - self._prev_ts
            # An AAC marker means end-of-access-unit, never a capture gap.
            marker = packet.marker and self.codec != "aac"
            expected = self._prev_frames
            stepped = expected is not None and step != expected
            if lost or marker or stepped:
                closed = self._flush()
                if lost:
                    # Loss also moves the timestamp, and no arithmetic can split
                    # a hole from a skipped send — so the frames are attributed
                    # to loss and gap_frames stays honest at 0.
                    gap = 0
                elif expected is not None:
                    gap = step - expected
                    if gap < 0:  # the timestamp went backwards
                        log.warning("audio RTP timestamp moved backwards at seq %d",
                                    packet.seq)
                        gap = 0

        if not self._packets:
            self._block_ts = timestamp
            self._block_raw_ts = packet.timestamp
            self._block_seq = packet.seq
            self._block_gap = gap
            self._block_lost = lost
            self._block_marker = marker

        self._packets.append(packet.payload)
        self._prev_seq = packet.seq
        self._prev_ts = timestamp
        self._prev_raw_ts = packet.timestamp
        self._prev_frames = _frames_in_payload(self.codec, packet.payload, self.channels)
        return closed

    def _flush(self) -> AudioBlock | None:
        """Close the open block (if any) and reset for the next one."""
        if not self._packets:
            return None
        block = AudioBlock(
            packets=tuple(self._packets),
            codec=self.codec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            rtp_timestamp=self._block_ts,
            rtp_timestamp_raw=self._block_raw_ts,
            seq_start=self._block_seq,
            n_packets=len(self._packets),
            gap_frames=self._block_gap,
            lost_packets=self._block_lost,
            marker=self._block_marker,
        )
        self._packets = []
        return block

    # ---------------------------------------------------------------- teardown

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """TEARDOWN the session and close the socket. Idempotent."""
        if self._closed:
            return
        self._closed = True
        sock, self._sock = self._sock, None
        if sock is not None and self.session_id:
            try:
                sock.settimeout(self.timeout)
                self._sock = sock  # _send needs it back for the one last write
                self._send("TEARDOWN", self.url)
            except (OSError, ProtocolError, ConnectionClosed, AssertionError):
                pass  # a phone that already hung up needs no goodbye
            finally:
                self._sock = None
        reader, self._reader = self._reader, None
        for closeable in (reader, sock):
            if closeable is not None:
                try:
                    closeable.close()
                except OSError:  # pragma: no cover
                    pass

    def __enter__(self) -> "AudioStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        state = "closed" if self._closed else "live"
        codec = f"{self.codec} {self.sample_rate}Hz x{self.channels}" if self.codec else "-"
        return f"<irtsp.AudioStream {self.url} {codec} [{state}]>"


def _session_id(header: str) -> str | None:
    """The session id out of a ``Session: 1234abcd;timeout=60`` header."""
    value = header.split(";")[0].strip()
    return value or None


def _interleaved(transport: str) -> tuple[int, int]:
    """The channel pair the server assigned, from its ``Transport`` response.

    The phone may hand back channels other than the ``0-1`` we asked for (it
    numbers per track), so the response wins — always.
    """
    for part in transport.split(";"):
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "interleaved" and value:
            first, _, second = value.partition("-")
            try:
                rtp = int(first)
                rtcp = int(second) if second else rtp + 1
            except ValueError:
                break
            if not (0 <= rtp <= 255 and 0 <= rtcp <= 255):
                break
            return rtp, rtcp
    log.debug("no usable interleaved channels in Transport %r; assuming 0-1", transport)
    return 0, 1


def audio_stream(
    host: str,
    *,
    port: int = 8554,
    path: str = "live",
    username: str | None = None,
    password: str | None = None,
    timeout: float = 5.0,
) -> AudioStream:
    """Open the phone's audio track and return a live :class:`AudioStream`.

    ``host`` is a hostname/IP (``"192.168.1.24"``, ``"ryans-iphone.local"``) or a
    whole ``rtsp://`` URL, in which case the port, path and any embedded
    credentials come from it.

    Blocks are yielded by iterating, and the stream is a context manager, so the
    shape mirrors :func:`irtsp.connect`::

        with irtsp.audio_stream("192.168.1.24") as audio:
            for block in audio:
                ...

    Args:
        path: RTSP path — the app serves ``/live`` by default.
        username / password: for a phone with Basic or Digest auth enabled.
        timeout: connect/handshake timeout in seconds; iteration itself blocks
            (call :meth:`AudioStream.close` from another thread to stop it).
    """
    return AudioStream(
        host,
        port=port,
        path=path,
        username=username,
        password=password,
        timeout=timeout,
    ).open()
