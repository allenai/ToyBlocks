# ruff: noqa  # verbatim port from TAMP tmp3 (kept diffable against upstream)
"""Lightweight mp4 validation shared by the encoder (validate-before-move), a post-write dataset
check, and the S3 upload gate. Deliberately dependency-light (stdlib + optional ffprobe) so it can be
imported by the isaaclab-free encoder subprocess AND run standalone.

Two levels:
  * has_moov(path)          : structural — atom-walk, no subprocess. A killed-mid-write file has
                              ftyp+mdat but NO moov -> unplayable ("moov atom not found"). Fast.
  * validate_mp4(path, ...) : has_moov + (optional) exact decoded frame count via ffprobe, and an
                              optional check that the count == the episode's expected length.
"""
import os
import struct
import subprocess
import time


def _scan_moov(path: str):
    """True / False / None (undetermined). Split out of has_moov so a transient I/O error can be
    distinguished from a genuinely moov-less file."""
    try:
        with open(path, "rb") as f:
            # Size from the OPEN handle, not a separate stat. os.path.getsize() before the open is
            # a second round trip that can report a stale/short length on a network filesystem,
            # which then makes the bounds test below reject a file whose moov is physically there.
            size = os.fstat(f.fileno()).st_size
            off = 0
            for _ in range(64):
                if off + 8 > size:
                    return False
                f.seek(off)
                hdr = f.read(16)
                if len(hdr) < 8:
                    return False
                asize = struct.unpack(">I", hdr[:4])[0]
                atype = hdr[4:8]
                hs = 8
                if asize == 1:
                    asize = struct.unpack(">Q", hdr[8:16])[0]; hs = 16
                elif asize == 0:
                    asize = size - off
                if atype == b"moov":
                    return True
                if asize < hs or off + asize > size:
                    return False  # truncated atom before any moov
                off += asize
        return False
    except OSError:
        return None                      # could not read it -- NOT evidence of a bad file


def has_moov(path: str, tries: int = 3, retry_delay: float = 0.5) -> bool:
    """True iff the mp4 contains a top-level `moov` atom reachable by walking atom sizes.

    Retries before condemning. This function decides whether a just-encoded video is publishable,
    and a False here USED TO DESTROY THE ONLY COPY: the caller skipped the publish and the staging
    file was deleted at finalize. A single stale stat or a transient read over weka was therefore
    enough to lose a perfectly good render -- 35 went that way on 2026-08-24. Negatives are now
    re-checked with a fresh open (and so a fresh size) before they are believed, and an unreadable
    file is retried rather than reported as invalid.
    """
    verdict = False
    for attempt in range(max(1, tries)):
        verdict = _scan_moov(path)
        if verdict is True:
            return True
        if attempt + 1 < tries:
            time.sleep(retry_delay * (attempt + 1))
    return False                          # False or None after every retry


def count_frames(path: str) -> int:
    """Exact decoded frame count via ffprobe (-count_frames). -1 on failure."""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=120)
        return int(p.stdout.strip())
    except Exception:
        return -1


def validate_mp4(path: str, expected_frames: int = None, check_frames: bool = False) -> dict:
    """Return {ok, reason, frames}. ok=False if no moov, unreadable, empty, or (when checked) the
    decoded frame count != expected_frames."""
    if not os.path.exists(path):
        return {"ok": False, "reason": "missing", "frames": -1}
    if os.path.getsize(path) == 0:
        return {"ok": False, "reason": "empty", "frames": -1}
    if not has_moov(path):
        return {"ok": False, "reason": "no-moov", "frames": -1}
    frames = count_frames(path) if (check_frames or expected_frames is not None) else None
    if frames is not None and frames <= 0:
        return {"ok": False, "reason": "undecodable", "frames": frames}
    if expected_frames is not None and frames != expected_frames:
        return {"ok": False, "reason": f"frame-mismatch:{frames}!={expected_frames}", "frames": frames}
    return {"ok": True, "reason": "ok", "frames": frames if frames is not None else -1}
