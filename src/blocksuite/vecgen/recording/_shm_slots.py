"""Shared-memory slots that carry camera frames from the recorder to an encoder subprocess.

Sending a frame's images through a ``multiprocessing.Queue`` pickles them in the recorder process:
a feeder thread copies every byte into the pipe while holding the GIL (measured: 29% of the
GIL-held time of a 64-env eval, competing with the simulation thread). Instead the recorder copies
a frame's images into a free fixed-size slot of a shared-memory arena (``np.copyto``, which
releases the GIL) and queues only a small descriptor; the encoder reads the images in place and
frees the slot once they are encoded.

Arena layout: ``n_slots`` one-byte flags (1 = free; the recorder clears a flag when it fills the
slot, the encoder sets it when done), padded to 64 bytes, then ``n_slots`` slots of
``slot_bytes``. A frame whose images do not fit a slot, or that finds no free slot, takes the
regular pickled path, so the arena only ever removes work.

Stdlib + numpy only: the encoder subprocess imports this.
"""

from __future__ import annotations

import contextlib
import os
from multiprocessing import shared_memory
from typing import Any

import numpy as np

SHM_KEY = "__shm__"
_ALIGN = 64


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def frame_image_bytes(images: dict[str, np.ndarray]) -> int:
    """Slot bytes the images of one frame need."""
    return sum(_aligned(a.nbytes) for a in images.values())


def shm_free_bytes() -> int:
    """Free bytes of the shared-memory filesystem (containers often cap it, e.g. Docker's 64 MB)."""
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_bavail * st.f_frsize)
    except OSError:
        return 0


class SlotArena:
    """Recorder side: owns the arena, fills slots, and hands out descriptors."""

    def __init__(self, slot_bytes: int, n_slots: int) -> None:
        self.slot_bytes = int(slot_bytes)
        self.n_slots = int(n_slots)
        self._header = _aligned(self.n_slots)
        size = self._header + self.n_slots * self.slot_bytes
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        # Reserve the pages now: /dev/shm files are sparse, so an arena larger than the space
        # left (other writers and eval processes size theirs from the same free count) is
        # created fine and SIGBUS-kills the recorder on a later write. Reserved, it fails here
        # with ENOSPC and the caller sends frames through the queue.
        try:
            os.posix_fallocate(self.shm._fd, 0, size)
        except OSError:
            self.shm.close()
            self.shm.unlink()
            raise
        self.flags = np.ndarray((self.n_slots,), dtype=np.uint8, buffer=self.shm.buf)
        self.flags[:] = 1
        self._next = 0

    @property
    def name(self) -> str:
        return self.shm.name

    def put(self, images: dict[str, np.ndarray]) -> dict[str, Any] | None:
        """Copy ``images`` into a free slot; returns its descriptor, or None (no slot / too big)."""
        if frame_image_bytes(images) > self.slot_bytes:
            return None
        # Scan a snapshot: the encoder process frees slots concurrently, and np.flatnonzero on
        # the live buffer raises if the count changes mid-scan. Only this side clears flags, so
        # a stale snapshot can at worst miss a just-freed slot, never offer a taken one.
        free = np.flatnonzero(self.flags.copy())
        if free.size == 0:
            return None
        # Round-robin from the last slot taken, so a slot is not reused the moment it frees.
        after = free[free >= self._next]
        slot = int(after[0] if after.size else free[0])
        self._next = slot + 1
        self.flags[slot] = 0
        base = self._header + slot * self.slot_bytes
        layout: list[tuple[str, int, tuple[int, ...], str]] = []
        offset = base
        for key, arr in images.items():
            arr = np.ascontiguousarray(arr)
            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf, offset=offset)
            np.copyto(view, arr)
            layout.append((key, offset, tuple(arr.shape), arr.dtype.str))
            offset += _aligned(arr.nbytes)
        return {SHM_KEY: (slot, layout)}

    def close(self) -> None:
        self.flags = None  # drop the buffer export before closing
        try:
            self.shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()


class SlotReader:
    """Encoder side: attaches to an arena by name, views slots, frees them."""

    def __init__(self, name: str, n_slots: int) -> None:
        # The recorder owns (and unlinks) the segment. Encoders share the recorder's resource
        # tracker, where attaching re-registers an already-registered name (a no-op); they must
        # not unregister it, or the recorder's own unlink later trips the tracker.
        try:
            self.shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:  # Python < 3.13 has no `track`
            self.shm = shared_memory.SharedMemory(name=name)
        self.flags = np.ndarray((n_slots,), dtype=np.uint8, buffer=self.shm.buf)

    def views(self, descriptor: dict[str, Any]) -> tuple[int, dict[str, np.ndarray]]:
        slot, layout = descriptor[SHM_KEY]
        return slot, {
            key: np.ndarray(shape, dtype=np.dtype(dt), buffer=self.shm.buf, offset=offset)
            for key, offset, shape, dt in layout
        }

    def free(self, slot: int) -> None:
        self.flags[slot] = 1
