"""Msgpack serialisation with first-class NumPy array support.

This is a self-contained copy of the helpers used by the openpi wire protocol
(``openpi_client.msgpack_numpy``).  Having our own copy means the blocksuite
serving stack does not depend on the ``openpi`` / ``openpi_client`` packages.

The encoding is fully compatible: a message packed here can be unpacked by the
openpi client and vice-versa.

Wire format for ndarray values::

    {b"__ndarray__": True, b"data": <bytes>, b"dtype": "<f4", b"shape": (H, W, C)}
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(obj: object) -> object:
    """Msgpack *default* hook — serialise ndarray / np scalar to a tagged dict."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: dict) -> object:
    """Msgpack *object_hook* — reconstruct ndarray / np scalar from tagged dict."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
