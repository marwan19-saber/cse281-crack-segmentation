"""Lossless compression of pipeline outputs (MEM3, Bonus 4).

RLE for binary crack masks: the mask is read row by row as one long sequence of runs.
    bytes = height, width (uint32 each) | first value (1 byte) | run lengths as LEB128 varints
Huffman for gray images: canonical prefix codes built from the 256-level histogram.
    bytes = height, width (uint32) | 256 code lengths (1 byte each) | packed code bits
Both decoders reproduce the input exactly (checked with np.array_equal in the tests).
"""
import heapq
import struct

import numpy as np


# ---------------------------------------------------------------- run-length encoding
def rle_runs(mask):
    """(value, length) runs of the row-major flattened bool mask."""
    flat = np.asarray(mask, bool).ravel()
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    return [(int(flat[s]), int(n)) for s, n in zip(starts, lengths)]


def _varint(n):
    """LEB128: 7 bits per byte, high bit set while more bytes follow."""
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def rle_encode(mask):
    """Encode a bool mask; see the module docstring for the byte layout."""
    h, w = np.asarray(mask).shape
    runs = rle_runs(mask)
    first = runs[0][0] if runs else 0
    return struct.pack("<IIB", h, w, first) + b"".join(_varint(n) for _, n in runs)


def rle_decode(blob):
    """Inverse of rle_encode."""
    h, w, value = struct.unpack_from("<IIB", blob)
    lengths, n, shift = [], 0, 0
    for byte in blob[9:]:
        n |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            lengths.append(n)
            n, shift = 0, 0
    values = (np.arange(len(lengths)) + value) % 2          # runs alternate 0/1
    return np.repeat(values.astype(bool), lengths).reshape(h, w)


# ---------------------------------------------------------------- Huffman coding
def huffman_code_lengths(counts):
    """Optimal code length per symbol with non-zero count (merge the two rarest)."""
    heap = [(int(c), i, (s,)) for i, (s, c) in enumerate(
        (s, c) for s, c in enumerate(counts) if c > 0)]
    heapq.heapify(heap)
    lengths = {syms[0]: 0 for _, _, syms in heap}
    if len(heap) == 1:
        return {heap[0][2][0]: 1}
    tie = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, tie, s1 + s2))
        tie += 1
    return lengths


def _canonical_codes(lengths):
    """Canonical codes: sort by (length, symbol), count upward, shift left on length change."""
    codes, code, prev = {}, 0, 0
    for sym, n in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= n - prev
        codes[sym] = (code, n)
        code, prev = code + 1, n
    return codes


def huffman_encode(img):
    """Encode a uint8 image; see the module docstring for the byte layout."""
    img = np.asarray(img, np.uint8)
    lengths = huffman_code_lengths(np.bincount(img.ravel(), minlength=256))
    codes = _canonical_codes(lengths)
    code_of = np.zeros(256, np.int64)
    len_of = np.zeros(256, np.int64)
    for sym, (code, n) in codes.items():
        code_of[sym], len_of[sym] = code, n
    c, n = code_of[img.ravel()], len_of[img.ravel()]
    width = int(n.max())
    j = np.arange(width)
    bits = (c[:, None] >> np.maximum(n[:, None] - 1 - j, 0)) & 1    # MSB first
    bits = bits[j < n[:, None]].astype(np.uint8)
    header = struct.pack("<II", *img.shape) + bytes(len_of.astype(np.uint8))
    return header + np.packbits(bits).tobytes()


def huffman_decode(blob):
    """Inverse of huffman_encode (bit-by-bit walk of the canonical code table)."""
    h, w = struct.unpack_from("<II", blob)
    len_of = np.frombuffer(blob, np.uint8, 256, 8)
    codes = _canonical_codes({s: int(n) for s, n in enumerate(len_of) if n})
    table = {(n, code): sym for sym, (code, n) in codes.items()}
    bits = np.unpackbits(np.frombuffer(blob, np.uint8, offset=264))
    out = np.empty(h * w, np.uint8)
    k = code = n = 0
    for bit in bits.tolist():
        if k == out.size:
            break
        code, n = (code << 1) | bit, n + 1
        sym = table.get((n, code))
        if sym is not None:
            out[k], k, code, n = sym, k + 1, 0, 0
    return out.reshape(h, w)
