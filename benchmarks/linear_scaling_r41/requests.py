"""Deterministic, immutable request descriptions; no Torch/device side effects."""
import math
from types import MappingProxyType


def build_request_bank(dense, offsets, indices, *, vocabulary=1024, size=8):
    if type(vocabulary) is not int or vocabulary <= 0 or type(size) is not int or not 1 <= size <= 64:
        raise ValueError('positive vocabulary and bank size1..64 required')
    rows = tuple(tuple(float(value) for value in row) for row in dense)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('nonempty rectangular dense batch required')
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError('finite dense values required')
    off = tuple(tuple(row) for row in offsets)
    ids = tuple(tuple(row) for row in indices)
    if not off or len(off) != len(ids):
        raise ValueError('matching nonempty sparse table lists required')
    for starts, values in zip(off, ids):
        if (len(starts) != len(rows) or starts[0] != 0 or
                any(type(v) is not int or v < 0 for v in starts) or
                any(a > b for a, b in zip(starts, starts[1:])) or starts[-1] > len(values)):
            raise ValueError('valid ragged batch offsets required')
        if any(type(v) is not int or not 0 <= v < vocabulary for v in values):
            raise ValueError('sparse ID out of vocabulary')
    bank = []
    for index in range(size):
        shift = index % len(rows)
        rotated = rows[-shift:] + rows[:-shift] if shift else rows
        values = tuple(tuple((v + 17 * index) % vocabulary for v in row) for row in ids)
        bank.append(MappingProxyType(dict(dense=rotated, offsets=off, indices=values)))
    return tuple(bank)
