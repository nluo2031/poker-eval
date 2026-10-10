"""
Steps 3-6: hand classes, 7-card reference oracle, and the two lookup tables
for the fast evaluator. Everything here is built from eval5.evaluate5.

Hand ranks are 1..7462, HIGHER IS BETTER:
    1    = 7-5-4-3-2 offsuit (worst)
    7462 = royal flush       (best)

Run `python3 tables.py` to write eval7_tables.h (for the C evaluator) and
ref5.bin (Python reference results for all 5-card hands, used to check the
C port of the reference evaluator).
"""

from array import array
from collections import Counter
from itertools import combinations, combinations_with_replacement, product

from eval5 import (
    CATEGORY_NAMES, STRAIGHT_FLUSH, evaluate5, make_card, rank_of, suit_of,
)

NUM_CLASSES = 7462
FLUSH_TABLE_SIZE = 1 << 13
NONFLUSH_TABLE_SIZE = 49205


# --------------------------------------------------------------------------
# Step 3: enumerate the 7,462 classes
# --------------------------------------------------------------------------

def _offsuit(ranks) -> list[int]:
    """Cards with the given ranks and no flush. In sorted order, card k gets
    suit k % 4: copies of one rank are consecutive (at most 4) so their suits
    differ, and no suit is used more than ceil(len/4) <= 2 times."""
    return [make_card(r, k % 4) for k, r in enumerate(sorted(ranks))]


def _suited(ranks) -> list[int]:
    return [make_card(r, 0) for r in ranks]


def enumerate_classes() -> list[tuple]:
    """Every distinct evaluate5 value, worst first. Index i has rank i + 1.

    A class is fixed by its rank multiset plus whether it's suited:
    6175 multisets of 5 ranks (each rank at most 4 times) evaluated offsuit,
    plus the 1287 sets of 5 distinct ranks evaluated as a flush."""
    values = set()
    for ranks in combinations_with_replacement(range(13), 5):
        if max(Counter(ranks).values()) > 4:
            continue
        values.add(evaluate5(_offsuit(ranks)))
        if len(set(ranks)) == 5:
            values.add(evaluate5(_suited(ranks)))
    return sorted(values)


CLASSES = enumerate_classes()
RANK_OF = {v: i + 1 for i, v in enumerate(CLASSES)}   # evaluate5 value -> 1..7462


def rank5(cards) -> int:
    return RANK_OF[evaluate5(cards)]


def class_of(rank: int) -> tuple:
    return CLASSES[rank - 1]


def category_counts() -> list[int]:
    """Number of classes per category, straight flush down to high card."""
    c = Counter(v[0] for v in CLASSES)
    return [c[cat] for cat in range(STRAIGHT_FLUSH, -1, -1)]


def class_key(value: tuple) -> int:
    """Pack an evaluate5 value into an int with the same ordering:
    category * 13^5 + tiebreak ranks as base-13 digits (left-aligned).
    Within a category every tuple has the same length, so this is
    order-preserving. The C oracle uses these to map its result to a rank."""
    key = value[0] * 13**5
    for k, r in enumerate(value[1:]):
        key += r * 13**(4 - k)
    return key


# --------------------------------------------------------------------------
# Step 4: 7-card reference (the oracle)
# --------------------------------------------------------------------------

def eval7_ref(cards) -> int:
    """Rank 1..7462 of the best 5 of 7 cards: max over all 21 subsets."""
    if len(cards) != 7 or len(set(cards)) != 7:
        raise ValueError(f"need 7 distinct cards, got {cards!r}")
    return RANK_OF[max(evaluate5(c) for c in combinations(cards, 5))]


# --------------------------------------------------------------------------
# Step 5: flush table, indexed by the 13-bit rank mask of the flush suit
# --------------------------------------------------------------------------

def build_flush_table() -> list[int]:
    """table[mask] = best flush / straight flush rank using the ranks in
    mask, for every mask with >= 5 bits; 0 elsewhere. Brute force over the
    5-subsets with the reference evaluator (329,472 evaluations)."""
    table = [0] * FLUSH_TABLE_SIZE
    for mask in range(FLUSH_TABLE_SIZE):
        ranks = [r for r in range(13) if mask >> r & 1]
        if len(ranks) >= 5:
            table[mask] = max(rank5(_suited(s)) for s in combinations(ranks, 5))
    return table


# --------------------------------------------------------------------------
# Step 6: non-flush table, indexed by a perfect hash of rank counts
# --------------------------------------------------------------------------
#
# A 7-card hand's ranks are a count vector n = (n_0, ..., n_12) with
# 0 <= n_i <= 4 and sum 7. There are exactly 49,205 of them. The index is
# the vector's position in lexicographic order:
#
#   N(k, s) = number of length-k vectors with entries 0..4 summing to s
#   R_i     = 7 - (n_0 + ... + n_{i-1})          cards left for ranks i..12
#   index   = sum over i of  sum_{v < n_i} N(12 - i, R_i - v)
#
# i.e. at each rank, count the vectors that agree so far but put fewer
# cards here. Ranks with n_i = 0 add nothing, so the fast evaluator only
# visits ranks actually present (<= 7 of them), low to high, doing
#   index += COEF[i][R][n_i];  R -= n_i
# COEF is a 13 x 8 x 5 table of those inner sums.

def _n_table():
    """N[k][s] for k = 0..13, s = 0..7."""
    N = [[0] * 8 for _ in range(14)]
    N[0][0] = 1
    for k in range(1, 14):
        for s in range(8):
            N[k][s] = sum(N[k - 1][s - v] for v in range(5) if v <= s)
    return N


_N = _n_table()
assert _N[13][7] == NONFLUSH_TABLE_SIZE

COEF = [[[sum(_N[12 - i][R - v] for v in range(n) if v <= R)
          for n in range(5)]
         for R in range(8)]
        for i in range(13)]


def nonflush_index(counts) -> int:
    idx, R = 0, 7
    for i, n in enumerate(counts):
        idx += COEF[i][R][n]
        R -= n
    return idx


def count_vectors():
    """All 13-entry vectors with entries 0..4 summing to 7."""
    def rec(i, left):
        if i == 12:
            if left <= 4:
                yield (left,)
            return
        for n in range(min(4, left) + 1):
            for rest in rec(i + 1, left - n):
                yield (n, *rest)
    return rec(0, 7)


def build_nonflush_table() -> list[int]:
    """table[index(counts)] = rank of the best non-flush hand with those
    rank counts, filled with the 7-card oracle. Also checks the indexing
    is a bijection onto [0, 49205)."""
    table = [0] * NONFLUSH_TABLE_SIZE
    seen = 0
    for counts in count_vectors():
        idx = nonflush_index(counts)
        if table[idx]:
            raise AssertionError(f"index collision at {idx}: {counts}")
        ranks = [r for r, n in enumerate(counts) for _ in range(n)]
        table[idx] = eval7_ref(_offsuit(ranks))
        seen += 1
    if seen != NONFLUSH_TABLE_SIZE or 0 in table:
        raise AssertionError("count-vector index is not a bijection")
    return table


# --------------------------------------------------------------------------
# Step 7 (Python mirror of the C evaluator; for testing the scheme)
# --------------------------------------------------------------------------

_FLUSH = None
_NONFLUSH = None


def _load():
    global _FLUSH, _NONFLUSH
    if _FLUSH is None:
        _FLUSH = build_flush_table()
        _NONFLUSH = build_nonflush_table()


def eval7_fast(cards) -> int:
    _load()
    suit_masks = [0, 0, 0, 0]
    counts = [0] * 13
    for c in cards:
        suit_masks[suit_of(c)] |= 1 << rank_of(c)
        counts[rank_of(c)] += 1
    for m in suit_masks:
        if m.bit_count() >= 5:
            return _FLUSH[m]
    return _NONFLUSH[nonflush_index(counts)]


# --------------------------------------------------------------------------
# Output for C
# --------------------------------------------------------------------------

def _c_array(ctype, name, values, per_line=16):
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        lines.append("  " + ",".join(str(v) for v in values[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_c_header(path, flush, nonflush):
    cat_first = [0] * 10   # cat_first[c] = lowest rank in category c
    for i in range(NUM_CLASSES - 1, -1, -1):
        cat_first[CLASSES[i][0]] = i + 1
    cat_first[9] = NUM_CLASSES + 1
    coef_flat = [COEF[i][R][n] for i in range(13) for R in range(8) for n in range(5)]
    assert max(coef_flat) < 65536
    parts = [
        "/* Generated by tables.py -- do not edit. Ranks: 1 = worst, 7462 = royal flush. */",
        "#ifndef EVAL7_TABLES_H",
        "#define EVAL7_TABLES_H",
        "#include <stdint.h>",
        f"#define NUM_CLASSES {NUM_CLASSES}",
        _c_array("uint16_t", "FLUSH_TABLE", flush),
        _c_array("uint16_t", "NONFLUSH_TABLE", nonflush),
        "/* COEF[(rank*8 + cards_left)*5 + count] */",
        _c_array("uint16_t", "COEF", coef_flat),
        "/* first rank of each category (index 9 = sentinel) */",
        _c_array("uint16_t", "CAT_FIRST", cat_first),
        "/* class_key() of each class, ascending; CLASS_KEYS[r-1] is rank r */",
        _c_array("uint32_t", "CLASS_KEYS", [class_key(v) for v in CLASSES], 8),
        "#endif",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(parts))


def write_ref5_bin(path):
    """uint16 rank of every 5-card hand, in itertools.combinations order."""
    out = array("H", (rank5(h) for h in combinations(range(52), 5)))
    with open(path, "wb") as f:
        out.tofile(f)


if __name__ == "__main__":
    import time
    t = time.time()
    print("classes per category (SF..HC):", category_counts())
    flush = build_flush_table()
    nonflush = build_nonflush_table()
    write_c_header("eval7_tables.h", flush, nonflush)
    print(f"wrote eval7_tables.h ({time.time() - t:.1f}s)")
    write_ref5_bin("ref5.bin")
    print(f"wrote ref5.bin ({time.time() - t:.1f}s)")
