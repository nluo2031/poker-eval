"""Tests for tables.py. Run:  python3 test_tables.py"""

import random
from itertools import combinations

from eval5 import evaluate5, parse_cards
from tables import (
    CLASSES, FLUSH_TABLE_SIZE, NONFLUSH_TABLE_SIZE, NUM_CLASSES,
    build_flush_table, category_counts, class_key, count_vectors,
    eval7_fast, eval7_ref, nonflush_index, rank5,
)


def r7(s):
    return eval7_ref(parse_cards(s))


def test_classes():
    assert len(CLASSES) == NUM_CLASSES
    assert category_counts() == [10, 156, 156, 1277, 10, 858, 858, 2860, 1277]
    # The enumeration matches the set of values seen over all 5-card hands
    # (checked exhaustively in test_eval5), here spot-checked at the ends.
    assert rank5(parse_cards("7c 5d 4h 3s 2c")) == 1
    assert rank5(parse_cards("As Ks Qs Js Ts")) == NUM_CLASSES
    assert rank5(parse_cards("5h 4h 3h 2h Ah")) == NUM_CLASSES - 9
    keys = [class_key(v) for v in CLASSES]
    assert keys == sorted(keys) and len(set(keys)) == NUM_CLASSES


def test_eval7_ref():
    assert r7("As Ks Qs Js Ts 2c 3d") == NUM_CLASSES
    # board plays: wheel on board, 6 in hand makes 6-high straight
    assert r7("6d Kc Ah 2s 3c 4d 5h") == rank5(parse_cards("6d 2s 3c 4d 5h"))
    # two trips -> full house with the higher trips
    assert r7("9c 9d 9h 5c 5d 5h Ks") == rank5(parse_cards("9c 9d 9h 5c 5d"))
    # three pairs -> best two pair + best kicker
    assert r7("Ac Ad Kc Kd Qc Qd 2s") == rank5(parse_cards("Ac Ad Kc Kd Qc"))


def test_flush_table():
    t = build_flush_table()
    assert len(t) == FLUSH_TABLE_SIZE
    for mask in range(FLUSH_TABLE_SIZE):
        assert (t[mask] > 0) == (mask.bit_count() >= 5)
    # A K Q J T 9 8 of one suit -> royal flush
    assert t[0b1111111000000] == NUM_CLASSES
    # A 2 3 4 5 + K Q -> steel wheel beats the K-high flush
    assert t[(1 << 12) | 0b1111 | (1 << 11) | (1 << 10)] == NUM_CLASSES - 9


def test_nonflush_bijection():
    vecs = list(count_vectors())
    assert len(vecs) == NONFLUSH_TABLE_SIZE
    idx = sorted(nonflush_index(v) for v in vecs)
    assert idx == list(range(NONFLUSH_TABLE_SIZE))


def test_fast_vs_ref_random():
    rng = random.Random(1)
    for _ in range(20000):
        hand = rng.sample(range(52), 7)
        assert eval7_fast(hand) == eval7_ref(hand), hand


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s)")
    print("all passed")
