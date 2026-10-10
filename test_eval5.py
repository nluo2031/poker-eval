"""Tests for eval5. Run:  python3 test_eval5.py   (exhaustive check ~10-30s)"""

from collections import Counter
from itertools import combinations

from eval5 import (
    CATEGORY_NAMES, FLUSH, FULL_HOUSE, HIGH_CARD, ONE_PAIR, QUADS,
    STRAIGHT, STRAIGHT_FLUSH, TRIPS, TWO_PAIR,
    best_of, card_str, evaluate5, parse_card, parse_cards,
)


def ev(s):
    return evaluate5(parse_cards(s))


def test_encoding():
    assert parse_card("2c") == 0
    assert parse_card("2s") == 3
    assert parse_card("3c") == 4
    assert parse_card("As") == 51
    for c in range(52):
        assert parse_card(card_str(c)) == c


def test_categories():
    assert ev("As Ks Qs Js Ts")[0] == STRAIGHT_FLUSH
    assert ev("5h 4h 3h 2h Ah") == (STRAIGHT_FLUSH, 3)   # steel wheel
    assert ev("9c 9d 9h 9s 2c")[0] == QUADS
    assert ev("9c 9d 9h 2s 2c")[0] == FULL_HOUSE
    assert ev("Ac 9c 7c 4c 2c")[0] == FLUSH
    assert ev("Ac Kd Qh Js Tc")[0] == STRAIGHT
    assert ev("Ac 2d 3h 4s 5c") == (STRAIGHT, 3)        # wheel
    assert ev("9c 9d 9h Ks 2c")[0] == TRIPS
    assert ev("9c 9d Kh Ks 2c")[0] == TWO_PAIR
    assert ev("9c 9d Kh 3s 2c")[0] == ONE_PAIR
    assert ev("Ac Jd 9h 4s 2c")[0] == HIGH_CARD
    # Not straights: no wrap-around, no 4-card runs
    assert ev("Qc Kd Ah 2s 3c")[0] == HIGH_CARD
    assert ev("Ac Kd Qh Js 9c")[0] == HIGH_CARD


def test_ordering():
    better = [  # each hand strictly beats the next
        "As Ks Qs Js Ts",
        "9s 8s 7s 6s 5s",
        "5h 4h 3h 2h Ah",   # steel wheel: lowest straight flush
        "Ac Ad Ah As Kc",
        "Ac Ad Ah As 2c",
        "2c 2d 2h 2s Ac",
        "Ac Ad Ah 2s 2c",
        "Kc Kd Kh As Ac",
        "Ac Kc Qc Jc 9c",
        "Ac Kc Qc Jc 8c",
        "Ac Kd Qh Js Tc",
        "6c 5d 4h 3s 2c",
        "5c 4d 3h 2s Ac",   # wheel: lowest straight
        "Ac Ad Ah Ks Qc",
        "Ac Ad Ah Ks Jc",
        "Ac Ad Kh Ks 2c",
        "Ac Ad Qh Qs Kc",
        "Ac Ad Qh Qs Jc",
        "Ac Ad Kh Qs Jc",
        "Ac Ad Kh Qs Tc",
        "Kc Kd Ah Qs Jc",
        "Ac Kd Qh Js 9c",
        "Ac Kd Qh Js 8c",
        "7c 5d 4h 3s 2c",   # worst possible hand
    ]
    vals = [ev(h) for h in better]
    for a, b, ha, hb in zip(vals, vals[1:], better, better[1:]):
        assert a > b, f"{ha} should beat {hb}: {a} vs {b}"


def test_ties():
    assert ev("Ac Kd Qh Js 9c") == ev("Ad Kh Qs Jc 9d")       # suits irrelevant
    assert ev("5c 4d 3h 2s Ac") == ev("5h 4h 3c 2d Ad")


def test_best_of_7():
    # Board makes a wheel; hole cards make a 6-high straight.
    assert best_of(parse_cards("6d Kc  Ah 2s 3c 4d 5h")) == (STRAIGHT, 4)
    # Flush on board beats nothing in hand; board plays.
    assert best_of(parse_cards("2d 3d  Ah Kh 9h 7h 4h"))[0] == FLUSH


def test_exhaustive():
    """All C(52,5) = 2,598,960 hands: known category frequencies and
    the known count of 7,462 distinct hand values."""
    expected = {
        STRAIGHT_FLUSH: 40, QUADS: 624, FULL_HOUSE: 3744, FLUSH: 5108,
        STRAIGHT: 10200, TRIPS: 54912, TWO_PAIR: 123552,
        ONE_PAIR: 1098240, HIGH_CARD: 1302540,
    }
    expected_classes = {
        STRAIGHT_FLUSH: 10, QUADS: 156, FULL_HOUSE: 156, FLUSH: 1277,
        STRAIGHT: 10, TRIPS: 858, TWO_PAIR: 858, ONE_PAIR: 2860,
        HIGH_CARD: 1277,
    }
    counts = Counter()
    distinct = set()
    for hand in combinations(range(52), 5):
        v = evaluate5(hand)
        counts[v[0]] += 1
        distinct.add(v)
    assert sum(counts.values()) == 2598960
    for cat, n in expected.items():
        assert counts[cat] == n, f"{CATEGORY_NAMES[cat]}: {counts[cat]} != {n}"
    classes = Counter(v[0] for v in distinct)
    for cat, n in expected_classes.items():
        assert classes[cat] == n, f"{CATEGORY_NAMES[cat]} classes: {classes[cat]} != {n}"
    assert len(distinct) == 7462


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s)")
    print("all passed")
