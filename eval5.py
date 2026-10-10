"""
Texas Hold'em hand evaluation: card encoding + slow, obviously-correct
5-card reference evaluator (ground truth for faster evaluators later).

Card encoding
-------------
    card = 4 * rank + suit          (an int in 0..51)

    rank: 0..12  ->  2 3 4 5 6 7 8 9 T J Q K A   (deuce = 0, ace = 12)
    suit: 0..3   ->  c d h s

    rank = card // 4   (or card >> 2)
    suit = card % 4    (or card & 3)

So 0 = 2c, 1 = 2d, 2 = 2h, 3 = 2s, 4 = 3c, ..., 51 = As.
Sorting cards by int sorts them by rank. Rank and suit come out with a
shift and a mask.
"""

from collections import Counter
from itertools import combinations

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

ACE = 12
FIVE = 3  # rank index of the 5, the top card of the wheel (A-2-3-4-5)


# --------------------------------------------------------------------------
# Card helpers
# --------------------------------------------------------------------------

def make_card(rank: int, suit: int) -> int:
    assert 0 <= rank < 13 and 0 <= suit < 4
    return 4 * rank + suit


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def parse_card(s: str) -> int:
    """'As' -> 51, 'Td' -> 33, '2c' -> 0. Case-insensitive rank ('t' ok)."""
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    r = RANK_CHARS.find(s[0].upper())
    u = SUIT_CHARS.find(s[1].lower())
    if r < 0 or u < 0:
        raise ValueError(f"bad card {s!r}")
    return make_card(r, u)


def parse_cards(s: str) -> list[int]:
    """'As Kd 7h' or 'AsKd7h' -> list of ints."""
    s = s.replace(" ", "").replace(",", "")
    if len(s) % 2:
        raise ValueError(f"bad card string {s!r}")
    return [parse_card(s[i:i + 2]) for i in range(0, len(s), 2)]


def card_str(card: int) -> str:
    return RANK_CHARS[rank_of(card)] + SUIT_CHARS[suit_of(card)]


def cards_str(cards) -> str:
    return " ".join(card_str(c) for c in cards)


# --------------------------------------------------------------------------
# Hand categories (higher is better)
# --------------------------------------------------------------------------

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
]


# --------------------------------------------------------------------------
# 5-card reference evaluator
# --------------------------------------------------------------------------

def _straight_high(distinct_ranks_desc: list[int]) -> int | None:
    """Given 5 distinct ranks sorted high->low, return the straight's top
    rank, or None if they don't form a straight. The wheel A-2-3-4-5 is a
    straight with top card 5 (so it loses to 2-3-4-5-6)."""
    if len(distinct_ranks_desc) != 5:
        return None
    if distinct_ranks_desc[0] - distinct_ranks_desc[4] == 4:
        return distinct_ranks_desc[0]
    if distinct_ranks_desc == [ACE, FIVE, 2, 1, 0]:  # A 5 4 3 2
        return FIVE
    return None


def evaluate5(cards) -> tuple:
    """Evaluate exactly 5 distinct cards.

    Returns (category, tiebreak_ranks...) as a tuple. Tuples compare
    lexicographically, so a bigger tuple is a better hand and equal tuples
    are an exact tie (split pot).

        straight flush / straight : (cat, top_rank)
        quads                     : (cat, quad_rank, kicker)
        full house                : (cat, trip_rank, pair_rank)
        flush / high card         : (cat, r1, r2, r3, r4, r5)   high->low
        trips                     : (cat, trip_rank, k1, k2)
        two pair                  : (cat, high_pair, low_pair, kicker)
        one pair                  : (cat, pair_rank, k1, k2, k3)
    """
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError(f"need 5 distinct cards, got {cards!r}")
    for c in cards:
        if not 0 <= c < 52:
            raise ValueError(f"card out of range: {c}")

    ranks = [rank_of(c) for c in cards]
    suits = [suit_of(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Group ranks by multiplicity, then by rank: the order kickers are
    # compared in. E.g. K K 7 7 2 -> [(2,K), (2,7), (1,2)].
    counts = Counter(ranks)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    shape = [n for n, _ in groups]            # e.g. [2, 2, 1]
    ordered = [r for _, r in groups]          # e.g. [K, 7, 2]

    straight_top = _straight_high(sorted(counts, reverse=True))

    if straight_top is not None and is_flush:
        return (STRAIGHT_FLUSH, straight_top)
    if shape == [4, 1]:
        return (QUADS, *ordered)
    if shape == [3, 2]:
        return (FULL_HOUSE, *ordered)
    if is_flush:
        return (FLUSH, *ordered)              # 5 distinct ranks, high->low
    if straight_top is not None:
        return (STRAIGHT, straight_top)
    if shape == [3, 1, 1]:
        return (TRIPS, *ordered)
    if shape == [2, 2, 1]:
        return (TWO_PAIR, *ordered)
    if shape == [2, 1, 1, 1]:
        return (ONE_PAIR, *ordered)
    return (HIGH_CARD, *ordered)


def category_of(value: tuple) -> int:
    return value[0]


def describe(value: tuple) -> str:
    cat = value[0]
    rest = "".join(RANK_CHARS[r] for r in value[1:])
    return f"{CATEGORY_NAMES[cat]} [{rest}]"


def best_of(cards) -> tuple:
    """Best 5-card hand from 5..7 cards (e.g. 2 hole + 5 board), by brute
    force over all 5-card subsets. Slow, but uses only evaluate5."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("need 5 to 7 cards")
    return max(evaluate5(combo) for combo in combinations(cards, 5))
