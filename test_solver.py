"""
Step 9 tests: run ./solver on small spots with known answers.

1. River, polarized vs bluff catchers, one pot-size (= all-in) bet:
   OOP KK (3 combos) + QJo air, IP AK. Card removal: each KK combo is
   compatible with 4 AK combos, each QJo with all 12. Theory: bluffs are 1/3
   of the betting mass. Value mass = 3 * 4 = 12, so bluff mass = 6 = half a
   QJo combo, i.e. each QJo bets 0.5 / 12 = 4.17%. IP calls 50% (MDF), KK
   always bets.
2. Turn spot converges below 0.5% of pot, and the reported OOP + IP root EVs
   sum to the pot (constant-sum check).
3. Suit isomorphism: on a turn with two missing suits the merged tree is
   smaller, gives the same exploitability as the full tree, and a river card
   followed through a merged subtree shows the suit-swapped strategy.
"""

import re
import subprocess
import sys

SOLVER = "./solver"


def run(*args):
    out = subprocess.run([SOLVER, *args], capture_output=True, text=True, check=True).stdout
    return out


def last_exploit_pct(out):
    return float(re.findall(r"\(([\d.]+)% of pot\)", out)[-1])


def node_rows(out, title):
    """{hand: [freq%, ...]} for the per-combo table of the node '== title =='."""
    block = out.split(f"== {title} ==")[1].split("\n== ")[0]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"^([2-9TJQKA][cdhs][2-9TJQKA][cdhs])\s", line)
        if m:
            rows[m.group(1)] = [float(x) for x in re.findall(r"([\d.]+)%", line)]
    return block, rows


def swap_suits(hand, a, b):
    """'AhTc' with h<->c -> 'AcTh', printed the way the solver does: higher card first."""
    key = lambda card: ("23456789TJQKA".index(card[0]), "cdhs".index(card[1]))
    cards = [r + {a: b, b: a}.get(u, u) for r, u in (hand[:2], hand[2:])]
    return "".join(sorted(cards, key=key, reverse=True))


def test_river_polarized():
    out = run("--board", "Kh9c5d3s2h", "--oop", "KK,QJo", "--ip", "AK",
              "--pot", "10", "--stack", "10", "--bet", "100", "--raise", "",
              "--no-allin", "--iters", "3000", "--target", "0.001", "--check", "100",
              "--combos", "all", "--show", "", "--show", "A", "--threads", "1")
    assert last_exploit_pct(out) < 0.01, out
    _, root = node_rows(out, "root")
    for hand, (x, a) in root.items():
        if hand[0] == "K":
            assert a > 99.5, (hand, a)
        else:
            assert abs(a - 100 / 24) < 0.3, (hand, a)
    _, facing = node_rows(out, "A")
    for hand, (f, c) in facing.items():
        assert abs(c - 50) < 1.0, (hand, c)
    print("river polarized: ok")


def test_turn_converges():
    out = run("--board", "Ks7h2d9c", "--oop", "22+,A2s+,KTs+,QTs+,JTs,T9s,98s,ATo+,KJo+",
              "--ip", "22-JJ,A2s-AJs,KTs+,QTs+,JTs,T9s,98s,87s,ATo-AQo,KJo+",
              "--pot", "20", "--stack", "90", "--bet", "50,100", "--raise", "75",
              "--iters", "500", "--target", "0.5", "--check", "20", "--combos", "0")
    pct = last_exploit_pct(out)
    assert pct <= 0.5, pct
    m = re.search(r"OOP ([-\d.]+), IP ([-\d.]+)", out)
    assert abs(float(m.group(1)) + float(m.group(2)) - 20) < 1e-2, m.group(0)
    print(f"turn converges: ok ({pct}% of pot)")


def test_isomorphism():
    args = ["--board", "Ks7s2s9d", "--oop", "22+,A2s+,KTs+,QTs+,JTs,T9s,98s,ATo+,KJo+",
            "--ip", "22-JJ,A2s-AJs,KTs+,QTs+,JTs,T9s,98s,87s,ATo-AQo,KJo+",
            "--pot", "20", "--stack", "90", "--bet", "50,100", "--raise", "75",
            "--iters", "60", "--target", "0", "--check", "60", "--threads", "1",
            "--combos", "all", "--show", "X B100 C 4h B100", "--show", "X B100 C 4c B100"]
    iso, full = run(*args), run(*args, "--no-iso")
    nodes = lambda out: int(re.search(r"tree: (\d+) nodes", out).group(1))
    assert nodes(iso) < nodes(full), (nodes(iso), nodes(full))
    # same game, so same trajectory up to float rounding (which compounds)
    ei, ef = last_exploit_pct(iso), last_exploit_pct(full)
    assert abs(ei - ef) < 0.05 * ef, (ei, ef)
    ev = lambda out: float(re.search(r"OOP ([-\d.]+), IP", out).group(1))
    assert abs(ev(iso) - ev(full)) < 0.01, (ev(iso), ev(full))
    _, on_4h = node_rows(iso, "X B100 C 4h B100")
    _, on_4c = node_rows(iso, "X B100 C 4c B100")
    for hand, freqs in on_4h.items():
        mirrored = swap_suits(hand, "h", "c")
        assert on_4c[mirrored] == freqs, (hand, mirrored)
    print(f"isomorphism: ok ({nodes(iso)} vs {nodes(full)} nodes)")


if __name__ == "__main__":
    test_river_polarized()
    test_turn_converges()
    test_isomorphism()
    print("all solver tests passed")
    sys.exit(0)
