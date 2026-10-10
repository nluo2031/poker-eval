/*
 * Step 7: fast 7-card evaluator. Header-only.
 *
 * Cards are ints 0..51, card = 4*rank + suit (rank 0 = deuce .. 12 = ace).
 * Returns 1..7462, higher is better (7462 = royal flush).
 *
 * With 7 cards at most one suit can have >= 5, and a flush, when present,
 * always beats anything else the 7 cards can make (quads or a full house
 * next to a 5-card flush would need 8 cards). So:
 *   flush      -> FLUSH_TABLE[13-bit rank mask of the flush suit]
 *   otherwise  -> NONFLUSH_TABLE[index of the rank-count vector]
 * The index is computed by walking the present ranks low to high, adding
 * COEF[rank][cards_left][count] (see tables.py for the derivation).
 */
#ifndef EVAL7_H
#define EVAL7_H

#include <stdint.h>
#include "eval7_tables.h"

static inline uint16_t eval7(const uint8_t *c)
{
    uint64_t counts = 0;    /* 4 bits per rank */
    uint64_t suits = 0;     /* 16 bits per suit: rank mask */
    uint32_t sc = 0x3333;   /* 4 bits per suit, biased by 3: bit 3 set <=> >= 5 cards */

    for (int i = 0; i < 7; i++) {
        unsigned r = c[i] >> 2, s = c[i] & 3;
        counts += 1ull << (4 * r);
        suits |= 1ull << (16 * s + r);
        sc += 1u << (4 * s);
    }

    uint32_t f = sc & 0x8888;
    if (f) {
        unsigned s = (unsigned)__builtin_ctz(f) >> 2;
        return FLUSH_TABLE[(suits >> (16 * s)) & 0x1fff];
    }

    uint32_t present = (uint32_t)((suits | suits >> 16 | suits >> 32 | suits >> 48) & 0x1fff);
    unsigned idx = 0, left = 7;
    while (present) {
        unsigned r = (unsigned)__builtin_ctz(present);
        present &= present - 1;
        unsigned n = (unsigned)(counts >> (4 * r)) & 0xf;
        idx += COEF[(r * 8 + left) * 5 + n];
        left -= n;
    }
    return NONFLUSH_TABLE[idx];
}

/* Category of a rank: 0 = high card .. 8 = straight flush. */
static inline int eval7_category(uint16_t rank)
{
    int cat = 0;
    while (rank >= CAT_FIRST[cat + 1])
        cat++;
    return cat;
}

#endif
