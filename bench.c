/*
 * Step 8b: benchmark eval7, single thread.
 *   random: 10M pre-generated random hands from an array
 *   enumeration: all 133,784,560 hands in nested-loop order
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "eval7.h"

#define NRANDOM 10000000

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_rand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

int main(void)
{
    uint8_t *hands = malloc((size_t)NRANDOM * 7);
    for (size_t i = 0; i < NRANDOM; i++) {
        uint64_t used = 0;
        for (int k = 0; k < 7;) {
            unsigned c = (unsigned)(next_rand() % 52);
            if (!(used >> c & 1)) {
                used |= 1ull << c;
                hands[i * 7 + k++] = (uint8_t)c;
            }
        }
    }

    uint64_t sum = 0;
    double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double t = now();
        for (size_t i = 0; i < NRANDOM; i++)
            sum += eval7(hands + i * 7);
        t = now() - t;
        if (t < best)
            best = t;
    }
    printf("random:      %6.1f M hands/s  (%.2f ns/hand, best of 5 x %dM)\n",
           NRANDOM / best / 1e6, best / NRANDOM * 1e9, NRANDOM / 1000000);

    best = 1e9;
    uint64_t n = 0;
    for (int rep = 0; rep < 3; rep++) {
        n = 0;
        double t = now();
        uint8_t c[7];
        for (c[0] = 0; c[0] < 52; c[0]++)
        for (c[1] = c[0] + 1; c[1] < 52; c[1]++)
        for (c[2] = c[1] + 1; c[2] < 52; c[2]++)
        for (c[3] = c[2] + 1; c[3] < 52; c[3]++)
        for (c[4] = c[3] + 1; c[4] < 52; c[4]++)
        for (c[5] = c[4] + 1; c[5] < 52; c[5]++)
        for (c[6] = c[5] + 1; c[6] < 52; c[6]++, n++)
            sum += eval7(c);
        t = now() - t;
        if (t < best)
            best = t;
    }
    printf("enumeration: %6.1f M hands/s  (%.2f ns/hand, best of 3 x %llu)\n",
           n / best / 1e6, best / n * 1e9, (unsigned long long)n);

    printf("(checksum %llu)\n", (unsigned long long)sum);
    free(hands);
    return 0;
}
