/*
 * Step 8a: exhaustive verification.
 *
 *  1. C port of the 5-card reference evaluator == Python evaluate5 on all
 *     2,598,960 hands (compared against ref5.bin written by tables.py).
 *  2. Fast eval7 == oracle (max over the 21 subsets) on all 133,784,560
 *     7-card hands, multithreaded.
 *  3. Known 7-card category frequencies and the 4,824 distinct ranks
 *     reachable with 7 cards.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "eval7.h"

/* ---- 5-card reference, ported line-for-line from eval5.evaluate5 ---- */

enum { HIGH_CARD, ONE_PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH };

static const uint32_t P13[6] = {1, 13, 169, 2197, 28561, 371293};

/* Returns tables.class_key(evaluate5(cards)). */
static uint32_t ref5_key(const uint8_t *c)
{
    int cnt[13] = {0};
    int flush = 1;
    for (int i = 0; i < 5; i++) {
        cnt[c[i] >> 2]++;
        if ((c[i] & 3) != (c[0] & 3))
            flush = 0;
    }
    /* ranks ordered by (count desc, rank desc) */
    int ord[5], shape[5], k = 0;
    for (int n = 4; n >= 1; n--)
        for (int r = 12; r >= 0; r--)
            if (cnt[r] == n) {
                ord[k] = r;
                shape[k] = n;
                k++;
            }

    int top = -1;
    if (k == 5) {
        if (ord[0] - ord[4] == 4)
            top = ord[0];
        else if (ord[0] == 12 && ord[1] == 3) /* A 5 4 3 2 */
            top = 3;
    }

    int cat;
    if (top >= 0 && flush)                 cat = STRAIGHT_FLUSH;
    else if (shape[0] == 4)                cat = QUADS;
    else if (shape[0] == 3 && shape[1] == 2) cat = FULL_HOUSE;
    else if (flush)                        cat = FLUSH;
    else if (top >= 0)                     cat = STRAIGHT;
    else if (shape[0] == 3)                cat = TRIPS;
    else if (shape[0] == 2 && shape[1] == 2) cat = TWO_PAIR;
    else if (shape[0] == 2)                cat = ONE_PAIR;
    else                                   cat = HIGH_CARD;

    uint32_t key = cat * P13[5];
    if (cat == STRAIGHT || cat == STRAIGHT_FLUSH)
        return key + top * P13[4];
    for (int i = 0; i < k; i++)
        key += ord[i] * P13[4 - i];
    return key;
}

static uint16_t key_to_rank(uint32_t key)
{
    int lo = 0, hi = NUM_CLASSES - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (CLASS_KEYS[mid] < key) lo = mid + 1;
        else if (CLASS_KEYS[mid] > key) hi = mid - 1;
        else return (uint16_t)(mid + 1);
    }
    fprintf(stderr, "key %u is not a hand class\n", key);
    exit(1);
}

/* ---- 7-card oracle ---- */

static uint8_t SUBSETS[21][5];

static void init_subsets(void)
{
    int n = 0;
    for (int skip1 = 0; skip1 < 7; skip1++)
        for (int skip2 = skip1 + 1; skip2 < 7; skip2++) {
            int k = 0;
            for (int i = 0; i < 7; i++)
                if (i != skip1 && i != skip2)
                    SUBSETS[n][k++] = (uint8_t)i;
            n++;
        }
}

static uint16_t eval7_oracle(const uint8_t *c)
{
    uint32_t best = 0;
    for (int s = 0; s < 21; s++) {
        uint8_t h[5];
        for (int i = 0; i < 5; i++)
            h[i] = c[SUBSETS[s][i]];
        uint32_t k = ref5_key(h);
        if (k > best)
            best = k;
    }
    return key_to_rank(best);
}

/* ---- step 1: C reference vs Python reference ---- */

static void check_ref5(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    static uint16_t expect[2598960];
    if (fread(expect, 2, 2598960, f) != 2598960) {
        fprintf(stderr, "%s: short read\n", path);
        exit(1);
    }
    fclose(f);

    size_t i = 0, bad = 0;
    uint8_t c[5];
    for (c[0] = 0; c[0] < 52; c[0]++)
    for (c[1] = c[0] + 1; c[1] < 52; c[1]++)
    for (c[2] = c[1] + 1; c[2] < 52; c[2]++)
    for (c[3] = c[2] + 1; c[3] < 52; c[3]++)
    for (c[4] = c[3] + 1; c[4] < 52; c[4]++, i++)
        if (key_to_rank(ref5_key(c)) != expect[i] && bad++ < 10)
            fprintf(stderr, "ref5 mismatch at hand %zu\n", i);
    printf("C ref5 vs Python evaluate5: %zu hands, %zu mismatches\n", i, bad);
    if (bad)
        exit(1);
}

/* ---- step 2/3: exhaustive 7-card check, parallel over (c0, c1) ---- */

typedef struct {
    uint64_t hands, mismatches;
    uint64_t cat[9];
    uint8_t seen[NUM_CLASSES + 1];
} Tally;

static atomic_int next_unit;

static void *worker(void *arg)
{
    Tally *t = arg;
    for (;;) {
        int u = atomic_fetch_add(&next_unit, 1);
        if (u >= 52 * 52)
            break;
        uint8_t c[7];
        c[0] = (uint8_t)(u / 52);
        c[1] = (uint8_t)(u % 52);
        if (c[1] <= c[0])
            continue;
        for (c[2] = c[1] + 1; c[2] < 52; c[2]++)
        for (c[3] = c[2] + 1; c[3] < 52; c[3]++)
        for (c[4] = c[3] + 1; c[4] < 52; c[4]++)
        for (c[5] = c[4] + 1; c[5] < 52; c[5]++)
        for (c[6] = c[5] + 1; c[6] < 52; c[6]++) {
            uint16_t fast = eval7(c), ref = eval7_oracle(c);
            if (fast != ref && t->mismatches++ < 5)
                fprintf(stderr, "MISMATCH %d %d %d %d %d %d %d: fast %u oracle %u\n",
                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], fast, ref);
            t->hands++;
            t->cat[eval7_category(ref)]++;
            t->seen[ref] = 1;
        }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    const char *ref5_path = argc > 1 ? argv[1] : "ref5.bin";
    init_subsets();
    check_ref5(ref5_path);

    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1)
        nthreads = 1;
    Tally *tallies = calloc((size_t)nthreads, sizeof(Tally));
    pthread_t *th = malloc(sizeof(pthread_t) * (size_t)nthreads);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < nthreads; i++)
        pthread_create(&th[i], NULL, worker, &tallies[i]);
    for (int i = 0; i < nthreads; i++)
        pthread_join(th[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    Tally total = {0};
    for (int i = 0; i < nthreads; i++) {
        total.hands += tallies[i].hands;
        total.mismatches += tallies[i].mismatches;
        for (int k = 0; k < 9; k++)
            total.cat[k] += tallies[i].cat[k];
        for (int r = 1; r <= NUM_CLASSES; r++)
            total.seen[r] |= tallies[i].seen[r];
    }
    int distinct = 0;
    for (int r = 1; r <= NUM_CLASSES; r++)
        distinct += total.seen[r];

    printf("7-card hands: %llu, mismatches: %llu (%d threads, %.1fs)\n",
           (unsigned long long)total.hands, (unsigned long long)total.mismatches,
           nthreads, (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);

    static const char *names[9] = {"High Card", "One Pair", "Two Pair", "Three of a Kind",
                                   "Straight", "Flush", "Full House", "Four of a Kind",
                                   "Straight Flush"};
    static const uint64_t known[9] = {23294460, 58627800, 31433400, 6461620, 6180020,
                                      4047644, 3473184, 224848, 41584};
    int ok = total.hands == 133784560 && total.mismatches == 0 && distinct == 4824;
    for (int k = 8; k >= 0; k--) {
        int good = total.cat[k] == known[k];
        ok &= good;
        printf("  %-16s %10llu %s\n", names[k], (unsigned long long)total.cat[k],
               good ? "ok" : "EXPECTED DIFFERENT");
    }
    printf("distinct 7-card ranks: %d (expected 4824)\n", distinct);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
