/*
 * Step 9: heads-up postflop solver (flop, turn or river root).
 *
 * Abstraction: fixed starting ranges, board, pot and effective stack, a menu
 * of bet / raise sizes (percent of pot) per street, optional all-in, and a
 * cap on raises per street. OOP (player 0) acts first on every street.
 *
 * Algorithm: Discounted CFR (alpha = 1.5, beta = 0, gamma = 2) with
 * alternating updates, run on the public tree. Every public node carries
 * vectors indexed by each player's hand list, so one traversal updates the
 * information sets "this node x each of my hands" for all hands at once:
 *   down the tree: reach probabilities of each player's hands
 *   up the tree:   counterfactual values of the traverser's hands
 *
 * Utilities are chips relative to the root: a player ends with -c (what they
 * put in since the root) when losing, pot0 + c when winning, pot0/2 - 0 on a
 * split (contributions are equal at showdown). u0 + u1 = pot0 at every leaf,
 * so the game is constant-sum and the root value is each player's share of
 * the starting pot.
 *
 * Card removal at terminals (opponent hands that share a card with mine are
 * impossible):
 *   fold:     inclusion-exclusion over per-card reach sums          O(n)
 *   showdown: sweep both hand lists presorted by strength           O(n)
 * Chance nodes zero the reach of hands holding the dealt card and average
 * children over the K - 4 cards compatible with both hands.
 *
 * Exploitability = (BR0 + BR1 - pot0) / 2, from a best-response pass against
 * the average strategy. Output: per-hand strategy and per-action EV at any
 * node (EV measured from that node: folding is worth 0).
 */
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "eval7.h"

#define MAX_SIZES 8
#define MAX_ACTIONS (MAX_SIZES + 4)
#define NCOMBOS 1326

static const char RANKS[] = "23456789TJQKA";
static const char SUITS[] = "cdhs";

/* ------------------------------------------------------------------------ */
/* Cards, combos, ranges                                                     */
/* ------------------------------------------------------------------------ */

static int combo_c1[NCOMBOS], combo_c2[NCOMBOS]; /* c1 > c2 */
static int combo_id[52][52];

static void init_combos(void)
{
    int k = 0;
    for (int a = 0; a < 52; a++)
        for (int b = 0; b < a; b++) {
            combo_c1[k] = a;
            combo_c2[k] = b;
            combo_id[a][b] = combo_id[b][a] = k++;
        }
}

static int parse_card(const char *s)
{
    const char *r = strchr(RANKS, s[0] >= 'a' ? s[0] - 32 : s[0]);
    const char *u = strchr(SUITS, s[1] | 32);
    if (!s[0] || !s[1] || !r || !u)
        return -1;
    return 4 * (int)(r - RANKS) + (int)(u - SUITS);
}

static void card_str(int c, char *out)
{
    out[0] = RANKS[c >> 2];
    out[1] = SUITS[c & 3];
    out[2] = 0;
}

static int rank_of_char(char ch)
{
    const char *r = strchr(RANKS, ch >= 'a' ? ch - 32 : ch);
    return ch && r ? (int)(r - RANKS) : -1;
}

/* kind: 'p' pair, 's' suited, 'o' offsuit, 'a' both */
static void add_class(float *w, int hi, int lo, char kind, float wt)
{
    for (int s1 = 0; s1 < 4; s1++)
        for (int s2 = 0; s2 < 4; s2++) {
            if (hi == lo && s2 <= s1)
                continue;
            if (hi != lo && ((kind == 's' && s1 != s2) || (kind == 'o' && s1 == s2)))
                continue;
            w[combo_id[4 * hi + s1][4 * lo + s2]] = wt;
        }
}

/* Parses "QQ+,AKs,AQo:0.5,KTs+,A5s-A2s,99-66,AhKh,random". Returns 0 on success. */
static int parse_range(const char *text, float *w)
{
    memset(w, 0, NCOMBOS * sizeof *w);
    char *buf = strdup(text), *save = NULL;
    for (char *tok = strtok_r(buf, ", \t\n", &save); tok; tok = strtok_r(NULL, ", \t\n", &save)) {
        float wt = 1.0f;
        char *colon = strchr(tok, ':');
        if (colon) {
            *colon = 0;
            wt = strtof(colon + 1, NULL);
        }
        if (!strcasecmp(tok, "random") || !strcasecmp(tok, "any")) {
            for (int k = 0; k < NCOMBOS; k++)
                w[k] = wt;
            continue;
        }
        size_t n = strlen(tok);
        if (n == 4 && parse_card(tok) >= 0 && parse_card(tok + 2) >= 0) {
            int a = parse_card(tok), b = parse_card(tok + 2);
            if (a == b)
                goto bad;
            w[combo_id[a][b]] = wt;
            continue;
        }
        int r1 = rank_of_char(tok[0]), r2 = n > 1 ? rank_of_char(tok[1]) : -1;
        if (r1 < 0 || r2 < 0)
            goto bad;
        int hi = r1 > r2 ? r1 : r2, lo = r1 > r2 ? r2 : r1;
        char kind = hi == lo ? 'p' : 'a';
        const char *rest = tok + 2;
        if (kind != 'p' && (*rest == 's' || *rest == 'S' || *rest == 'o' || *rest == 'O'))
            kind = (char)(*rest++ | 32);
        if (*rest == 0) {
            add_class(w, hi, lo, kind, wt);
        } else if (!strcmp(rest, "+")) {
            if (kind == 'p')
                for (int r = hi; r < 13; r++)
                    add_class(w, r, r, 'p', wt);
            else
                for (int r = lo; r < hi; r++)
                    add_class(w, hi, r, kind, wt);
        } else if (*rest == '-') {
            rest++;
            int e1 = rank_of_char(rest[0]), e2 = rank_of_char(rest[1]);
            if (e1 < 0 || e2 < 0)
                goto bad;
            if (kind == 'p') {
                int a = e1 < hi ? e1 : hi, b = e1 < hi ? hi : e1;
                for (int r = a; r <= b; r++)
                    add_class(w, r, r, 'p', wt);
            } else {
                if (e1 != hi)
                    goto bad;
                int a = e2 < lo ? e2 : lo, b = e2 < lo ? lo : e2;
                for (int r = a; r <= b && r < hi; r++)
                    add_class(w, hi, r, kind, wt);
            }
        } else {
            goto bad;
        }
        continue;
    bad:
        fprintf(stderr, "bad range token '%s'\n", tok);
        free(buf);
        return -1;
    }
    free(buf);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Game configuration and per-player hand lists                              */
/* ------------------------------------------------------------------------ */

typedef struct {
    int nbet[3], nraise[3];            /* per street: 0 flop, 1 turn, 2 river */
    float bet[3][MAX_SIZES], raise[3][MAX_SIZES];
    int allin, raise_cap;
    double pot0, stack;
} Config;

static Config cfg;
static int board0[5], nboard0;

static int nh[2];                      /* hand counts */
static int *hc1[2], *hc2[2];           /* hand cards */
static uint64_t *hmask[2];             /* hand card masks */
static float *hweight[2];              /* starting range weights */
static int *same[2];                   /* index of the same combo in the other list, or -1 */

/* ------------------------------------------------------------------------ */
/* Showdown boards: hand strengths presorted, shared by all showdown nodes  */
/* on the same 5-card board                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    uint16_t str;       /* eval7 rank, 0 if blocked by the board */
    uint8_t c1, c2;
    uint16_t idx;       /* index in the player's hand list */
} SortedHand;

typedef struct {
    SortedHand *sorted[2]; /* each player's hands by ascending strength */
} BoardEval;

#define BCACHE 8192
static uint64_t bcache_key[BCACHE];
static BoardEval *bcache_val[BCACHE];

static int cmp_by_strength(const void *a, const void *b)
{
    uint16_t x = ((const SortedHand *)a)->str, y = ((const SortedHand *)b)->str;
    return (x > y) - (x < y);
}

static BoardEval *board_eval(const int *board)
{
    uint64_t key = 0;
    for (int i = 0; i < 5; i++)
        key |= 1ull << board[i];
    size_t slot = (size_t)((key * 0x9e3779b97f4a7c15ull) >> 51) & (BCACHE - 1);
    while (bcache_val[slot] && bcache_key[slot] != key)
        slot = (slot + 1) & (BCACHE - 1);
    if (bcache_val[slot])
        return bcache_val[slot];

    BoardEval *be = malloc(sizeof *be);
    for (int p = 0; p < 2; p++) {
        SortedHand *sh = be->sorted[p] = malloc(nh[p] * sizeof(SortedHand));
        for (int h = 0; h < nh[p]; h++) {
            uint8_t c[7] = {board[0], board[1], board[2], board[3], board[4], hc1[p][h], hc2[p][h]};
            sh[h] = (SortedHand){(hmask[p][h] & key) ? 0 : eval7(c), c[5], c[6], (uint16_t)h};
        }
        qsort(sh, nh[p], sizeof(SortedHand), cmp_by_strength);
    }
    bcache_key[slot] = key;
    bcache_val[slot] = be;
    return be;
}

/* ------------------------------------------------------------------------ */
/* Suit isomorphism                                                          */
/*                                                                           */
/* A suit permutation g that maps both ranges onto themselves (same weights) */
/* and fixes the board at a chance node maps the subtree after card c onto   */
/* the subtree after g(c), hands and all. Every strategy above that node is  */
/* then g-symmetric too, so the reach vectors arriving there are g-invariant */
/* and value_{c}[h] = value_{g(c)}[g(h)]. Only one card per orbit gets a     */
/* subtree; the others reuse it through a hand-index permutation. Monotone   */
/* boards gain most, two-tone and paired boards some, rainbow flops nothing. */
/* ------------------------------------------------------------------------ */

static int suit_perm[24][4];
static int range_sym[24];           /* both ranges invariant under the perm */
static int *hperm[24][2];           /* hand index -> index of the permuted hand */

static int perm_card(int g, int c)
{
    return (c & ~3) | suit_perm[g][c & 3];
}

static void init_suit_perms(int iso)
{
    int g = 0;
    for (int a = 0; a < 4; a++)        /* g = 0 is the identity */
        for (int b = 0; b < 4; b++)
            for (int c = 0; c < 4; c++)
                for (int d = 0; d < 4; d++)
                    if (a != b && a != c && a != d && b != c && b != d && c != d) {
                        suit_perm[g][0] = a;
                        suit_perm[g][1] = b;
                        suit_perm[g][2] = c;
                        suit_perm[g][3] = d;
                        g++;
                    }
    int index_of[2][NCOMBOS];
    for (int p = 0; p < 2; p++) {
        for (int k = 0; k < NCOMBOS; k++)
            index_of[p][k] = -1;
        for (int h = 0; h < nh[p]; h++)
            index_of[p][combo_id[hc1[p][h]][hc2[p][h]]] = h;
    }
    for (g = 0; g < 24; g++) {
        range_sym[g] = g == 0 || iso;
        for (int p = 0; p < 2; p++) {
            hperm[g][p] = malloc(nh[p] * sizeof(int));
            for (int h = 0; h < nh[p]; h++) {
                int j = index_of[p][combo_id[perm_card(g, hc1[p][h])][perm_card(g, hc2[p][h])]];
                hperm[g][p][h] = j;
                if (j < 0 || hweight[p][j] != hweight[p][h])
                    range_sym[g] = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Public tree                                                               */
/* ------------------------------------------------------------------------ */

enum { N_ACTION, N_CHANCE, N_FOLD, N_SHOWDOWN };

typedef struct Node Node;
struct Node {
    uint8_t type, player, nchild, nboard; /* player: actor, or folder for N_FOLD */
    int board[5];
    double c[2];                          /* chips put in since the root */
    Node **child;
    char (*label)[8];                     /* N_ACTION: action labels; N_CHANCE: card strings */
    int *card;                            /* N_CHANCE: card dealt to each child (canonical) */
    uint8_t ncards;                       /* N_CHANCE: all cards that can come, K = 52 - nboard */
    uint8_t *map_child, *map_perm;        /* N_CHANCE, per card: canonical child, suit perm */
    int *map_card;                        /*   mapping that card onto the child's card */
    float *regret, *ssum;                 /* N_ACTION: [nchild][nh[player]] */
    BoardEval *be;                        /* N_SHOWDOWN */
};

static size_t n_action_nodes, n_nodes, storage_floats;

static Node *new_node(int type, const int *board, int nboard, const double *c)
{
    Node *n = calloc(1, sizeof *n);
    n->type = (uint8_t)type;
    n->nboard = (uint8_t)nboard;
    memcpy(n->board, board, nboard * sizeof(int));
    n->c[0] = c[0];
    n->c[1] = c[1];
    n_nodes++;
    return n;
}

static Node *build_action(const int *board, int nboard, const double *c, int player, int nraises, int checked);

/* After a street closes with equal contributions. */
static Node *build_street_end(const int *board, int nboard, const double *c)
{
    if (nboard == 5) {
        Node *n = new_node(N_SHOWDOWN, board, nboard, c);
        n->be = board_eval(board);
        return n;
    }
    Node *n = new_node(N_CHANCE, board, nboard, c);
    uint64_t used = 0;
    for (int i = 0; i < nboard; i++)
        used |= 1ull << board[i];

    /* symmetries of this board */
    int group[24], ng = 0;
    for (int g = 0; g < 24; g++) {
        int ok = range_sym[g];
        for (int i = 0; i < nboard && ok; i++)
            ok = used >> perm_card(g, board[i]) & 1;
        if (ok)
            group[ng++] = g;
    }

    int K = 52 - nboard, nchild = 0, child_of[52];
    n->ncards = (uint8_t)K;
    n->map_card = malloc(K * sizeof(int));
    n->map_child = malloc(K);
    n->map_perm = malloc(K);
    n->card = malloc(K * sizeof(int));
    for (int card = 0, j = 0; card < 52; card++) {
        if (used >> card & 1)
            continue;
        /* canonical card: smallest image; identity wins ties */
        int canon = card, gbest = 0;
        for (int i = 0; i < ng; i++)
            if (perm_card(group[i], card) < canon) {
                canon = perm_card(group[i], card);
                gbest = group[i];
            }
        if (canon == card) {
            child_of[card] = nchild;
            n->card[nchild++] = card;
        }
        n->map_card[j] = card;
        n->map_child[j] = (uint8_t)child_of[canon];
        n->map_perm[j] = (uint8_t)gbest;
        j++;
    }

    n->nchild = (uint8_t)nchild;
    n->child = malloc(nchild * sizeof(Node *));
    n->label = malloc(nchild * sizeof *n->label);
    int nb[5];
    memcpy(nb, board, nboard * sizeof(int));
    for (int k = 0; k < nchild; k++) {
        nb[nboard] = n->card[k];
        card_str(n->card[k], n->label[k]);
        if (c[0] >= cfg.stack) /* all in: run it out */
            n->child[k] = build_street_end(nb, nboard + 1, c);
        else
            n->child[k] = build_action(nb, nboard + 1, c, 0, 0, 0);
    }
    return n;
}

static int add_size(double *amts, int *labels_pct, int n, double amt, int pct)
{
    for (int i = 0; i < n; i++)
        if (fabs(amts[i] - amt) < 1e-9)
            return n;
    amts[n] = amt;
    labels_pct[n] = pct;
    return n + 1;
}

static Node *build_action(const int *board, int nboard, const double *c, int player, int nraises, int checked)
{
    Node *n = new_node(N_ACTION, board, nboard, c);
    n->player = (uint8_t)player;
    int opp = 1 - player, street = nboard - 3;
    double pot = cfg.pot0 + c[0] + c[1];
    double facing = c[opp] - c[player];
    double left = cfg.stack - c[player];

    Node *kids[MAX_ACTIONS];
    char labels[MAX_ACTIONS][8];
    int k = 0;
    double nc[2];

    if (facing <= 0) {
        /* check */
        strcpy(labels[k], "X");
        kids[k++] = checked || player == 1 ? build_street_end(board, nboard, c)
                                           : build_action(board, nboard, c, 1, 0, 1);
        /* bets: total chips added by player */
        double amts[MAX_SIZES + 1];
        int pcts[MAX_SIZES + 1], na = 0;
        for (int i = 0; i < cfg.nbet[street]; i++) {
            double a = cfg.bet[street][i] / 100.0 * pot;
            na = a >= left ? add_size(amts, pcts, na, left, -1)
                           : add_size(amts, pcts, na, a, (int)lround(cfg.bet[street][i]));
        }
        if (cfg.allin)
            na = add_size(amts, pcts, na, left, -1);
        for (int i = 0; i < na; i++) {
            nc[0] = c[0];
            nc[1] = c[1];
            nc[player] += amts[i];
            if (pcts[i] < 0)
                strcpy(labels[k], "A");
            else
                snprintf(labels[k], 8, "B%d", pcts[i]);
            kids[k++] = build_action(board, nboard, nc, opp, 0, checked);
        }
    } else {
        strcpy(labels[k], "F");
        Node *f = new_node(N_FOLD, board, nboard, c);
        f->player = (uint8_t)player;
        kids[k++] = f;

        nc[0] = c[0];
        nc[1] = c[1];
        nc[player] = c[opp];
        strcpy(labels[k], "C");
        kids[k++] = build_street_end(board, nboard, nc);

        if (nraises < cfg.raise_cap && c[opp] < cfg.stack && left > facing) {
            double amts[MAX_SIZES + 1];
            int pcts[MAX_SIZES + 1], na = 0;
            for (int i = 0; i < cfg.nraise[street]; i++) {
                /* call, then bet the given fraction of the new pot */
                double a = facing + cfg.raise[street][i] / 100.0 * (pot + facing);
                na = a >= left ? add_size(amts, pcts, na, left, -1)
                               : add_size(amts, pcts, na, a, (int)lround(cfg.raise[street][i]));
            }
            if (cfg.allin)
                na = add_size(amts, pcts, na, left, -1);
            for (int i = 0; i < na; i++) {
                nc[0] = c[0];
                nc[1] = c[1];
                nc[player] += amts[i];
                if (pcts[i] < 0)
                    strcpy(labels[k], "A");
                else
                    snprintf(labels[k], 8, "R%d", pcts[i]);
                kids[k++] = build_action(board, nboard, nc, opp, nraises + 1, checked);
            }
        }
    }

    n->nchild = (uint8_t)k;
    n->child = malloc(k * sizeof(Node *));
    n->label = malloc(k * sizeof *n->label);
    memcpy(n->child, kids, k * sizeof(Node *));
    memcpy(n->label, labels, k * sizeof *n->label);
    n_action_nodes++;
    storage_floats += 2ull * k * nh[player];
    return n;
}

static void alloc_storage(Node *n)
{
    if (n->type == N_ACTION) {
        size_t sz = (size_t)n->nchild * nh[n->player];
        n->regret = calloc(sz, sizeof(float));
        n->ssum = calloc(sz, sizeof(float));
    }
    if (n->type == N_ACTION || n->type == N_CHANCE)
        for (int i = 0; i < n->nchild; i++)
            alloc_storage(n->child[i]);
}

/* ------------------------------------------------------------------------ */
/* Strategies                                                                */
/* ------------------------------------------------------------------------ */

/* Normalizes src (clamped at 0 when clamp is set) per hand into s; uniform
   where the sum is 0. Loops run action-major so they vectorize. */
static void normalize(const float *src, int A, int H, int clamp, float *s)
{
    float sum[H];
    for (int h = 0; h < H; h++)
        sum[h] = 0;
    for (int a = 0; a < A; a++)
        for (int h = 0; h < H; h++) {
            float x = src[a * H + h];
            s[a * H + h] = clamp && x < 0 ? 0 : x;
            sum[h] += s[a * H + h];
        }
    for (int h = 0; h < H; h++)
        sum[h] = sum[h] > 0 ? 1.0f / sum[h] : 0;
    for (int a = 0; a < A; a++)
        for (int h = 0; h < H; h++)
            s[a * H + h] = sum[h] > 0 ? s[a * H + h] * sum[h] : 1.0f / A;
}

static void current_strategy(const Node *n, float *s)
{
    normalize(n->regret, n->nchild, nh[n->player], 1, s);
}

static void average_strategy(const Node *n, float *s)
{
    normalize(n->ssum, n->nchild, nh[n->player], 0, s);
}

/* ------------------------------------------------------------------------ */
/* Terminal values                                                           */
/* ------------------------------------------------------------------------ */

/* compat[h] = total opponent reach over hands sharing no card with h */
static void compat_reach(int p, const float *ro, float *compat)
{
    int o = 1 - p;
    double total = 0, card[52] = {0};
    for (int j = 0; j < nh[o]; j++) {
        total += ro[j];
        card[hc1[o][j]] += ro[j];
        card[hc2[o][j]] += ro[j];
    }
    for (int h = 0; h < nh[p]; h++) {
        double v = total - card[hc1[p][h]] - card[hc2[p][h]];
        if (same[p][h] >= 0)
            v += ro[same[p][h]];
        compat[h] = (float)v;
    }
}

static void fold_value(const Node *n, int p, const float *ro, float *out)
{
    compat_reach(p, ro, out);
    double u = n->player == p ? -n->c[p] : cfg.pot0 + n->c[1 - p];
    for (int h = 0; h < nh[p]; h++)
        out[h] = (float)(out[h] * u);
}

/* One ascending sweep over both lists sorted by strength. For each group of
   my hands with equal strength, the opponent reach seen so far gives
   "beaten by me" (before adding the opponent's hands of that strength) and
   "not beating me" (after). Card removal: subtract the reach of opponent
   hands holding either of my cards, add back the identical combo, which got
   subtracted twice. At the end the running sums cover every opponent hand. */
static void showdown_value(const Node *n, int p, const float *ro, float *out)
{
    int H = nh[p], O = nh[1 - p];
    const SortedHand *mine = n->be->sorted[p], *theirs = n->be->sorted[1 - p];
    const int *sm = same[p];
    float win[NCOMBOS], le[NCOMBOS];

    double total = 0, card[52] = {0};
    int j = 0;
#define ADD_OPP()                         \
    do {                                  \
        const SortedHand *t = &theirs[j++]; \
        double r = ro[t->idx];            \
        total += r;                       \
        card[t->c1] += r;                 \
        card[t->c2] += r;                 \
    } while (0)
    for (int i = 0; i < H;) {
        uint16_t s = mine[i].str;
        int end = i;
        while (end < H && mine[end].str == s)
            end++;
        while (j < O && theirs[j].str < s)
            ADD_OPP();
        for (int k = i; k < end; k++)
            win[mine[k].idx] = (float)(total - card[mine[k].c1] - card[mine[k].c2]);
        while (j < O && theirs[j].str == s)
            ADD_OPP();
        for (int k = i; k < end; k++) {
            int h = mine[k].idx;
            le[h] = (float)(total - card[mine[k].c1] - card[mine[k].c2] + (sm[h] >= 0 ? ro[sm[h]] : 0));
        }
        i = end;
    }
    while (j < O)
        ADD_OPP();
#undef ADD_OPP

    double c = n->c[p], half = cfg.pot0 / 2;
    for (int h = 0; h < H; h++) {
        double compat = total - card[hc1[p][h]] - card[hc2[p][h]] + (sm[h] >= 0 ? ro[sm[h]] : 0);
        double lose = compat - le[h], tie = le[h] - win[h];
        out[h] = (float)((cfg.pot0 + c) * win[h] - c * lose + half * tie);
    }
}

/* ------------------------------------------------------------------------ */
/* Traversals                                                                */
/* ------------------------------------------------------------------------ */

enum { M_CFR, M_BR, M_EV };

typedef struct {
    int mode, p;
    float rpos, rneg, sdisc; /* DCFR discounts for this iteration */
} Pass;

static int nthreads = 1;

static void walk(const Pass *ps, Node *n, const float *rp, const float *ro, float *out, int par);

typedef struct {
    const Pass *ps;
    Node *n;
    const float *rp, *ro;
    float *vals;
    atomic_int next;
} ChanceJob;

static void chance_child(const Pass *ps, Node *n, int i, const float *rp, const float *ro, float *val)
{
    int p = ps->p, o = 1 - p, card = n->card[i];
    float rp2[NCOMBOS], ro2[NCOMBOS];
    for (int j = 0; j < nh[o]; j++)
        ro2[j] = hc1[o][j] == card || hc2[o][j] == card ? 0 : ro[j];
    if (rp)
        for (int h = 0; h < nh[p]; h++)
            rp2[h] = hc1[p][h] == card || hc2[p][h] == card ? 0 : rp[h];
    walk(ps, n->child[i], rp ? rp2 : NULL, ro2, val, 0);
}

static void *chance_worker(void *arg)
{
    ChanceJob *job = arg;
    int H = nh[job->ps->p];
    for (int i; (i = atomic_fetch_add(&job->next, 1)) < job->n->nchild;)
        chance_child(job->ps, job->n, i, job->rp, job->ro, job->vals + (size_t)i * H);
    return NULL;
}

/* Adds child i's values to out for every card whose canonical child is i. */
static void chance_accumulate(const Node *n, int p, int i, const float *v, float *out)
{
    for (int j = 0; j < n->ncards; j++) {
        if (n->map_child[j] != i)
            continue;
        int card = n->map_card[j];
        const int *hp = hperm[n->map_perm[j]][p];
        for (int h = 0; h < nh[p]; h++)
            if (hc1[p][h] != card && hc2[p][h] != card)
                out[h] += v[hp[h]];
    }
}

static void walk_chance(const Pass *ps, Node *n, const float *rp, const float *ro, float *out, int par)
{
    int p = ps->p, H = nh[p], K = n->nchild;
    memset(out, 0, H * sizeof(float));

    if (par && nthreads > 1) {
        ChanceJob job = {ps, n, rp, ro, malloc((size_t)K * H * sizeof(float)), 0};
        pthread_t tid[64];
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 64 << 20);
        int nt = nthreads < K ? nthreads : K;
        for (int t = 0; t < nt; t++)
            pthread_create(&tid[t], &attr, chance_worker, &job);
        for (int t = 0; t < nt; t++)
            pthread_join(tid[t], NULL);
        pthread_attr_destroy(&attr);
        for (int i = 0; i < K; i++)
            chance_accumulate(n, p, i, job.vals + (size_t)i * H, out);
        free(job.vals);
    } else {
        float v[NCOMBOS];
        for (int i = 0; i < K; i++) {
            chance_child(ps, n, i, rp, ro, v);
            chance_accumulate(n, p, i, v, out);
        }
    }
    float scale = 1.0f / (n->ncards - 4);
    for (int h = 0; h < H; h++)
        out[h] *= scale;
}

static void walk(const Pass *ps, Node *n, const float *rp, const float *ro, float *out, int par)
{
    int p = ps->p;
    switch (n->type) {
    case N_FOLD:
        fold_value(n, p, ro, out);
        return;
    case N_SHOWDOWN:
        showdown_value(n, p, ro, out);
        return;
    case N_CHANCE:
        walk_chance(ps, n, rp, ro, out, par);
        return;
    }

    int A = n->nchild, q = n->player, H = nh[q], P = nh[p];
    float strat[A * H], vals[A * (H > P ? H : P)], next[H];
    if (ps->mode == M_CFR)
        current_strategy(n, strat);
    else
        average_strategy(n, strat);

    if (q != p) {
        memset(out, 0, P * sizeof(float));
        for (int a = 0; a < A; a++) {
            float any = 0;
            for (int h = 0; h < H; h++) {
                next[h] = ro[h] * strat[a * H + h];
                any += next[h];
            }
            /* The subtree is worth 0. CFR still walks it to accumulate the
               traverser's average strategy there. */
            if (any == 0 && ps->mode != M_CFR)
                continue;
            walk(ps, n->child[a], rp, next, vals, par);
            for (int h = 0; h < P; h++)
                out[h] += vals[h];
        }
        return;
    }

    for (int a = 0; a < A; a++) {
        if (rp)
            for (int h = 0; h < H; h++)
                next[h] = rp[h] * strat[a * H + h];
        walk(ps, n->child[a], rp ? next : NULL, ro, vals + a * H, par);
    }

    if (ps->mode == M_BR) {
        for (int h = 0; h < H; h++) {
            float best = vals[h];
            for (int a = 1; a < A; a++)
                if (vals[a * H + h] > best)
                    best = vals[a * H + h];
            out[h] = best;
        }
        return;
    }

    for (int h = 0; h < H; h++)
        out[h] = 0;
    for (int a = 0; a < A; a++)
        for (int h = 0; h < H; h++)
            out[h] += strat[a * H + h] * vals[a * H + h];
    if (ps->mode == M_EV)
        return;

    for (int a = 0; a < A; a++)
        for (int h = 0; h < H; h++) {
            float *r = &n->regret[a * H + h];
            *r = *r * (*r > 0 ? ps->rpos : ps->rneg) + vals[a * H + h] - out[h];
            float *s = &n->ssum[a * H + h];
            *s = *s * ps->sdisc + rp[h] * strat[a * H + h];
        }
}

/* ------------------------------------------------------------------------ */
/* Solve loop                                                                */
/* ------------------------------------------------------------------------ */

static double total_mass; /* sum over compatible hand pairs of w0 * w1 */

/* Expected value for player p in chips (root-relative), under mode M_BR or M_EV. */
static double root_value(Node *root, int mode, int p)
{
    Pass ps = {mode, p, 0, 0, 0};
    float out[NCOMBOS];
    walk(&ps, root, NULL, hweight[1 - p], out, 1);
    double v = 0;
    for (int h = 0; h < nh[p]; h++)
        v += (double)hweight[p][h] * out[h];
    return v / total_mass;
}

static double exploitability(Node *root, double *br0, double *br1)
{
    *br0 = root_value(root, M_BR, 0);
    *br1 = root_value(root, M_BR, 1);
    return (*br0 + *br1 - cfg.pot0) / 2;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void solve(Node *root, int iters, double target_pct, int check_every)
{
    const double alpha = 1.5, beta = 0, gamma = 2;
    double t0 = now();
    float out[NCOMBOS];
    for (int t = 1; t <= iters; t++) {
        /* discounts applied to sums accumulated through iteration t-1 */
        double ta = pow(t - 1, alpha), tb = pow(t - 1, beta);
        Pass ps = {M_CFR, 0,
                   (float)(ta / (ta + 1)),
                   (float)(t == 1 ? 0 : tb / (tb + 1)),
                   (float)pow((double)(t - 1) / t, gamma)};
        for (int p = 0; p < 2; p++) {
            ps.p = p;
            walk(&ps, root, hweight[p], hweight[1 - p], out, 1);
        }
        if (t % check_every == 0 || t == iters) {
            double b0, b1, e = exploitability(root, &b0, &b1);
            double pct = 100 * e / cfg.pot0;
            printf("iter %5d  exploitability %.4f chips (%.3f%% of pot)  %.1fs\n", t, e, pct, now() - t0);
            fflush(stdout);
            if (pct <= target_pct)
                break;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Output                                                                    */
/* ------------------------------------------------------------------------ */

static void print_board(const int *b, int n)
{
    char s[3];
    for (int i = 0; i < n; i++) {
        card_str(b[i], s);
        printf("%s%s", s, i + 1 < n ? " " : "");
    }
}

/* from_tree maps the tree's suits back to the real ones (see show) */
static void hand_str(int p, int h, const int *from_tree, char *s)
{
    int a = (hc1[p][h] & ~3) | from_tree[hc1[p][h] & 3];
    int b = (hc2[p][h] & ~3) | from_tree[hc2[p][h] & 3];
    card_str(a > b ? a : b, s);
    card_str(a > b ? b : a, s + 2);
}

typedef struct {
    int h;
    float ev;
} HandEV;

static int cmp_ev(const void *a, const void *b)
{
    float x = ((const HandEV *)a)->ev, y = ((const HandEV *)b)->ev;
    return (x < y) - (x > y);
}

/* Walks the path (action labels / cards), then prints the node's strategy
   and per-action EVs for the acting player. A card that was merged into an
   isomorphic one is followed through the canonical subtree; to_tree tracks
   the suit relabeling from the real cards to the tree's. */
static int show(Node *root, const char *path, int combos_limit)
{
    float reach[2][NCOMBOS];
    memcpy(reach[0], hweight[0], nh[0] * sizeof(float));
    memcpy(reach[1], hweight[1], nh[1] * sizeof(float));

    Node *n = root;
    int to_tree[4] = {0, 1, 2, 3}, from_tree[4], real_board[5], nreal = nboard0;
    memcpy(real_board, board0, sizeof real_board);
    char *buf = strdup(path ? path : ""), *save = NULL;
    for (char *tok = strtok_r(buf, " ,/", &save); tok; tok = strtok_r(NULL, " ,/", &save)) {
        int found = -1;
        if (n->type == N_ACTION) {
            for (int i = 0; i < n->nchild && found < 0; i++)
                if (!strcasecmp(n->label[i], tok))
                    found = i;
        } else if (n->type == N_CHANCE && strlen(tok) == 2 && parse_card(tok) >= 0) {
            int real = parse_card(tok), tree = (real & ~3) | to_tree[real & 3];
            for (int j = 0; j < n->ncards && found < 0; j++)
                if (n->map_card[j] == tree) {
                    found = n->map_child[j];
                    for (int s = 0; s < 4; s++)
                        to_tree[s] = suit_perm[n->map_perm[j]][to_tree[s]];
                    real_board[nreal++] = real;
                }
        }
        if (n->type == N_ACTION && found >= 0) {
            float s[MAX_ACTIONS * NCOMBOS];
            average_strategy(n, s);
            int q = n->player;
            for (int h = 0; h < nh[q]; h++)
                reach[q][h] *= s[found * nh[q] + h];
        } else if (n->type == N_CHANCE && found >= 0) {
            int card = n->card[found];
            for (int p = 0; p < 2; p++)
                for (int h = 0; h < nh[p]; h++)
                    if (hc1[p][h] == card || hc2[p][h] == card)
                        reach[p][h] = 0;
        } else {
            fprintf(stderr, "path token '%s' not found here; choices:", tok);
            if (n->type == N_ACTION)
                for (int i = 0; i < n->nchild; i++)
                    fprintf(stderr, " %s", n->label[i]);
            else
                fprintf(stderr, " %s", n->type == N_CHANCE ? "a card not on the board" : "none (terminal)");
            fprintf(stderr, "\n");
            free(buf);
            return -1;
        }
        n = n->child[found];
    }
    free(buf);
    for (int s = 0; s < 4; s++)
        from_tree[to_tree[s]] = s;

    printf("\n== %s ==\nboard: ", path && *path ? path : "root");
    print_board(real_board, nreal);
    printf("   pot %.2f   OOP in %.2f, IP in %.2f\n", cfg.pot0 + n->c[0] + n->c[1], n->c[0], n->c[1]);
    if (n->type != N_ACTION) {
        printf("(%s)\n", n->type == N_CHANCE ? "chance node: add a card to the path" : "terminal");
        return 0;
    }

    int q = n->player, A = n->nchild, H = nh[q];
    float s[MAX_ACTIONS * NCOMBOS], vals[MAX_ACTIONS * NCOMBOS], norm[NCOMBOS];
    average_strategy(n, s);
    Pass ps = {M_EV, q, 0, 0, 0};
    for (int a = 0; a < A; a++)
        walk(&ps, n->child[a], NULL, reach[1 - q], vals + a * H, 1);
    compat_reach(q, reach[1 - q], norm);

    /* aggregate frequencies weighted by the actor's reach */
    double tot = 0, freq[MAX_ACTIONS] = {0}, ev_total = 0;
    HandEV *rows = malloc(H * sizeof *rows);
    int nrows = 0;
    for (int h = 0; h < H; h++) {
        double w = reach[q][h] * norm[h];
        if (w <= 0)
            continue;
        double ev = 0;
        for (int a = 0; a < A; a++) {
            freq[a] += w * s[a * H + h];
            ev += s[a * H + h] * (vals[a * H + h] / norm[h] + n->c[q]);
        }
        tot += w;
        ev_total += w * ev;
        rows[nrows++] = (HandEV){h, (float)ev};
    }
    printf("%s to act. Range EV %.3f.  Overall:", q ? "IP" : "OOP", tot > 0 ? ev_total / tot : 0);
    for (int a = 0; a < A; a++)
        printf("  %s %.1f%%", n->label[a], tot > 0 ? 100 * freq[a] / tot : 0);
    printf("\n");

    qsort(rows, nrows, sizeof *rows, cmp_ev);
    if (combos_limit != 0) {
        printf("\n%-5s %6s %8s ", "hand", "weight", "EV");
        for (int a = 0; a < A; a++)
            printf(" | %6s %7s", n->label[a], "EV");
        printf("\n");
        for (int i = 0; i < nrows && (combos_limit < 0 || i < combos_limit); i++) {
            int h = rows[i].h;
            char hs[5];
            hand_str(q, h, from_tree, hs);
            printf("%-5s %6.3f %8.3f ", hs, reach[q][h], rows[i].ev);
            for (int a = 0; a < A; a++)
                printf(" | %5.1f%% %7.3f", 100 * s[a * H + h], vals[a * H + h] / norm[h] + n->c[q]);
            printf("\n");
        }
        if (combos_limit > 0 && nrows > combos_limit)
            printf("... %d more (use --combos all)\n", nrows - combos_limit);
    }
    free(rows);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Main                                                                      */
/* ------------------------------------------------------------------------ */

static int parse_sizes(const char *s, float *out)
{
    int n = 0;
    char *buf = strdup(s), *save = NULL;
    for (char *t = strtok_r(buf, ", ", &save); t && n < MAX_SIZES; t = strtok_r(NULL, ", ", &save))
        out[n++] = strtof(t, NULL);
    free(buf);
    return n;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: solver --board CARDS --oop RANGE --ip RANGE --pot X --stack Y [options]\n"
        "  --board       flop, turn or river, e.g. 'Ks7h2d' or 'Ks 7h 2d 9c'\n"
        "  --oop/--ip    ranges, e.g. 'QQ+,AKs,AQo:0.5,KTs+,A5s-A2s,99-66,AhKh'\n"
        "  --pot         starting pot;  --stack  effective stack behind\n"
        "  --bet  P,..   bet sizes in %% of pot, all streets (default 33,75)\n"
        "  --raise P,..  raise sizes in %% of pot after calling (default 75)\n"
        "  --flop-bet/--turn-bet/--river-bet, --flop-raise/--turn-raise/--river-raise\n"
        "                per-street overrides ('' for none)\n"
        "  --no-allin    don't add an all-in option everywhere\n"
        "  --raise-cap N max raises per street (default 3)\n"
        "  --iters N     max iterations (default 1000)\n"
        "  --target P    stop at exploitability P%% of pot (default 0.3)\n"
        "  --check N     compute exploitability every N iterations (default 10)\n"
        "  --threads N   (default: all cores)\n"
        "  --no-iso      don't merge suit-isomorphic turn/river cards\n"
        "  --max-mem GB  refuse to build trees bigger than this (default 8)\n"
        "  --show PATH   node to print, e.g. 'X B75 C Td X' (repeatable; default root)\n"
        "  --combos N    combos to list per node, or 'all' (default 40)\n");
    exit(2);
}

int main(int argc, char **argv)
{
    init_combos();
    const char *board_s = NULL, *oop_s = NULL, *ip_s = NULL;
    const char *street_bet[3] = {0}, *street_raise[3] = {0};
    const char *bet_s = "33,75", *raise_s = "75";
    const char *shows[32];
    int nshow = 0, iso = 1, iters = 1000, check_every = 10, combos = 40;
    double target = 0.3, max_mem = 8;
    cfg.allin = 1;
    cfg.raise_cap = 3;
    cfg.pot0 = -1;
    cfg.stack = -1;
    nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);

    static struct option opts[] = {
        {"board", 1, 0, 'b'}, {"oop", 1, 0, 'o'}, {"ip", 1, 0, 'i'}, {"pot", 1, 0, 'p'},
        {"stack", 1, 0, 's'}, {"bet", 1, 0, 'B'}, {"raise", 1, 0, 'R'}, {"no-allin", 0, 0, 'n'},
        {"raise-cap", 1, 0, 'c'}, {"iters", 1, 0, 'I'}, {"target", 1, 0, 't'}, {"check", 1, 0, 'k'},
        {"threads", 1, 0, 'T'}, {"max-mem", 1, 0, 'm'}, {"show", 1, 0, 'S'}, {"combos", 1, 0, 'C'},
        {"flop-bet", 1, 0, 1}, {"turn-bet", 1, 0, 2}, {"river-bet", 1, 0, 3},
        {"flop-raise", 1, 0, 4}, {"turn-raise", 1, 0, 5}, {"river-raise", 1, 0, 6},
        {"no-iso", 0, 0, 7}, {"help", 0, 0, 'h'}, {0, 0, 0, 0}};
    for (int ch; (ch = getopt_long(argc, argv, "", opts, NULL)) != -1;) {
        switch (ch) {
        case 'b': board_s = optarg; break;
        case 'o': oop_s = optarg; break;
        case 'i': ip_s = optarg; break;
        case 'p': cfg.pot0 = atof(optarg); break;
        case 's': cfg.stack = atof(optarg); break;
        case 'B': bet_s = optarg; break;
        case 'R': raise_s = optarg; break;
        case 'n': cfg.allin = 0; break;
        case 'c': cfg.raise_cap = atoi(optarg); break;
        case 'I': iters = atoi(optarg); break;
        case 't': target = atof(optarg); break;
        case 'k': check_every = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'T': nthreads = atoi(optarg) > 0 ? atoi(optarg) : 1; break;
        case 'm': max_mem = atof(optarg); break;
        case 'S': if (nshow < 32) shows[nshow++] = optarg; break;
        case 'C': combos = !strcmp(optarg, "all") ? -1 : atoi(optarg); break;
        case 1: case 2: case 3: street_bet[ch - 1] = optarg; break;
        case 4: case 5: case 6: street_raise[ch - 4] = optarg; break;
        case 7: iso = 0; break;
        default: usage();
        }
    }
    if (!board_s || !oop_s || !ip_s || cfg.pot0 <= 0 || cfg.stack < 0)
        usage();
    if (nthreads > 64)
        nthreads = 64;
    for (int st = 0; st < 3; st++) {
        cfg.nbet[st] = parse_sizes(street_bet[st] ? street_bet[st] : bet_s, cfg.bet[st]);
        cfg.nraise[st] = parse_sizes(street_raise[st] ? street_raise[st] : raise_s, cfg.raise[st]);
    }

    /* board */
    uint64_t bmask = 0;
    for (const char *s = board_s; *s;) {
        if (*s == ' ' || *s == ',') { s++; continue; }
        int c = parse_card(s);
        if (c < 0 || nboard0 == 5 || (bmask >> c & 1)) {
            fprintf(stderr, "bad board '%s'\n", board_s);
            return 2;
        }
        bmask |= 1ull << c;
        board0[nboard0++] = c;
        s += 2;
    }
    if (nboard0 < 3) {
        fprintf(stderr, "board needs 3-5 cards\n");
        return 2;
    }

    /* hand lists */
    float w[2][NCOMBOS];
    if (parse_range(oop_s, w[0]) || parse_range(ip_s, w[1]))
        return 2;
    int index_of[2][NCOMBOS];
    for (int p = 0; p < 2; p++) {
        hc1[p] = malloc(NCOMBOS * sizeof(int));
        hc2[p] = malloc(NCOMBOS * sizeof(int));
        hmask[p] = malloc(NCOMBOS * sizeof(uint64_t));
        hweight[p] = malloc(NCOMBOS * sizeof(float));
        same[p] = malloc(NCOMBOS * sizeof(int));
        for (int k = 0; k < NCOMBOS; k++) {
            index_of[p][k] = -1;
            uint64_t m = 1ull << combo_c1[k] | 1ull << combo_c2[k];
            if (w[p][k] <= 0 || (m & bmask))
                continue;
            int h = nh[p]++;
            index_of[p][k] = h;
            hc1[p][h] = combo_c1[k];
            hc2[p][h] = combo_c2[k];
            hmask[p][h] = m;
            hweight[p][h] = w[p][k];
        }
        if (!nh[p]) {
            fprintf(stderr, "%s range is empty on this board\n", p ? "IP" : "OOP");
            return 2;
        }
    }
    for (int p = 0; p < 2; p++)
        for (int h = 0; h < nh[p]; h++)
            same[p][h] = index_of[1 - p][combo_id[hc1[p][h]][hc2[p][h]]];
    {
        float compat[NCOMBOS];
        compat_reach(0, hweight[1], compat);
        for (int h = 0; h < nh[0]; h++)
            total_mass += (double)hweight[0][h] * compat[h];
        if (total_mass <= 0) {
            fprintf(stderr, "no compatible hand pairs\n");
            return 2;
        }
    }

    init_suit_perms(iso);

    printf("board ");
    print_board(board0, nboard0);
    printf(" | pot %.2f stack %.2f | OOP %d combos, IP %d combos | %d threads\n",
           cfg.pot0, cfg.stack, nh[0], nh[1], nthreads);

    double c0[2] = {0, 0};
    double tb = now();
    Node *root = build_action(board0, nboard0, c0, 0, 0, 0);
    double gb = storage_floats * 4.0 / (1 << 30);
    printf("tree: %zu nodes, %zu decision nodes, %.2f GB of regrets+strategy (built in %.1fs)\n",
           n_nodes, n_action_nodes, gb, now() - tb);
    if (gb > max_mem) {
        fprintf(stderr, "tree too big for --max-mem %.1f; use fewer sizes, a lower --raise-cap, or a later street\n", max_mem);
        return 1;
    }
    alloc_storage(root);

    solve(root, iters, target, check_every);

    printf("\nEV at root (share of the %.2f pot): OOP %.3f, IP %.3f\n", cfg.pot0,
           root_value(root, M_EV, 0), root_value(root, M_EV, 1));
    if (!nshow)
        shows[nshow++] = "";
    for (int i = 0; i < nshow; i++)
        show(root, shows[i], combos);
    return 0;
}
