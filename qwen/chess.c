/*
 * Text-mode chess: complete rules + alpha/beta search engine.
 * Single self-contained source file, C11, standard library only.
 *
 * Build: cc -O3 -std=c11 -o chess chess.c
 */

#if defined(__unix__) || defined(__APPLE__)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* basic types                                                         */
/* ------------------------------------------------------------------ */

enum { PS_EMPTY = 0 };
enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { WHITE = 0, BLACK = 1 };

/* named piece codes (colour << 3 | type) */
enum {
    W_PAWN_ = 1, W_KNIGHT_, W_BISHOP_, W_ROOK_, W_QUEEN_, W_KING_,
    B_PAWN_ = 9, B_KNIGHT_, B_BISHOP_, B_ROOK_, B_QUEEN_, B_KING_
};

#define P_COLOR(p) ((p) >> 3)
#define P_TYPE(p) ((p) & 7)
#define MAKE_PIECE(c, t) ((int8_t)(((c) << 3) | (t)))

/* 0x88 board: rank 0 = rank 1 (white home), file 0 = a */
#define SQ(r, f) (((r) << 4) | (f))
#define SQ_RANK(s) ((s) >> 4)
#define SQ_FILE(s) ((s) & 15)
#define OFF_88(s) ((s) & 0x88)

enum { WK_CASTLE = 1, WQ_CASTLE = 2, BK_CASTLE = 4, BQ_CASTLE = 8 };

enum { M_NORMAL = 0, M_DOUBLE = 1, M_CASTLE = 2, M_EP = 3 };

/*
 * Move encoding. 0x88 squares need 7 bits each, so:
 *   bits 0..6 from, bits 7..13 to, bits 14..16 promotion, bits 17..19 flag
 */
#define MAKE_MOVE(f, t, fl, pr)                                                  \
    ((int)(f) | ((int)(t) << 7) | ((int)(pr) << 14) | ((int)(fl) << 17))
#define MOVE_FROM(m) ((m) & 0x7f)
#define MOVE_TO(m) (((m) >> 7) & 0x7f)
#define MOVE_PROMO(m) (((m) >> 14) & 7)
#define MOVE_FLAG(m) (((m) >> 17) & 7)

#define MAX_PLY 80
#define MAX_MOVES 384 /* pseudo-legal lists must never overflow this */
#define INF 30000
#define MATE 29000
#define MATE_BOUND (MATE - 1000)
#define NO_EVAL 0x7fffffff
#define START_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

static const int PieceVal[7] = { 0, 100, 305, 330, 563, 936, 0 };

static const int KnightOff[8] = { -18, -33, -31, -14, 18, 33, 31, 14 };
static const int KingOff[8] = { -17, -16, -15, -1, 1, 15, 16, 17 };
static const int BishopOff[4] = { -17, -15, 15, 17 };
static const int RookOff[4] = { -16, -1, 1, 16 };
static const int QueenOff[8] = { -17, -15, 15, 17, -16, -1, 1, 16 };

/* ------------------------------------------------------------------ */
/* position                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int8_t sq[128];
    int side;
    int castling;
    int ep; /* en passant target square, -1 when none */
    int halfmove;
    int moveNumber;
    uint64_t key;
    int kingSq[2];
    int pieceCount[16];
    int colorCount[2];
    int colorSq[2][32];
    int sqIndex[128];
} Pos;

typedef struct {
    int move;
    int captured;
    int castling;
    int ep;
    int halfmove;
    uint64_t key;
} Undo;

static uint64_t zPiece[16][128];
static uint64_t zSide;
static uint64_t zCastle[16];
static uint64_t zEp[8];
static int castleMask[128];

#define EPI_KEY(e) ((e) < 0 ? (uint64_t)0 : zEp[SQ_FILE(e)])

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void initTables(void)
{
    uint64_t s = 0x123456789abcdefULL;
    for (int p = 0; p < 16; p++)
        for (int i = 0; i < 128; i++)
            zPiece[p][i] = splitmix64(&s);
    zSide = splitmix64(&s);
    for (int i = 0; i < 16; i++)
        zCastle[i] = splitmix64(&s);
    for (int i = 0; i < 8; i++)
        zEp[i] = splitmix64(&s);

    for (int i = 0; i < 128; i++)
        castleMask[i] = 15;
    castleMask[SQ(0, 0)] = 15 & ~WQ_CASTLE;
    castleMask[SQ(0, 7)] = 15 & ~WK_CASTLE;
    castleMask[SQ(0, 4)] = 15 & ~(WK_CASTLE | WQ_CASTLE);
    castleMask[SQ(7, 0)] = 15 & ~BQ_CASTLE;
    castleMask[SQ(7, 7)] = 15 & ~BK_CASTLE;
    castleMask[SQ(7, 4)] = 15 & ~(BK_CASTLE | BQ_CASTLE);
}

static void addPiece(Pos *p, int s, int piece)
{
    int c = P_COLOR(piece);
    int idx = p->colorCount[c]++;
    p->sq[s] = (int8_t)piece;
    p->colorSq[c][idx] = s;
    p->sqIndex[s] = idx;
    p->pieceCount[piece]++;
    p->key ^= zPiece[piece][s];
    if (P_TYPE(piece) == KING)
        p->kingSq[c] = s;
}

static void delPiece(Pos *p, int s)
{
    int piece = p->sq[s];
    int c = P_COLOR(piece);
    int idx = p->sqIndex[s];
    int last = --p->colorCount[c];
    int moved = p->colorSq[c][last];
    p->colorSq[c][idx] = moved;
    p->sqIndex[moved] = idx;
    p->sq[s] = PS_EMPTY;
    p->pieceCount[piece]--;
    p->key ^= zPiece[piece][s];
}

static void clearPos(Pos *p)
{
    memset(p, 0, sizeof *p);
    for (int i = 0; i < 128; i++)
        p->sq[i] = PS_EMPTY;
    p->ep = -1;
    p->moveNumber = 1;
}

static int charToType(int c)
{
    switch (tolower(c)) {
    case 'p': return PAWN;
    case 'n': return KNIGHT;
    case 'b': return BISHOP;
    case 'r': return ROOK;
    case 'q': return QUEEN;
    case 'k': return KING;
    default: return 0;
    }
}

static int setFen(Pos *p, const char *fen)
{
    clearPos(p);
    int r = 7, f = 0;
    const char *s = fen;

    while (*s && *s != ' ') {
        if (*s == '/') {
            r--;
            f = 0;
            if (r < 0)
                return 0;
        } else if (isdigit((unsigned char)*s)) {
            f += *s - '0';
            if (f > 8)
                return 0;
        } else {
            int t = charToType(*s);
            if (!t || r < 0 || r > 7 || f < 0 || f > 7)
                return 0;
            addPiece(p, SQ(r, f),
                     MAKE_PIECE(islower((unsigned char)*s) ? BLACK : WHITE, t));
            f++;
        }
        s++;
    }
    if (p->pieceCount[W_KING_] != 1 || p->pieceCount[B_KING_] != 1)
        return 0;

    while (*s == ' ')
        s++;
    if (*s == 'w')
        p->side = WHITE;
    else if (*s == 'b')
        p->side = BLACK;
    else
        return 0;
    s++;

    while (*s == ' ')
        s++;
    int rights = 0;
    if (*s == '-') {
        s++;
    } else {
        while (*s && *s != ' ') {
            switch (*s) {
            case 'K': rights |= WK_CASTLE; break;
            case 'Q': rights |= WQ_CASTLE; break;
            case 'k': rights |= BK_CASTLE; break;
            case 'q': rights |= BQ_CASTLE; break;
            default: return 0;
            }
            s++;
        }
    }
    p->castling = rights;

    while (*s == ' ')
        s++;
    p->ep = -1;
    if (*s != '-') {
        if (!isalpha((unsigned char)*s) || !isdigit((unsigned char)s[1]))
            return 0;
        int ff = tolower(*s) - 'a';
        int rr = s[1] - '1';
        if (ff < 0 || ff > 7 || rr < 0 || rr > 7)
            return 0;
        p->ep = SQ(rr, ff);
        s += 2;
    } else {
        s++;
    }

    while (*s == ' ')
        s++;
    int hm = 0;
    while (isdigit((unsigned char)*s))
        hm = hm * 10 + (*s++ - '0');
    p->halfmove = hm;

    while (*s == ' ')
        s++;
    int mn = 1;
    if (isdigit((unsigned char)*s)) {
        mn = 0;
        while (isdigit((unsigned char)*s))
            mn = mn * 10 + (*s++ - '0');
    }
    p->moveNumber = mn;

    p->key ^= zCastle[p->castling] ^ EPI_KEY(p->ep);
    if (p->side == BLACK)
        p->key ^= zSide;
    return 1;
}

/* ------------------------------------------------------------------ */
/* key stack (repetition detection)                                    */
/* ------------------------------------------------------------------ */

#define KEY_STACK 4096
static uint64_t keyStack[KEY_STACK];
static int keyLen;

static void pushKey(uint64_t k)
{
    if (keyLen < KEY_STACK)
        keyStack[keyLen++] = k;
}

/* ------------------------------------------------------------------ */
/* make / unmake                                                       */
/* ------------------------------------------------------------------ */

static void castleRookSquares(int to, int *rookFrom, int *rookTo)
{
    *rookFrom = *rookTo = -1;
    if (to == SQ(0, 6)) { *rookFrom = SQ(0, 7); *rookTo = SQ(0, 5); }
    else if (to == SQ(0, 2)) { *rookFrom = SQ(0, 0); *rookTo = SQ(0, 3); }
    else if (to == SQ(7, 6)) { *rookFrom = SQ(7, 7); *rookTo = SQ(7, 5); }
    else if (to == SQ(7, 2)) { *rookFrom = SQ(7, 0); *rookTo = SQ(7, 3); }
}

static void makeMove(Pos *p, int m, Undo *u)
{
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    int flag = MOVE_FLAG(m), promo = MOVE_PROMO(m);
    int piece = p->sq[from];
    int mover = p->side;
    int oldCastle = p->castling, oldEp = p->ep;

    u->move = m;
    u->castling = oldCastle;
    u->ep = oldEp;
    u->halfmove = p->halfmove;
    u->key = p->key;

    if (flag == M_EP) {
        int capSq = (mover == WHITE) ? to - 16 : to + 16;
        u->captured = p->sq[capSq];
        delPiece(p, capSq);
    } else {
        u->captured = p->sq[to];
        if (u->captured)
            delPiece(p, to);
    }

    delPiece(p, from);
    addPiece(p, to, promo ? MAKE_PIECE(mover, promo) : piece);

    if (flag == M_CASTLE) {
        int rf, rt;
        castleRookSquares(to, &rf, &rt);
        if (rf >= 0) {
            int rp = p->sq[rf];
            delPiece(p, rf);
            addPiece(p, rt, rp);
        }
    }

    p->ep = (flag == M_DOUBLE) ? ((mover == WHITE) ? to - 16 : to + 16) : -1;
    p->castling &= castleMask[from] & castleMask[to];

    if (P_TYPE(piece) == PAWN || u->captured)
        p->halfmove = 0;
    else
        p->halfmove++;

    p->key ^= zCastle[oldCastle] ^ zCastle[p->castling];
    p->key ^= EPI_KEY(oldEp) ^ EPI_KEY(p->ep);

    p->side ^= 1;
    p->key ^= zSide;
    if (mover == BLACK)
        p->moveNumber++;

    pushKey(p->key);
}

static void unmakeMove(Pos *p, const Undo *u)
{
    int m = u->move;
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    int flag = MOVE_FLAG(m), promo = MOVE_PROMO(m);
    int mover = 1 - p->side;
    int atTo = p->sq[to];
    int origPiece = MAKE_PIECE(mover, promo ? PAWN : P_TYPE(atTo));

    delPiece(p, to);
    addPiece(p, from, origPiece);

    if (flag == M_CASTLE) {
        int rf, rt;
        castleRookSquares(to, &rf, &rt);
        if (rt >= 0) {
            int rp = p->sq[rt];
            delPiece(p, rt);
            addPiece(p, rf, rp);
        }
    }

    if (u->captured) {
        int capSq = (flag == M_EP) ? ((mover == WHITE) ? to - 16 : to + 16) : to;
        addPiece(p, capSq, u->captured);
    }

    p->castling = u->castling;
    p->ep = u->ep;
    p->halfmove = u->halfmove;
    p->key = u->key;
    p->side = mover;
    if (mover == BLACK)
        p->moveNumber--;

    if (keyLen > 0)
        keyLen--;
}

static void doNullMove(Pos *p, Undo *u)
{
    u->move = 0;
    u->captured = 0;
    u->castling = p->castling;
    u->ep = p->ep;
    u->halfmove = p->halfmove;
    u->key = p->key;
    p->ep = -1;
    p->key ^= zSide;
    p->side ^= 1;
    if (p->side == WHITE)
        p->moveNumber++;
    pushKey(p->key);
}

static void undoNullMove(Pos *p, const Undo *u)
{
    int mover = 1 - p->side;
    p->ep = u->ep;
    p->castling = u->castling;
    p->halfmove = u->halfmove;
    p->key = u->key;
    p->side = mover;
    if (mover == BLACK)
        p->moveNumber--;
    if (keyLen > 0)
        keyLen--;
}

/* ------------------------------------------------------------------ */
/* attacks                                                             */
/* ------------------------------------------------------------------ */

static int isAttacked(const Pos *p, int s, int by)
{
    int r = SQ_RANK(s), f = SQ_FILE(s);

    if (by == WHITE) {
        /* a white pawn sits one rank below s: s-15 = (rank-1,file+1), s-17 = (rank-1,file-1) */
        if (r >= 1) {
            if (f <= 6 && p->sq[s - 15] == W_PAWN_)
                return 1;
            if (f >= 1 && p->sq[s - 17] == W_PAWN_)
                return 1;
        }
    } else {
        /* a black pawn sits one rank above s: s+15 = (rank+1,file-1), s+17 = (rank+1,file+1) */
        if (r <= 6) {
            if (f >= 1 && p->sq[s + 15] == B_PAWN_)
                return 1;
            if (f <= 6 && p->sq[s + 17] == B_PAWN_)
                return 1;
        }
    }

    for (int i = 0; i < 8; i++) {
        int t = s + KnightOff[i];
        if (!OFF_88(t) && p->sq[t] == MAKE_PIECE(by, KNIGHT))
            return 1;
    }
    for (int i = 0; i < 8; i++) {
        int t = s + KingOff[i];
        if (!OFF_88(t) && p->sq[t] == MAKE_PIECE(by, KING))
            return 1;
    }
    for (int i = 0; i < 4; i++) {
        int t = s + BishopOff[i];
        while (!OFF_88(t)) {
            int q = p->sq[t];
            if (q) {
                if (P_COLOR(q) == by && (P_TYPE(q) == BISHOP || P_TYPE(q) == QUEEN))
                    return 1;
                break;
            }
            t += BishopOff[i];
        }
    }
    for (int i = 0; i < 4; i++) {
        int t = s + RookOff[i];
        while (!OFF_88(t)) {
            int q = p->sq[t];
            if (q) {
                if (P_COLOR(q) == by && (P_TYPE(q) == ROOK || P_TYPE(q) == QUEEN))
                    return 1;
                break;
            }
            t += RookOff[i];
        }
    }
    return 0;
}

static int inCheckOf(const Pos *p, int side)
{
    return isAttacked(p, p->kingSq[side], 1 - side);
}

/* ------------------------------------------------------------------ */
/* move generation                                                     */
/* ------------------------------------------------------------------ */

static int generateMoves(const Pos *p, int *m, int capsOnly)
{
    int n = 0;
    int side = p->side;
    int count = p->colorCount[side];

    for (int i = 0; i < count; i++) {
        int from = p->colorSq[side][i];
        int piece = p->sq[from];
        int type = P_TYPE(piece);

        if (type == PAWN) {
            int fwd = (side == WHITE) ? 16 : -16;
            int startRank = (side == WHITE) ? 1 : 6;
            int promoRank = (side == WHITE) ? 7 : 0;
            int t = from + fwd;

            if (!OFF_88(t) && p->sq[t] == PS_EMPTY) {
                if (SQ_RANK(t) == promoRank) {
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, QUEEN);
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, ROOK);
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, BISHOP);
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, KNIGHT);
                } else if (!capsOnly) {
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, 0);
                    if (SQ_RANK(from) == startRank && p->sq[from + 2 * fwd] == PS_EMPTY)
                        m[n++] = MAKE_MOVE(from, from + 2 * fwd, M_DOUBLE, 0);
                }
            }

            int f = SQ_FILE(from);
            int cand[2], nc = 0;
            if (side == WHITE) {
                if (f >= 1) cand[nc++] = from + 15; /* file - 1 */
                if (f <= 6) cand[nc++] = from + 17; /* file + 1 */
            } else {
                if (f <= 6) cand[nc++] = from - 15; /* file + 1 */
                if (f >= 1) cand[nc++] = from - 17; /* file - 1 */
            }
            for (int k = 0; k < nc; k++) {
                int t2 = cand[k];
                if (OFF_88(t2))
                    continue;
                int occ = p->sq[t2];
                if (occ != PS_EMPTY && P_COLOR(occ) != side) {
                    if (SQ_RANK(t2) == promoRank) {
                        m[n++] = MAKE_MOVE(from, t2, M_NORMAL, QUEEN);
                        m[n++] = MAKE_MOVE(from, t2, M_NORMAL, ROOK);
                        m[n++] = MAKE_MOVE(from, t2, M_NORMAL, BISHOP);
                        m[n++] = MAKE_MOVE(from, t2, M_NORMAL, KNIGHT);
                    } else {
                        m[n++] = MAKE_MOVE(from, t2, M_NORMAL, 0);
                    }
                } else if (t2 == p->ep) {
                    m[n++] = MAKE_MOVE(from, t2, M_EP, 0);
                }
            }
            continue;
        }

        if (type == KNIGHT || type == KING) {
            const int *off = (type == KNIGHT) ? KnightOff : KingOff;
            for (int k = 0; k < 8; k++) {
                int t = from + off[k];
                if (OFF_88(t))
                    continue;
                int occ = p->sq[t];
                if (occ == PS_EMPTY) {
                    if (!capsOnly)
                        m[n++] = MAKE_MOVE(from, t, M_NORMAL, 0);
                } else if (P_COLOR(occ) != side) {
                    m[n++] = MAKE_MOVE(from, t, M_NORMAL, 0);
                }
            }
            if (type == KING && !capsOnly) {
                int rank = (side == WHITE) ? 0 : 7;
                int ks = (side == WHITE) ? WK_CASTLE : BK_CASTLE;
                int qs = (side == WHITE) ? WQ_CASTLE : BQ_CASTLE;
                int ksq = SQ(rank, 4);
                if ((p->castling & ks) && !inCheckOf(p, side) &&
                    p->sq[SQ(rank, 5)] == PS_EMPTY && p->sq[SQ(rank, 6)] == PS_EMPTY &&
                    !isAttacked(p, SQ(rank, 5), 1 - side) &&
                    !isAttacked(p, SQ(rank, 6), 1 - side) &&
                    p->sq[SQ(rank, 7)] == MAKE_PIECE(side, ROOK)) {
                    m[n++] = MAKE_MOVE(ksq, SQ(rank, 6), M_CASTLE, 0);
                }
                if ((p->castling & qs) && !inCheckOf(p, side) &&
                    p->sq[SQ(rank, 3)] == PS_EMPTY && p->sq[SQ(rank, 2)] == PS_EMPTY &&
                    p->sq[SQ(rank, 1)] == PS_EMPTY &&
                    !isAttacked(p, SQ(rank, 3), 1 - side) &&
                    !isAttacked(p, SQ(rank, 2), 1 - side) &&
                    p->sq[SQ(rank, 0)] == MAKE_PIECE(side, ROOK)) {
                    m[n++] = MAKE_MOVE(ksq, SQ(rank, 2), M_CASTLE, 0);
                }
            }
            continue;
        }

        const int *dirs;
        int nd;
        if (type == BISHOP) {
            dirs = BishopOff;
            nd = 4;
        } else if (type == ROOK) {
            dirs = RookOff;
            nd = 4;
        } else {
            dirs = QueenOff;
            nd = 8;
        }
        for (int d = 0; d < nd; d++) {
            int delta = dirs[d];
            int t = from + delta;
            while (!OFF_88(t)) {
                int occ = p->sq[t];
                if (occ == PS_EMPTY) {
                    if (!capsOnly)
                        m[n++] = MAKE_MOVE(from, t, M_NORMAL, 0);
                } else {
                    if (P_COLOR(occ) != side)
                        m[n++] = MAKE_MOVE(from, t, M_NORMAL, 0);
                    break;
                }
                t += delta;
            }
        }
    }
    return n;
}

/* pseudo-legal -> legal filter; returns the number of legal moves */
static int generateLegal(Pos *p, int *m)
{
    int n = generateMoves(p, m, 0);
    int legal = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        int side = p->side;
        makeMove(p, m[i], &u);
        if (!isAttacked(p, p->kingSq[side], 1 - side))
            m[legal++] = m[i];
        unmakeMove(p, &u);
    }
    return legal;
}

/* ------------------------------------------------------------------ */
/* static exchange evaluation                                          */
/* ------------------------------------------------------------------ */

/* least valuable attacker of s by colour `by`; removes it from occ */
static int seeLeast(int8_t *occ, int s, int by, int *outFrom)
{
    int r = SQ_RANK(s), f = SQ_FILE(s);

    if (by == WHITE) {
        if (r >= 1) {
            if (f <= 6 && occ[s - 15] == W_PAWN_) {
                *outFrom = s - 15;
                occ[s - 15] = PS_EMPTY;
                return PAWN;
            }
            if (f >= 1 && occ[s - 17] == W_PAWN_) {
                *outFrom = s - 17;
                occ[s - 17] = PS_EMPTY;
                return PAWN;
            }
        }
    } else {
        if (r <= 6) {
            if (f >= 1 && occ[s + 15] == B_PAWN_) {
                *outFrom = s + 15;
                occ[s + 15] = PS_EMPTY;
                return PAWN;
            }
            if (f <= 6 && occ[s + 17] == B_PAWN_) {
                *outFrom = s + 17;
                occ[s + 17] = PS_EMPTY;
                return PAWN;
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        int t = s + KnightOff[i];
        if (!OFF_88(t) && occ[t] == MAKE_PIECE(by, KNIGHT)) {
            *outFrom = t;
            occ[t] = PS_EMPTY;
            return KNIGHT;
        }
    }

    for (int d = 0; d < 8; d++) {
        int delta = (d < 4) ? BishopOff[d] : RookOff[d - 4];
        int diag = (d < 4);
        int t = s + delta;
        while (!OFF_88(t)) {
            int q = occ[t];
            if (q) {
                if (P_COLOR(q) == by) {
                    int ty = P_TYPE(q);
                    if (ty == KING || ty == QUEEN || (diag && ty == BISHOP) ||
                        (!diag && ty == ROOK)) {
                        *outFrom = t;
                        occ[t] = PS_EMPTY;
                        return ty;
                    }
                }
                break;
            }
            t += delta;
        }
    }
    return 0;
}

/* 1 when the swap sequence on the target square gains at least `threshold` */
static int seeGe(const Pos *p, int move, int threshold)
{
    int from = MOVE_FROM(move), to = MOVE_TO(move);
    int flag = MOVE_FLAG(move), promo = MOVE_PROMO(move);
    int mover = p->sq[from];
    if (mover == PS_EMPTY)
        return 0;
    int color = P_COLOR(mover);

    int8_t occ[128];
    memcpy(occ, p->sq, 128);

    int victim = (flag == M_EP) ? MAKE_PIECE(1 - color, PAWN) : occ[to];
    int swap = PieceVal[victim ? P_TYPE(victim) : 0] - threshold;
    if (promo)
        swap += PieceVal[promo] - PieceVal[PAWN];
    if (swap <= 0)
        return 1;

    swap = PieceVal[P_TYPE(mover)] - swap;
    if (swap > 0)
        return 1;

    occ[from] = PS_EMPTY;
    if (flag == M_EP)
        occ[(color == WHITE) ? to - 16 : to + 16] = PS_EMPTY;

    int c = 1 - color;
    for (int depth = 0; depth < 16; depth++) {
        int afrom = -1;
        int atype = seeLeast(occ, to, c, &afrom);
        if (atype == 0)
            break;
        swap = PieceVal[atype] - swap;
        if (swap <= 0)
            return 0;
        if (atype == KING)
            break;
        c = 1 - c;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* evaluation                                                          */
/* ------------------------------------------------------------------ */

/* piece-square tables, written rank 8 first / file a first (white view) */
static const int PstPawnMg[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    62, 62, 62, 64, 64, 62, 62, 62,
    12, 16, 26, 36, 36, 26, 16, 12,
    4,  8,  16, 28, 28, 16, 8,  4,
    0,  2,  4,  18, 18, 4,  2,  0,
    4,  -4, -10, 2,  2,  -10, -4, 4,
    6,  12, 10, -22, -22, 10, 12, 6,
    0,  0,  0,  0,  0,  0,  0,  0
};
static const int PstKnight[64] = {
    -54, -32, -20, -20, -20, -20, -32, -54,
    -30, -12, 12,  20,  20,  12,  -12, -30,
    -18, 4,   20,  28,  28,  20,  4,   -18,
    -16, 6,   24,  30,  30,  24,  6,   -16,
    -16, 4,   22,  28,  28,  22,  4,   -16,
    -18, 4,   16,  24,  24,  16,  4,   -18,
    -32, -16, 0,   6,   6,   0,   -16, -32,
    -56, -34, -20, -18, -18, -20, -34, -56
};
static const int PkBishopMg[64] = {
    -14, -8, -4, -4, -4, -4, -8, -14,
    -8,  4,  8,  8,  8,  8,  4,  -8,
    -4,  8,  12, 12, 12, 12, 8,  -4,
    -4,  8,  12, 14, 14, 12, 8,  -4,
    -4,  8,  12, 14, 14, 12, 8,  -4,
    -4,  8,  12, 12, 12, 12, 8,  -4,
    -8,  4,  8,  8,  8,  8,  4,  -8,
    -14, -8, -4, -4, -4, -4, -8, -14
};
static const int PkBishopPg[64] = {
    -4, 2,  6,  8,  8,  6,  2,  -4,
    2,  8,  14, 16, 16, 14, 8,  2,
    6,  14, 18, 20, 20, 18, 14, 6,
    8,  16, 20, 22, 22, 20, 16, 8,
    8,  16, 20, 22, 22, 20, 16, 8,
    6,  14, 18, 20, 20, 18, 14, 6,
    2,  8,  14, 16, 16, 14, 8,  2,
    -4, 2,  6,  8,  8,  6,  2,  -4
};
static const int PstRook[64] = {
    2, 6, 6, 6,  6,  6, 6, 2,
    8, 14, 14, 14, 14, 14, 14, 8,
    2, 6, 8, 10, 10, 8,  6, 2,
    2, 6, 8, 10, 10, 8,  6, 2,
    2, 6, 8, 10, 10, 8,  6, 2,
    2, 6, 8, 10, 10, 8,  6, 2,
    2, 6, 6, 8,  8,  6,  6, 2,
    4, 10, 10, 12, 12, 10, 10, 4
};
static const int PstQueen[64] = {
    -8, -4, -4, -2, -2, -4, -4, -8,
    -4, 0,  2,  0,  0,  2,  0,  -4,
    -4, 2,  4,  4,  4,  4,  2,  -4,
    -2, 0,  4,  4,  4,  4,  0,  -2,
    -2, 2,  4,  4,  4,  4,  2,  -2,
    -4, 2,  4,  4,  4,  4,  2,  -4,
    -4, 0,  2,  0,  0,  2,  0,  -4,
    -8, -4, -4, -2, -2, -4, -4, -8
};
static const int PkKingMg[64] = {
    -64, -56, -46, -64, -64, -46, -56, -64,
    -48, -36, -32, -42, -42, -32, -36, -48,
    -48, -38, -24, -28, -28, -24, -38, -48,
    -46, -36, -28, -22, -22, -28, -36, -46,
    -32, -26, -18, -12, -12, -18, -26, -32,
    -12, -12, -16, -18, -18, -16, -12, -12,
    18,  20, 4,  0,  0,  4,  20, 18,
    22,  32, 14, 4,  4,  14, 32, 22
};
static const int PkKingPg[64] = {
    -70, -50, -30, -18, -18, -30, -50, -70,
    -40, -20, 0,   10,  10,  0,   -20, -40,
    -20, 6,   24,  34,  34,  24,  6,   -20,
    -10, 16,  34,  46,  46,  34,  16,  -10,
    -10, 16,  34,  46,  46,  34,  16,  -10,
    -20, 6,   24,  34,  34,  24,  6,   -20,
    -40, -20, 0,   10,  10,  0,   -20, -40,
    -70, -50, -30, -18, -18, -30, -50, -70
};

static const int MatMg[7] = { 0, 100, 320, 330, 500, 900, 0 };
static const int MatPg[7] = { 0, 104, 322, 336, 520, 940, 0 };

static const int *mgTable(int t)
{
    switch (t) {
    case KNIGHT: return PstKnight;
    case BISHOP: return PkBishopMg;
    case ROOK: return PstRook;
    case QUEEN: return PstQueen;
    default: return PkKingMg;
    }
}

static const int *pgTable(int t)
{
    switch (t) {
    case KNIGHT: return PstKnight;
    case BISHOP: return PkBishopPg;
    case ROOK: return PstRook;
    case QUEEN: return PstQueen;
    default: return PkKingPg;
    }
}

/* index into the visual tables (rank 8 first) */
static int visualIdx(int s)
{
    return (7 - SQ_RANK(s)) * 8 + SQ_FILE(s);
}

static int passedMg(int rel)
{
    static const int v[8] = { 0, 2, 6, 12, 26, 50, 88, 0 };
    return v[rel];
}

static int passedPg(int rel)
{
    static const int v[8] = { 0, 8, 16, 28, 48, 76, 120, 0 };
    return v[rel];
}

static int slidingAttacks(const Pos *p, int from, int ownColor, int delta)
{
    int cnt = 0;
    int t = from + delta;
    while (!OFF_88(t)) {
        int q = p->sq[t];
        if (q) {
            if (P_COLOR(q) != ownColor)
                cnt++;
            break;
        }
        cnt++;
        t += delta;
    }
    return cnt;
}

static int evalWhite(const Pos *p)
{
    int mg = 0, pg = 0, phase = 0;
    int pawnGrid[2][8][8];
    int pawnCnt[2][8];
    memset(pawnGrid, 0, sizeof pawnGrid);
    memset(pawnCnt, 0, sizeof pawnCnt);

    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++) {
            int q = p->sq[SQ(r, f)];
            if (q && P_TYPE(q) == PAWN) {
                int c = P_COLOR(q);
                pawnGrid[c][r][f] = 1;
                pawnCnt[c][f]++;
            }
        }

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        int enemy = 1 - c;
        int cnt = p->colorCount[c];
        int nBishop = 0;

        for (int i = 0; i < cnt; i++) {
            int s = p->colorSq[c][i];
            int piece = p->sq[s];
            int t = P_TYPE(piece);
            int vidx = (c == WHITE) ? visualIdx(s) : (63 - visualIdx(s));
            int r = SQ_RANK(s), f = SQ_FILE(s);

            if (t != PAWN && t != KING)
                phase += (t == ROOK) ? 2 : (t == QUEEN) ? 4 : 1;

            if (t == PAWN) {
                int rel = (c == WHITE) ? r : 7 - r;
                mg += sign * (100 + PstPawnMg[vidx]);
                pg += sign * (104 + 2 * rel);

                if (pawnCnt[c][f] > 1) {
                    mg -= sign * 12;
                    pg -= sign * 16;
                }
                if (!((f >= 1 && pawnCnt[c][f - 1]) ||
                      (f <= 6 && pawnCnt[c][f + 1]))) {
                    mg -= sign * 14;
                    pg -= sign * 10;
                }

                int passed = 1;
                int lo = (c == WHITE) ? r + 1 : 0;
                int hi = (c == WHITE) ? 7 : r - 1;
                for (int rr = lo; rr <= hi && passed; rr++)
                    for (int ff = f - 1; ff <= f + 1; ff++)
                        if (ff >= 0 && ff <= 7 && pawnGrid[enemy][rr][ff]) {
                            passed = 0;
                            break;
                        }
                if (passed) {
                    mg += sign * passedMg(rel);
                    pg += sign * passedPg(rel);
                }
                continue;
            }

            mg += sign * (MatMg[t] + mgTable(t)[vidx]);
            pg += sign * (MatPg[t] + pgTable(t)[vidx]);

            switch (t) {
            case KNIGHT: {
                int mob = 0;
                for (int k = 0; k < 8; k++) {
                    int u = s + KnightOff[k];
                    if (OFF_88(u))
                        continue;
                    int q = p->sq[u];
                    if (q == PS_EMPTY || P_COLOR(q) != c)
                        mob++;
                }
                mg += sign * 4 * mob;
                pg += sign * 5 * mob;
                int rel = (c == WHITE) ? r : 7 - r;
                if (rel >= 4 && rel <= 5) {
                    int sup = 0;
                    if (c == WHITE) {
                        if (r >= 1 && ((f >= 1 && p->sq[s - 17] == W_PAWN_) ||
                                       (f <= 6 && p->sq[s - 15] == W_PAWN_)))
                            sup = 1;
                    } else {
                        if (r <= 6 && ((f >= 1 && p->sq[s + 17] == B_PAWN_) ||
                                       (f <= 6 && p->sq[s + 15] == B_PAWN_)))
                            sup = 1;
                    }
                    if (sup) {
                        mg += sign * 18;
                        pg += sign * 12;
                    }
                }
                break;
            }
            case BISHOP: {
                int mob = 0;
                for (int k = 0; k < 4; k++)
                    mob += slidingAttacks(p, s, c, BishopOff[k]);
                mg += sign * 3 * mob;
                pg += sign * 4 * mob;
                nBishop++;
                break;
            }
            case ROOK: {
                int mob = 0;
                for (int k = 0; k < 4; k++)
                    mob += slidingAttacks(p, s, c, RookOff[k]);
                mg += sign * 2 * mob;
                pg += sign * 2 * mob;
                if (pawnCnt[c][f] == 0) {
                    int b = (pawnCnt[enemy][f] == 0) ? 18 : 12;
                    mg += sign * b;
                    pg += sign * (b / 2);
                }
                int rel = (c == WHITE) ? r : 7 - r;
                if (rel == 6) {
                    mg += sign * 12;
                    pg += sign * 6;
                }
                break;
            }
            case QUEEN: {
                int mob = 0;
                for (int k = 0; k < 8; k++)
                    mob += slidingAttacks(p, s, c, QueenOff[k]);
                mg += sign * mob;
                pg += sign * 2 * mob;
                break;
            }
            case KING: {
                int shield = 0;
                for (int ff = f - 1; ff <= f + 1; ff++) {
                    if (ff < 0 || ff > 7)
                        continue;
                    int found = 0;
                    for (int d = 1; d <= 2; d++) {
                        int rr = (c == WHITE) ? r + d : r - d;
                        if (rr < 0 || rr > 7)
                            break;
                        if (pawnGrid[c][rr][ff]) {
                            found = 1;
                            break;
                        }
                    }
                    shield += found ? 1 : -1;
                }
                mg += sign * 9 * shield;
                break;
            }
            default:
                break;
            }
        }

        if (nBishop >= 2) {
            mg += sign * 34;
            pg += sign * 48;
        }
    }

    if (phase > 24)
        phase = 24;
    int score = (mg * phase + pg * (24 - phase)) / 24;
    score += (p->side == WHITE) ? 8 : -8;
    return score;
}

/* score from the point of view of the side to move */
static int evalSide(const Pos *p)
{
    int s = evalWhite(p);
    return (p->side == WHITE) ? s : -s;
}

static int hasMaterial(const Pos *p)
{
    int c = p->side;
    return p->pieceCount[MAKE_PIECE(c, KNIGHT)] + p->pieceCount[MAKE_PIECE(c, BISHOP)] +
               p->pieceCount[MAKE_PIECE(c, ROOK)] +
               p->pieceCount[MAKE_PIECE(c, QUEEN)] >
           0;
}

static int insufficientMaterial(const Pos *p)
{
    if (p->pieceCount[W_PAWN_] || p->pieceCount[B_PAWN_])
        return 0;
    if (p->pieceCount[W_QUEEN_] || p->pieceCount[B_QUEEN_])
        return 0;
    if (p->pieceCount[W_ROOK_] || p->pieceCount[B_ROOK_])
        return 0;

    int wn = p->pieceCount[W_KNIGHT_], wb = p->pieceCount[W_BISHOP_];
    int bn = p->pieceCount[B_KNIGHT_], bb = p->pieceCount[B_BISHOP_];
    int total = wn + wb + bn + bb;

    if (total <= 1)
        return 1;
    if (total == 2 && wn == 0 && bn == 0 && wb == 1 && bb == 1) {
        int first = -1;
        for (int r = 0; r < 8; r++)
            for (int f = 0; f < 8; f++) {
                int q = p->sq[SQ(r, f)];
                if (q == W_BISHOP_ || q == B_BISHOP_) {
                    int col = (r + f) & 1;
                    if (first < 0)
                        first = col;
                    else if (col != first)
                        return 0;
                }
            }
        return 1;
    }
    return 0;
}

static int isRepetition(const Pos *p)
{
    int n = p->halfmove;
    for (int i = keyLen - 3; i >= 0 && i >= keyLen - 1 - n; i -= 2)
        if (keyStack[i] == p->key)
            return 1;
    return 0;
}

static int repetitionCount(const Pos *p)
{
    int c = 1;
    for (int i = keyLen - 3; i >= 0; i -= 2)
        if (keyStack[i] == p->key)
            c++;
    return c;
}

static int isDrawNode(const Pos *p)
{
    return p->halfmove >= 100 || isRepetition(p) || insufficientMaterial(p);
}

/* ------------------------------------------------------------------ */
/* transposition table                                                 */
/* ------------------------------------------------------------------ */

#define TT_BITS 22
#define TT_SIZE (1u << TT_BITS)

enum { TT_NONE = 0, TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };

typedef struct {
    uint32_t key;
    int32_t move;
    int16_t score;
    int16_t depth;
    uint8_t flag;
    uint8_t age;
    uint8_t pad[2];
} TTEntry;

static TTEntry *tt;
static uint32_t ttMask;
static int currentAge;

static int scoreToTT(int s, int ply)
{
    if (s >= MATE_BOUND)
        return s + ply;
    if (s <= -MATE_BOUND)
        return s - ply;
    return s;
}

static int scoreFromTT(int s, int ply)
{
    if (s >= MATE_BOUND)
        return s - ply;
    if (s <= -MATE_BOUND)
        return s + ply;
    return s;
}

static TTEntry *ttProbe(uint64_t key, int *found)
{
    TTEntry *e = &tt[key & ttMask];
    *found = (e->flag != TT_NONE && e->key == (uint32_t)(key >> 32));
    return e;
}

static void ttStore(uint64_t key, int move, int score, int depth, int flag)
{
    TTEntry *e = &tt[key & ttMask];
    int same = (e->flag != TT_NONE && e->key == (uint32_t)(key >> 32));
    if (!same && depth < e->depth && e->age == currentAge)
        return;
    if (move == 0 && same)
        move = e->move;
    e->key = (uint32_t)(key >> 32);
    e->move = move;
    e->score = (int16_t)score;
    e->depth = (int16_t)depth;
    e->flag = (uint8_t)flag;
    e->age = (uint8_t)currentAge;
}

/* ------------------------------------------------------------------ */
/* search state                                                        */
/* ------------------------------------------------------------------ */

static int64_t nodes;
static int seldepth;
static int aborted;
static double timeLimitMs;
static double startTime;
static int maxRootDepth;

static int killers[MAX_PLY][2];
static int historyTbl[16][128];
static int evalStack[MAX_PLY];
static int pvLen[MAX_PLY];
static int pvTable[MAX_PLY][MAX_PLY];

static double nowMs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void tick(void)
{
    if (timeLimitMs > 0 && (nodes & 1023) == 0 && nowMs() - startTime >= timeLimitMs)
        aborted = 1;
}

static int lmrReduction(int depth, int played)
{
    int R = 0;
    if (depth >= 3) R = 1;
    if (depth >= 6) R = 2;
    if (depth >= 10) R = 3;
    if (depth >= 15) R = 4;
    if (depth >= 21) R = 5;
    if (depth >= 28) R = 6;
    if (played > 6) R += 1;
    if (played > 12) R += 1;
    if (played > 24) R += 1;
    if (played > 48) R += 1;
    if (R > depth - 2) R = depth - 2;
    if (R < 0) R = 0;
    return R;
}

/* moves are sorted by descending score (insertion sort, lists are short) */
static void sortMoves(int *moves, int *scores, int n)
{
    for (int i = 1; i < n; i++) {
        int m = moves[i], sc = scores[i], j = i - 1;
        while (j >= 0 && scores[j] < sc) {
            moves[j + 1] = moves[j];
            scores[j + 1] = scores[j];
            j--;
        }
        moves[j + 1] = m;
        scores[j + 1] = sc;
    }
}

static int captureScore(const Pos *p, int m)
{
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    int promo = MOVE_PROMO(m);
    int victim = (MOVE_FLAG(m) == M_EP) ? PAWN : P_TYPE(p->sq[to]);
    if (victim) {
        int s = 1000000 + PieceVal[victim] * 16 - PieceVal[P_TYPE(p->sq[from])];
        if (promo)
            s += 100000 + PieceVal[promo] * 16;
        if (!seeGe(p, m, 0))
            s -= 400000;
        return s;
    }
    if (promo)
        return 900000 + PieceVal[promo] * 16;
    return 0;
}

static int moveScore(const Pos *p, int m, int ttMove, int ply)
{
    if (m == ttMove)
        return 10000000;
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    int promo = MOVE_PROMO(m);
    int victim = (MOVE_FLAG(m) == M_EP) ? PAWN : P_TYPE(p->sq[to]);
    if (victim) {
        int s = 1000000 + PieceVal[victim] * 16 - PieceVal[P_TYPE(p->sq[from])];
        if (promo)
            s += 100000 + PieceVal[promo] * 16;
        if (!seeGe(p, m, 0))
            s -= 400000;
        return s;
    }
    if (promo)
        return 900000 + PieceVal[promo] * 16;
    if (ply < MAX_PLY) {
        if (m == killers[ply][0])
            return 800000;
        if (m == killers[ply][1])
            return 790000;
    }
    return historyTbl[p->sq[from]][to];
}

/* ------------------------------------------------------------------ */
/* quiescence search                                                   */
/* ------------------------------------------------------------------ */

static int quiesce(Pos *p, int alpha, int beta, int ply)
{
    nodes++;
    if (ply > seldepth)
        seldepth = ply;
    tick();
    if (aborted)
        return 0;
    if (ply >= MAX_PLY - 2)
        return evalSide(p);

    int inCheck = inCheckOf(p, p->side);
    int moves[MAX_MOVES];
    int n;
    int best;

    if (inCheck) {
        n = generateMoves(p, moves, 0);
        best = -INF;
    } else {
        best = evalSide(p);
        if (best >= beta)
            return best;
        if (best > alpha)
            alpha = best;
        n = generateMoves(p, moves, 1);
    }

    int scores[MAX_MOVES];
    for (int i = 0; i < n; i++)
        scores[i] = captureScore(p, moves[i]);
    sortMoves(moves, scores, n);

    int played = 0;
    for (int i = 0; i < n; i++) {
        int m = moves[i];
        int flag = MOVE_FLAG(m), promo = MOVE_PROMO(m);
        int victimType = (flag == M_EP) ? PAWN : P_TYPE(p->sq[MOVE_TO(m)]);
        if (!inCheck && !promo && victimType && best + PieceVal[victimType] + 120 < alpha)
            continue;

        int mover = p->side;
        Undo u;
        makeMove(p, m, &u);
        if (isAttacked(p, p->kingSq[mover], 1 - mover)) {
            unmakeMove(p, &u);
            continue;
        }
        played++;

        int sc = -quiesce(p, -beta, -alpha, ply + 1);
        unmakeMove(p, &u);
        if (aborted)
            return 0;

        if (sc > best) {
            best = sc;
            if (sc > alpha) {
                alpha = sc;
                if (sc >= beta)
                    break;
            }
        }
    }

    if (inCheck && played == 0)
        return -(MATE - ply);
    return best;
}

/* ------------------------------------------------------------------ */
/* main alpha/beta search (PVS)                                        */
/* ------------------------------------------------------------------ */

static int search(Pos *p, int depth, int alpha, int beta, int ply, int canNull)
{
    pvLen[ply] = 0;
    nodes++;
    if (ply > seldepth)
        seldepth = ply;
    tick();
    if (aborted)
        return 0;

    if (ply >= MAX_PLY - 2)
        return inCheckOf(p, p->side) ? 0 : evalSide(p);

    int inCheck = inCheckOf(p, p->side);

    if (ply > 0) {
        if (isDrawNode(p))
            return 0;
        int mateLb = -(MATE - ply);
        int mateUb = MATE - ply - 1;
        if (alpha < mateLb)
            alpha = mateLb;
        if (beta > mateUb)
            beta = mateUb;
        if (alpha >= beta)
            return alpha;
    }

    if (inCheck)
        depth++;
    if (depth <= 0)
        return quiesce(p, alpha, beta, ply);

    int pv = (beta - alpha > 1);

    int ttHit = 0;
    TTEntry *e = ttProbe(p->key, &ttHit);
    int ttMove = 0, ttScore = 0, ttDepth = -1, ttFlag = TT_NONE;
    if (ttHit) {
        ttMove = e->move;
        ttScore = scoreFromTT(e->score, ply);
        ttDepth = e->depth;
        ttFlag = e->flag;
    }

    int staticEval = NO_EVAL;
    if (!inCheck)
        staticEval = evalSide(p);
    evalStack[ply] = staticEval;

    if (!pv && ttHit && ttDepth >= depth) {
        if (ttFlag == TT_EXACT)
            return ttScore;
        if (ttFlag == TT_LOWER && ttScore >= beta)
            return ttScore;
        if (ttFlag == TT_UPPER && ttScore <= alpha)
            return ttScore;
    }

    int improving = 0;
    if (!inCheck && ply >= 2 && evalStack[ply - 2] != NO_EVAL)
        improving = (staticEval > evalStack[ply - 2]);

    if (!pv && !inCheck) {
        if (depth <= 7 && abs(beta) < MATE_BOUND && staticEval - 88 * depth >= beta)
            return staticEval;

        if (canNull && depth >= 3 && staticEval >= beta && abs(beta) < MATE_BOUND &&
            hasMaterial(p)) {
            int R = 2 + (depth > 6 ? 1 : 0) + (staticEval - beta) / 200;
            if (R > 5)
                R = 5;
            Undo nu;
            doNullMove(p, &nu);
            int sc = -search(p, depth - 1 - R, -beta, -beta + 1, ply + 1, 0);
            undoNullMove(p, &nu);
            if (aborted)
                return 0;
            if (sc >= beta) {
                if (sc >= MATE_BOUND)
                    sc = beta;
                return sc;
            }
        }
    }

    int singular = 0;
    if (!pv && !inCheck && depth >= 8 && ttMove && ttDepth >= depth - 3 &&
        (ttFlag == TT_LOWER || ttFlag == TT_EXACT) && abs(ttScore) < MATE_BOUND) {
        int sBeta = ttScore - 2 * depth;
        if (sBeta > -INF && sBeta < beta) {
            int sDepth = (depth - 1) / 2;
            int mover = p->side;
            Undo u;
            makeMove(p, ttMove, &u);
            int illegal = isAttacked(p, p->kingSq[mover], 1 - mover);
            int sc = illegal ? -INF : -search(p, sDepth, sBeta - 1, sBeta, ply + 1, 1);
            unmakeMove(p, &u);
            if (aborted)
                return 0;
            if (illegal || sc < sBeta) {
                depth += 2;
                singular = 1;
            }
        }
    }

    int moves[MAX_MOVES];
    int n = generateMoves(p, moves, 0);
    int scores[MAX_MOVES];
    for (int i = 0; i < n; i++)
        scores[i] = moveScore(p, moves[i], ttMove, ply);
    sortMoves(moves, scores, n);

    int bestScore = -INF;
    int bestMove = 0;
    int bestFlag = TT_UPPER;
    int played = 0;
    int quietPlayed = 0;

    for (int i = 0; i < n; i++) {
        int m = moves[i];
        int to = MOVE_TO(m);
        int flag = MOVE_FLAG(m), promo = MOVE_PROMO(m);
        int isCap = (flag == M_EP) || (p->sq[to] != PS_EMPTY);
        int attackerPiece = p->sq[MOVE_FROM(m)];
        int isKiller = (m == killers[ply][0] || m == killers[ply][1]);

        if (played > 0 && !pv && !inCheck && m != ttMove && bestScore > -MATE_BOUND) {
            if (isCap) {
                if (depth <= 4 && !seeGe(p, m, -70 * depth))
                    continue;
            } else if (!promo) {
                if (depth <= 3 && staticEval + 140 + 95 * depth <= alpha)
                    continue;
                if (depth <= 3 && quietPlayed >= 3 + depth * depth)
                    continue;
            }
        }

        int mover = p->side;
        Undo u;
        makeMove(p, m, &u);
        if (isAttacked(p, p->kingSq[mover], 1 - mover)) {
            unmakeMove(p, &u);
            continue;
        }
        played++;
        int givesCheck = inCheckOf(p, p->side);

        int newDepth = depth - 1;
        int R = 0;
        if (played > 1 && !inCheck && !isCap && !promo && !givesCheck && depth >= 3) {
            R = lmrReduction(depth, played);
            if (pv)
                R--;
            if (improving)
                R--;
            if (isKiller)
                R -= 2;
            if (historyTbl[attackerPiece][MOVE_TO(m)] > 400)
                R--;
            if (singular)
                R++;
            if (R < 0)
                R = 0;
            if (R > newDepth - 1)
                R = newDepth - 1;
        }

        int sc;
        if (R > 0) {
            sc = -search(p, newDepth - R, -alpha - 1, -alpha, ply + 1, 1);
            if (sc > alpha && !aborted)
                sc = -search(p, newDepth, -beta, -alpha, ply + 1, 1);
        } else if (played == 1) {
            sc = -search(p, newDepth, -beta, -alpha, ply + 1, 1);
        } else {
            sc = -search(p, newDepth, -alpha - 1, -alpha, ply + 1, 1);
            if (sc > alpha && !aborted)
                sc = -search(p, newDepth, -beta, -alpha, ply + 1, 1);
        }

        unmakeMove(p, &u);
        if (aborted)
            return 0;

        if (sc > bestScore) {
            bestScore = sc;
            bestMove = m;
            if (sc > alpha) {
                alpha = sc;
                bestFlag = TT_EXACT;
                pvTable[ply][0] = m;
                memcpy(&pvTable[ply][1], &pvTable[ply + 1][0],
                       (size_t)pvLen[ply + 1] * sizeof(int));
                pvLen[ply] = pvLen[ply + 1] + 1;
                if (sc >= beta) {
                    bestFlag = TT_LOWER;
                    if (!isCap) {
                        if (killers[ply][0] != m) {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = m;
                        }
                        int h = historyTbl[attackerPiece][MOVE_TO(m)] + depth * depth;
                        historyTbl[attackerPiece][MOVE_TO(m)] = (h > 900000) ? 900000 : h;
                    }
                    break;
                }
            }
        }
        if (!isCap)
            quietPlayed++;
    }

    if (played == 0)
        return inCheck ? -(MATE - ply) : 0;

    ttStore(p->key, bestMove, scoreToTT(bestScore, ply), depth, bestFlag);
    return bestScore;
}

/* ------------------------------------------------------------------ */
/* root search + iterative deepening                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    int moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int n;
} RootPrev;

static int searchRoot(Pos *p, int depth, int alpha, int beta, int *bestOut,
                      const RootPrev *prev, RootPrev *out)
{
    int moves[MAX_MOVES];
    int n = generateLegal(p, moves);
    if (n == 0) {
        *bestOut = 0;
        return inCheckOf(p, p->side) ? -(MATE - 1) : 0;
    }

    int ttHit = 0;
    TTEntry *e = ttProbe(p->key, &ttHit);
    int ttMove = ttHit ? e->move : 0;

    int scores[MAX_MOVES];
    for (int i = 0; i < n; i++) {
        int sc = -INF;
        if (prev) {
            for (int j = 0; j < prev->n; j++)
                if (prev->moves[j] == moves[i]) {
                    sc = prev->scores[j];
                    break;
                }
        }
        if (moves[i] == ttMove)
            sc = 10000000;
        else if (sc > -INF)
            sc += 2000000;
        scores[i] = sc;
    }
    sortMoves(moves, scores, n);

    int rootAlpha = alpha;
    int best = -INF;
    int bm = moves[0];

    for (int i = 0; i < n; i++) {
        int m = moves[i];
        Undo u;
        int sc;
        makeMove(p, m, &u);
        if (isDrawNode(p)) {
            sc = 0;
        } else if (i == 0) {
            sc = -search(p, depth - 1, -beta, -alpha, 1, 1);
        } else {
            sc = -search(p, depth - 1, -alpha - 1, -alpha, 1, 1);
            if (!aborted && sc > alpha)
                sc = -search(p, depth - 1, -beta, -alpha, 1, 1);
        }
        unmakeMove(p, &u);

        if (aborted) {
            out->n = 0;
            *bestOut = bm;
            return best > -INF ? best : 0;
        }

        out->moves[i] = m;
        out->scores[i] = sc;
        if (sc > best) {
            best = sc;
            bm = m;
            if (sc > alpha)
                alpha = sc;
        }
    }
    out->n = n;

    int flag = TT_EXACT;
    if (best <= rootAlpha)
        flag = TT_UPPER;
    else if (best >= beta)
        flag = TT_LOWER;
    ttStore(p->key, bm, scoreToTT(best, 0), depth, flag);

    *bestOut = bm;
    return best;
}

static int thinkMove(Pos *p, double limitMs, int *outMove)
{
    int moves[MAX_MOVES];
    int n = generateLegal(p, moves);
    if (n == 0) {
        *outMove = 0;
        return 0;
    }
    *outMove = moves[0];
    if (n == 1)
        return 0;

    startTime = nowMs();
    timeLimitMs = (limitMs > 0.0) ? limitMs : 0.0;
    aborted = 0;
    nodes = 0;
    seldepth = 0;
    currentAge++;
    pvLen[0] = 0;
    memset(killers, 0, sizeof killers);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 128; j++)
            historyTbl[i][j] /= 2;

    RootPrev prev, cur;
    prev.n = 0;
    int best = moves[0];
    int bestScore = 0;
    int havePrev = 0;

    for (int depth = 1; depth <= maxRootDepth; depth++) {
        int alpha = -INF, beta = INF;
        if (havePrev && depth >= 5 && abs(bestScore) < MATE_BOUND) {
            alpha = bestScore - 32;
            beta = bestScore + 32;
        }

        int score = 0;
        int mv = best;
        for (;;) {
            cur.n = 0;
            score = searchRoot(p, depth, alpha, beta, &mv, prev.n ? &prev : NULL, &cur);
            if (aborted)
                break;
            if (score <= alpha) {
                alpha = -INF;
                continue;
            }
            if (score >= beta) {
                beta = INF;
                continue;
            }
            break;
        }
        if (aborted)
            break;

        if (mv)
            best = mv;
        bestScore = score;
        prev = cur;
        havePrev = 1;

        if (bestScore >= MATE_BOUND && MATE - bestScore <= depth + 1)
            break;
        if (bestScore <= -MATE_BOUND)
            break;
        if (timeLimitMs > 0.0 && nowMs() - startTime >= timeLimitMs)
            break;
    }

    *outMove = best;
    return bestScore;
}

/* ------------------------------------------------------------------ */
/* text interface                                                      */
/* ------------------------------------------------------------------ */

static const char *moveToStr(int m, char *buf)
{
    static const char promoChar[7] = { '?', '?', 'n', 'b', 'r', 'q', '?' };
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    int len = 4;
    buf[0] = (char)('a' + SQ_FILE(from));
    buf[1] = (char)('1' + SQ_RANK(from));
    buf[2] = (char)('a' + SQ_FILE(to));
    buf[3] = (char)('1' + SQ_RANK(to));
    if (MOVE_PROMO(m))
        buf[len++] = promoChar[MOVE_PROMO(m)];
    buf[len] = '\0';
    return buf;
}

/* exactly 8 lines of exactly 8 characters, rank 8 first, file a leftmost */
static void printBoard(const Pos *p)
{
    static const char letters[] = "?PNBRQK";
    for (int r = 7; r >= 0; r--) {
        for (int f = 0; f < 8; f++) {
            int q = p->sq[SQ(r, f)];
            char c = ' ';
            if (q) {
                c = letters[P_TYPE(q)];
                if (P_COLOR(q) == BLACK)
                    c = (char)tolower((unsigned char)c);
            }
            putchar(c);
        }
        putchar('\n');
    }
    fflush(stdout);
}

/* 1 when the game is over; then score/reason describe the result */
static int gameOver(Pos *p, const char **score, const char **reason)
{
    int moves[MAX_MOVES];
    int n = generateLegal(p, moves);
    if (n == 0) {
        if (inCheckOf(p, p->side)) {
            *score = (p->side == WHITE) ? "0-1" : "1-0";
            *reason = "checkmate";
        } else {
            *score = "1/2-1/2";
            *reason = "stalemate";
        }
        return 1;
    }
    if (p->halfmove >= 100) {
        *score = "1/2-1/2";
        *reason = "fifty";
        return 1;
    }
    if (repetitionCount(p) >= 3) {
        *score = "1/2-1/2";
        *reason = "repetition";
        return 1;
    }
    if (insufficientMaterial(p)) {
        *score = "1/2-1/2";
        *reason = "insufficient";
        return 1;
    }
    return 0;
}

static int readLine(FILE *f, char *buf, int cap)
{
    if (!fgets(buf, cap, f))
        return 0;
    int n = (int)strlen(buf);
    int overrun = (n > 0 && buf[n - 1] != '\n' && !feof(f));
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        buf[--n] = '\0';
    if (overrun) {
        int c;
        while ((c = fgetc(f)) != EOF && c != '\n')
            ;
    }
    return 1;
}

/* lowercase, whitespace removed */
static void normalizeToken(const char *in, char *out, int cap)
{
    int j = 0;
    for (const char *s = in; *s && j < cap - 1; s++)
        if (!isspace((unsigned char)*s))
            out[j++] = (char)tolower((unsigned char)*s);
    out[j] = '\0';
}

/* reason: 1 = unparsable or illegal, 2 = promotion piece missing */
static int parseMove(Pos *p, const char *tok, int *reason)
{
    *reason = 1;
    int len = (int)strlen(tok);
    if (len < 4 || len > 5)
        return 0;

    int f1 = tok[0] - 'a', r1 = tok[1] - '1';
    int f2 = tok[2] - 'a', r2 = tok[3] - '1';
    if (f1 < 0 || f1 > 7 || r1 < 0 || r1 > 7 || f2 < 0 || f2 > 7 || r2 < 0 || r2 > 7)
        return 0;

    int from = SQ(r1, f1), to = SQ(r2, f2);
    int promo = 0;
    if (len == 5) {
        switch (tok[4]) {
        case 'q': promo = QUEEN; break;
        case 'r': promo = ROOK; break;
        case 'b': promo = BISHOP; break;
        case 'n': promo = KNIGHT; break;
        default: return 0;
        }
    }

    int moves[MAX_MOVES];
    int n = generateLegal(p, moves);
    for (int i = 0; i < n; i++) {
        if (MOVE_FROM(moves[i]) == from && MOVE_TO(moves[i]) == to) {
            int mp = MOVE_PROMO(moves[i]);
            if (mp && !promo) {
                *reason = 2;
                return 0;
            }
            if (!mp && promo)
                continue;
            return moves[i];
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* game loop: human plays White, the program plays Black               */
/* ------------------------------------------------------------------ */

#define MAX_PLIES 3000

static int runGame(double limitSec)
{
    Pos p;
    static Undo undoStack[MAX_PLIES];
    int ply = 0;
    double limitMs = limitSec * 1000.0;
    char b[8];

    if (!setFen(&p, START_FEN))
        return 1;
    keyLen = 0;
    pushKey(p.key);

    setvbuf(stdout, NULL, _IOLBF, 0);

    for (;;) {
        char line[512];
        if (!readLine(stdin, line, (int)sizeof line))
            break;

        char tok[512];
        normalizeToken(line, tok, (int)sizeof tok);
        if (tok[0] == '\0')
            continue;

        if (strcmp(tok, "d") == 0) {
            printBoard(&p);
            continue;
        }

        if (strcmp(tok, "c") == 0) {
            /* the engine chooses White's move as well */
            int m = 0;
            thinkMove(&p, limitMs, &m);
            if (!m)
                break;
            makeMove(&p, m, &undoStack[ply < MAX_PLIES ? ply : 0]);
            ply++;
            printf("%s\n", moveToStr(m, b));
            fflush(stdout);

            const char *score = "", *reason = "";
            if (gameOver(&p, &score, &reason)) {
                printf("%s %s\n", score, reason);
                fflush(stdout);
                break;
            }
            if (inCheckOf(&p, p.side)) {
                printf("check\n");
                fflush(stdout);
            }
        } else {
            int why = 1;
            int m = parseMove(&p, tok, &why);
            if (!m) {
                if (why == 2)
                    printf("illegal move: promotion piece required (q, r, b or n)\n");
                else
                    printf("illegal move\n");
                fflush(stdout);
                continue;
            }
            makeMove(&p, m, &undoStack[ply < MAX_PLIES ? ply : 0]);
            ply++;

            const char *score = "", *reason = "";
            if (gameOver(&p, &score, &reason)) {
                printf("%s %s\n", score, reason);
                fflush(stdout);
                break;
            }
            if (inCheckOf(&p, p.side)) {
                printf("check\n");
                fflush(stdout);
            }
        }

        /* computer's reply as Black */
        {
            int m = 0;
            thinkMove(&p, limitMs, &m);
            if (!m)
                break;
            makeMove(&p, m, &undoStack[ply < MAX_PLIES ? ply : 0]);
            ply++;
            printf("%s\n", moveToStr(m, b));

            const char *score = "", *reason = "";
            if (gameOver(&p, &score, &reason)) {
                printf("%s %s\n", score, reason);
                fflush(stdout);
                break;
            }
            if (inCheckOf(&p, p.side))
                printf("check\n");
            fflush(stdout);
        }
    }
    return 0;
}

static int parseLimit(const char *arg, double *out)
{
    char *end = NULL;
    double v = strtod(arg, &end);
    if (end == arg)
        return 0;
    while (*end == ' ' || *end == '\t')
        end++;
    if (*end != '\0')
        return 0;
    if (!(v >= 0.0) || v > 1e9)
        return 0;
    *out = v;
    return 1;
}

#ifndef SELFTEST
int main(int argc, char **argv)
{
    if (argc > 2) {
        fprintf(stderr, "usage: %s [seconds]\n", argv[0]);
        return 2;
    }

    double limitSec = 600.0;
    if (argc == 2 && !parseLimit(argv[1], &limitSec)) {
        fprintf(stderr, "invalid time limit: %s\n", argv[1]);
        return 2;
    }

    initTables();
    tt = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
    if (!tt) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    ttMask = TT_SIZE - 1;

    /* 0 seconds means unlimited: search down to a fixed maximum depth */
    maxRootDepth = (limitSec > 0.0) ? (MAX_PLY - 4) : 20;

    int rc = runGame(limitSec);
    free(tt);
    tt = NULL;
    return rc;
}
#else
/* ------------------------------------------------------------------ */
/* build-time self test: perft move-generation counts                  */
/* ------------------------------------------------------------------ */

static uint64_t perft(Pos *p, int depth)
{
    if (depth <= 0)
        return 1;
    int moves[MAX_MOVES];
    int n = generateMoves(p, moves, 0);
    uint64_t total = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        int mover = p->side;
        makeMove(p, moves[i], &u);
        if (isAttacked(p, p->kingSq[mover], 1 - mover)) {
            unmakeMove(p, &u);
            continue;
        }
        total += perft(p, depth - 1);
        unmakeMove(p, &u);
    }
    return total;
}

/* the hash key must be a pure function of the position */
static uint64_t recomputeKey(const Pos *p)
{
    uint64_t k = 0;
    for (int s = 0; s < 128; s++)
        if (p->sq[s])
            k ^= zPiece[p->sq[s]][s];
    k ^= zCastle[p->castling] ^ EPI_KEY(p->ep);
    if (p->side == BLACK)
        k ^= zSide;
    return k;
}

/* every position reachable within depth plies must carry a consistent key, and
   unmake must restore key, key stack, castling, ep, clocks and move number */
static int keyWalk(Pos *p, int depth)
{
    if (p->key != recomputeKey(p))
        return 1;
    if (depth <= 0)
        return 0;
    int moves[MAX_MOVES];
    int n = generateMoves(p, moves, 0);
    Undo u;
    for (int i = 0; i < n; i++) {
        uint64_t k0 = p->key;
        int kl0 = keyLen;
        int mn0 = p->moveNumber, hm0 = p->halfmove;
        int mover = p->side;
        makeMove(p, moves[i], &u);
        int bad = 0;
        if (!isAttacked(p, p->kingSq[mover], 1 - mover))
            bad = keyWalk(p, depth - 1);
        unmakeMove(p, &u);
        if (bad)
            return 1;
        if (p->key != k0 || keyLen != kl0 || p->moveNumber != mn0 ||
            p->halfmove != hm0 || p->castling != u.castling || p->ep != u.ep)
            return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    initTables();
    tt = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
    if (!tt) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    ttMask = TT_SIZE - 1;
    maxRootDepth = 20;

    /* search "<fen>" [seconds] : report the move the search chooses */
    if (argc >= 3 && strcmp(argv[1], "search") == 0) {
        Pos p;
        if (!setFen(&p, argv[2])) {
            fprintf(stderr, "bad fen\n");
            return 2;
        }
        keyLen = 0;
        pushKey(p.key);
        double lim = 1.0;
        if (argc >= 4 && !parseLimit(argv[3], &lim)) {
            fprintf(stderr, "bad limit\n");
            return 2;
        }
        maxRootDepth = (lim > 0.0) ? (MAX_PLY - 4) : 20;
        int mv = 0;
        int score = thinkMove(&p, lim * 1000.0, &mv);
        char b[8];
        printf("%s score=%d seldepth=%d nodes=%lld\n", moveToStr(mv, b), score,
               seldepth, (long long)nodes);
        return 0;
    }

    if (argc < 2 || strcmp(argv[1], "perft") != 0) {
        double lim = 5.0;
        if (argc >= 2 && !parseLimit(argv[1], &lim)) {
            fprintf(stderr, "usage: %s perft [maxdepth] | %s [seconds]\n", argv[0],
                    argv[0]);
            return 2;
        }
        maxRootDepth = (lim > 0.0) ? (MAX_PLY - 4) : 20;
        return runGame(lim);
    }

    static const struct {
        const char *fen;
        uint64_t exp[6];
    } cases[] = {
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
          { 20, 400, 8902, 197281, 4865609, 119060324 } },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
          { 48, 2039, 97862, 4085603, 193690690, 8031647685ULL } },
        { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
          { 14, 191, 2812, 43238, 674624, 11030083 } },
        { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
          { 6, 264, 9467, 422333, 15833292, 706045033 } },
        { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
          { 44, 1486, 62379, 2103487, 89941194, 16228947210ULL } },
        { "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
          { 46, 2079, 89890, 3894594, 164075551, 32493633204ULL } },
    };

    int wantDepth = 5;
    if (argc >= 3) {
        wantDepth = atoi(argv[2]);
        if (wantDepth < 1)
            wantDepth = 1;
        if (wantDepth > 6)
            wantDepth = 6;
    }

    int fails = 0;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        Pos p;
        if (!setFen(&p, cases[i].fen)) {
            printf("FAIL fen-parse %s\n", cases[i].fen);
            fails++;
            continue;
        }
        keyLen = 0;
        pushKey(p.key);
        for (int d = 1; d <= wantDepth; d++) {
            double t0 = nowMs();
            uint64_t got = perft(&p, d);
            double ms = nowMs() - t0;
            int ok = (got == cases[i].exp[d - 1]);
            if (!ok || d == wantDepth)
                printf("%s case%d perft(%d) = %llu (expected %llu) %.0f ms %lld knps\n",
                       ok ? "ok  " : "FAIL", (int)i, d, (unsigned long long)got,
                       (unsigned long long)cases[i].exp[d - 1], ms,
                       (long long)(got / (ms > 0 ? ms : 1)));
            if (!ok)
                fails++;
        }
    }

    /* rule sanity checks */
    {
        Pos p;
        int fails2 = 0;
        if (!setFen(&p, "8/8/8/4k3/8/8/8/4K2R w K - 0 1"))
            fails2++;
        int m[16];
        keyLen = 0;
        pushKey(p.key);
        int n = generateLegal(&p, m);
        int castle = 0;
        for (int i = 0; i < n; i++)
            if (MOVE_FLAG(m[i]) == M_CASTLE)
                castle = 1;
        if (!castle) {
            printf("FAIL castling not generated\n");
            fails2++;
        }
        /* stalemate: black to move, no legal move, not in check */
        if (!setFen(&p, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
            fails2++;
        const char *sc = NULL, *rs = NULL;
        keyLen = 0;
        pushKey(p.key);
        if (!gameOver(&p, &sc, &rs) || strcmp(rs, "stalemate") != 0) {
            printf("FAIL stalemate detection (%s)\n", rs ? rs : "-");
            fails2++;
        }
        /* checkmate */
        if (!setFen(&p, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"))
            fails2++;
        keyLen = 0;
        pushKey(p.key);
        if (!gameOver(&p, &sc, &rs) || strcmp(rs, "checkmate") != 0 ||
            strcmp(sc, "0-1") != 0) {
            printf("FAIL mate detection (%s %s)\n", sc ? sc : "-", rs ? rs : "-");
            fails2++;
        }
        /* insufficient material: bishops of the same square colour (f8/f2) */
        if (!setFen(&p, "5b2/8/4k3/8/8/3K4/5B2/8 w - - 0 1"))
            fails2++;
        if (!insufficientMaterial(&p)) {
            printf("FAIL insufficient material (same colour bishops)\n");
            fails2++;
        }
        /* opposite coloured bishops are not insufficient material */
        if (!setFen(&p, "8/8/4k3/8/8/3K4/8/2B2b2 w - - 0 1"))
            fails2++;
        if (insufficientMaterial(&p)) {
            printf("FAIL insufficient material (opposite colour bishops)\n");
            fails2++;
        }
        /* the game must actually be declared drawn on insufficient material */
        if (!setFen(&p, "5b2/8/4k3/8/8/3K4/5B2/8 w - - 0 1"))
            fails2++;
        keyLen = 0;
        pushKey(p.key);
        if (!gameOver(&p, &sc, &rs) || strcmp(rs, "insufficient") != 0 ||
            strcmp(sc, "1/2-1/2") != 0) {
            printf("FAIL insufficient material game end (%s %s)\n", sc ? sc : "-",
                   rs ? rs : "-");
            fails2++;
        }
        /* fifty-move rule: halfmove 99 is still alive, the 100th quiet move ends it */
        if (!setFen(&p, "4k3/8/8/8/8/8/8/R2K3R w - - 99 60"))
            fails2++;
        keyLen = 0;
        pushKey(p.key);
        if (gameOver(&p, &sc, &rs)) {
            printf("FAIL fifty-move fired too early (%s)\n", rs);
            fails2++;
        }
        {
            Undo u;
            int mv = 0, mm[64];
            int nn = generateLegal(&p, mm);
            for (int i = 0; i < nn; i++)
                if (MOVE_FROM(mm[i]) == SQ(0, 0) && MOVE_TO(mm[i]) == SQ(1, 0))
                    mv = mm[i];
            if (!mv) {
                printf("FAIL Ra2 not generated for fifty-move check\n");
                fails2++;
            } else {
                makeMove(&p, mv, &u);
                if (!gameOver(&p, &sc, &rs) || strcmp(rs, "fifty") != 0 ||
                    strcmp(sc, "1/2-1/2") != 0) {
                    printf("FAIL fifty-move detection (%s %s)\n", sc ? sc : "-",
                           rs ? rs : "-");
                    fails2++;
                }
                unmakeMove(&p, &u);
            }
        }
        /* threefold repetition: rook and king shuffles */
        if (!setFen(&p, "4k3/8/8/8/8/8/8/R2K3R w - - 0 1"))
            fails2++;
        keyLen = 0;
        pushKey(p.key);
        {
            static const int shuffle[8][2] = {
                { SQ(0, 0), SQ(1, 0) }, { SQ(7, 4), SQ(6, 4) },
                { SQ(1, 0), SQ(0, 0) }, { SQ(6, 4), SQ(7, 4) },
                { SQ(0, 0), SQ(1, 0) }, { SQ(7, 4), SQ(6, 4) },
                { SQ(1, 0), SQ(0, 0) }, { SQ(6, 4), SQ(7, 4) },
            };
            Undo u;
            int repOk = 1;
            for (int i = 0; i < 8; i++) {
                int mv = 0, mm[64];
                int nn = generateLegal(&p, mm);
                for (int j = 0; j < nn; j++)
                    if (MOVE_FROM(mm[j]) == shuffle[i][0] &&
                        MOVE_TO(mm[j]) == shuffle[i][1])
                        mv = mm[j];
                if (!mv) {
                    printf("FAIL shuffle move %d not legal\n", i);
                    repOk = 0;
                    break;
                }
                makeMove(&p, mv, &u);
                if (i < 7 && gameOver(&p, &sc, &rs)) {
                    printf("FAIL repetition fired too early (ply %d: %s)\n", i, rs);
                    repOk = 0;
                    break;
                }
            }
            if (repOk) {
                if (!gameOver(&p, &sc, &rs) || strcmp(rs, "repetition") != 0 ||
                    strcmp(sc, "1/2-1/2") != 0) {
                    printf("FAIL repetition detection (%s %s)\n", sc ? sc : "-",
                           rs ? rs : "-");
                    repOk = 0;
                }
            }
            if (!repOk)
                fails2++;
        }
        printf(fails2 ? "RULE CHECKS FAILED: %d\n" : "rule checks ok\n", fails2);
        fails += fails2;
    }

    /* zobrist key / make-unmake consistency */
    {
        static const char *kfens[] = {
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "rnbqkbnr/1pp1pppp/p2p4/2p1P3/8/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 4",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/R2K3R w - - 30 40",
        };
        int kbad = 0;
        for (size_t i = 0; i < sizeof kfens / sizeof kfens[0]; i++) {
            Pos p;
            if (!setFen(&p, kfens[i])) {
                printf("FAIL key-check fen parse: %s\n", kfens[i]);
                kbad++;
                continue;
            }
            keyLen = 0;
            pushKey(p.key);
            if (keyWalk(&p, 3)) {
                printf("FAIL key/unmake consistency: %s\n", kfens[i]);
                kbad++;
            }
        }
        printf(kbad ? "key checks FAILED: %d\n" : "key checks ok\n", kbad);
        fails += kbad;
    }

    printf(fails ? "SELFTEST FAILURES: %d\n" : "SELFTEST OK\n", fails);
    return fails ? 1 : 0;
}
#endif
