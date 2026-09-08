// chess.go - complete text-mode chess game per /data/ai/chess.md spec.
// Human = White (moves first, stdin), computer = Black (stdout), coordinate
// notation (e2e4, e1g1 castling, e7e8q promotion). Full rules: castling,
// en passant, promotion, check, checkmate, stalemate, 50-move rule, threefold
// repetition, insufficient material.
// Extra stdin commands: "d" draws the board (exactly 8x8, UPPERCASE=White),
// "c" makes the computer choose and play YOUR move, then reply as usual.
// One optional CLI arg - exactly ONE limit per mode:
//
//	limit > 0 : TIME-limited - max seconds per move (default 600), wall
//	            clock enforced for real; depth and RAM unlimited
//	limit < 0 : DEPTH-limited - fixed depth (-10 = depth 10); time and RAM
//	            unlimited
//	limit = 0 : RAM-limited - no clock, no depth cap; the hash table is
//	            auto-sized to ~40% of available RAM (CHESS_HASH_MB env
//	            overrides) and the move is played once one full table's
//	            worth of positions has been stored. More RAM = deeper.
//
// Memory can never OOM: all data is fixed-size + a Go runtime memory limit.
// Searches in parallel on all CPU cores (Lazy SMP, shared lock-free TT).
// Extra modes: "demo" = self-play with the board after every move,
// "silentdemo" = the same without boards; VERBOSE=1 env = full search/
// debug trace on stderr (zero overhead when off; stdout stays protocol).
// Build: go build -o chess chess.go
// Self-test (perft + zobrist): CHESS_SELFTEST=1 ./chess
package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/bits"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---------- basic types ----------

const (
	Empty = 0
	WP    = 1
	WN    = 2
	WB    = 3
	WR    = 4
	WQ    = 5
	WK    = 6
	BP    = -1
	BN    = -2
	BB    = -3
	BR    = -4
	BQ    = -5
	BK    = -6
)

const (
	White = 0
	Black = 1
)

const (
	CastleWK = 1
	CastleWQ = 2
	CastleBK = 4
	CastleBQ = 8
)

const (
	fCapture = 1 << iota
	fDouble
	fEP
	fCastle
)

const (
	// MateScore/Inf fit int16 so scores pack into lock-free TT entries.
	MateScore = 30000
	Inf       = 31000
	MaxPly    = 128
	// Iterative-deepening ceiling (structural; unreachable in practice --
	// every +1 depth costs ~2-4x the time, so this is years, not hours).
	MaxDepth = 64
)

type Move struct {
	From, To int
	Promo    int // 0 or WN..WQ magnitude (2..5)
	Flags    int
}

var NoMove = Move{}

type Undo struct {
	captured int
	capSq    int
	rights   int
	ep       int
	half     int
	hash     uint64
	pawnKey  uint64
}

type Pos struct {
	b       [64]int
	side    int
	rights  int
	ep      int // -1 = none
	half    int // halfmove clock
	hash    uint64
	pawnKey uint64
	kingSq  [2]int
	hist    []uint64 // hashes after every move (incl. start position)
}

// ---------- zobrist ----------

var zPiece [13][64]uint64 // index pc+6
var zSide uint64
var zCastle [16]uint64
var zEP [8]uint64

func initZobrist() {
	rng := rand.New(rand.NewSource(20260907))
	for i := 0; i < 13; i++ {
		for s := 0; s < 64; s++ {
			zPiece[i][s] = rng.Uint64()
		}
	}
	zSide = rng.Uint64()
	for i := 0; i < 16; i++ {
		zCastle[i] = rng.Uint64()
	}
	for i := 0; i < 8; i++ {
		zEP[i] = rng.Uint64()
	}
}

func pidx(pc int) int { return pc + 6 }

func (p *Pos) computePawnKey() uint64 {
	var h uint64
	for s := 0; s < 64; s++ {
		if p.b[s] == WP || p.b[s] == BP {
			h ^= zPiece[pidx(p.b[s])][s]
		}
	}
	return h
}

func (p *Pos) computeHash() uint64 {
	var h uint64
	for s := 0; s < 64; s++ {
		if p.b[s] != Empty {
			h ^= zPiece[pidx(p.b[s])][s]
		}
	}
	h ^= zCastle[p.rights]
	if p.ep >= 0 {
		h ^= zEP[p.ep&7]
	}
	if p.side == Black {
		h ^= zSide
	}
	return h
}

// ---------- castling rights masks ----------

var castleMask [64]int

func initMasks() {
	for i := range castleMask {
		castleMask[i] = 0xF
	}
	castleMask[0] = 0xF &^ CastleWQ
	castleMask[4] = 0xF &^ (CastleWK | CastleWQ)
	castleMask[7] = 0xF &^ CastleWK
	castleMask[56] = 0xF &^ CastleBQ
	castleMask[60] = 0xF &^ (CastleBK | CastleBQ)
	castleMask[63] = 0xF &^ CastleBK
}

// ---------- setup ----------

func startPos() *Pos {
	p := &Pos{ep: -1, rights: CastleWK | CastleWQ | CastleBK | CastleBQ}
	back := []int{WR, WN, WB, WQ, WK, WB, WN, WR}
	for f := 0; f < 8; f++ {
		p.b[f] = back[f]
		p.b[8+f] = WP
		p.b[48+f] = BP
		p.b[56+f] = -back[f]
	}
	p.kingSq[White] = 4
	p.kingSq[Black] = 60
	p.side = White
	p.hash = p.computeHash()
	p.pawnKey = p.computePawnKey()
	p.hist = append(p.hist, p.hash)
	return p
}

// ---------- attack detection ----------

var knightD = [8][2]int{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
var kingD = [8][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
var rookD = [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
var bishD = [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

// attacked reports whether square sq is attacked by side "by".
func (p *Pos) attacked(sq, by int) bool {
	r, f := sq>>3, sq&7
	if by == White {
		if r > 0 {
			if f > 0 && p.b[sq-9] == WP {
				return true
			}
			if f < 7 && p.b[sq-7] == WP {
				return true
			}
		}
	} else {
		if r < 7 {
			if f > 0 && p.b[sq+7] == BP {
				return true
			}
			if f < 7 && p.b[sq+9] == BP {
				return true
			}
		}
	}
	kn, kg := WN, WK
	if by == Black {
		kn, kg = BN, BK
	}
	for _, d := range knightD {
		nr, nf := r+d[0], f+d[1]
		if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 && p.b[nr*8+nf] == kn {
			return true
		}
	}
	for _, d := range kingD {
		nr, nf := r+d[0], f+d[1]
		if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 && p.b[nr*8+nf] == kg {
			return true
		}
	}
	rq1, rq2 := WR, WQ
	if by == Black {
		rq1, rq2 = BR, BQ
	}
	for _, d := range rookD {
		nr, nf := r+d[0], f+d[1]
		for nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
			pc := p.b[nr*8+nf]
			if pc != Empty {
				if pc == rq1 || pc == rq2 {
					return true
				}
				break
			}
			nr += d[0]
			nf += d[1]
		}
	}
	bq1, bq2 := WB, WQ
	if by == Black {
		bq1, bq2 = BB, BQ
	}
	for _, d := range bishD {
		nr, nf := r+d[0], f+d[1]
		for nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
			pc := p.b[nr*8+nf]
			if pc != Empty {
				if pc == bq1 || pc == bq2 {
					return true
				}
				break
			}
			nr += d[0]
			nf += d[1]
		}
	}
	return false
}

func (p *Pos) inCheck(side int) bool {
	return p.attacked(p.kingSq[side], side^1)
}

// ---------- move generation (pseudo-legal) ----------

func (p *Pos) genPseudo(capturesOnly bool) []Move {
	moves := make([]Move, 0, 64)
	us := p.side
	own := func(pc int) bool {
		if us == White {
			return pc > 0
		}
		return pc < 0
	}
	enemy := func(pc int) bool {
		if us == White {
			return pc < 0
		}
		return pc > 0
	}
	addPromos := func(from, to, flags int) {
		for _, pr := range []int{WQ, WR, WB, WN} {
			moves = append(moves, Move{From: from, To: to, Promo: pr, Flags: flags})
		}
	}
	for sq := 0; sq < 64; sq++ {
		pc := p.b[sq]
		if pc == Empty || !own(pc) {
			continue
		}
		r, f := sq>>3, sq&7
		switch pc {
		case WP, BP:
			dir, startR, promoR := 8, 1, 6
			if us == Black {
				dir, startR, promoR = -8, 6, 1
			}
			// pushes
			to := sq + dir
			if to >= 0 && to < 64 && p.b[to] == Empty {
				if r == promoR {
					if !capturesOnly {
						addPromos(sq, to, 0)
					} else {
						// queen promotion counts as a "loud" move in quiescence
						moves = append(moves, Move{From: sq, To: to, Promo: WQ})
					}
				} else if !capturesOnly {
					moves = append(moves, Move{From: sq, To: to})
					if r == startR && p.b[sq+2*dir] == Empty {
						moves = append(moves, Move{From: sq, To: sq + 2*dir, Flags: fDouble})
					}
				}
			}
			// captures
			for _, df := range []int{-1, 1} {
				nf := f + df
				if nf < 0 || nf > 7 {
					continue
				}
				to := sq + dir + df
				if to < 0 || to > 63 {
					continue
				}
				if enemy(p.b[to]) {
					if r == promoR {
						addPromos(sq, to, fCapture)
					} else {
						moves = append(moves, Move{From: sq, To: to, Flags: fCapture})
					}
				} else if to == p.ep {
					moves = append(moves, Move{From: sq, To: to, Flags: fCapture | fEP})
				}
			}
		case WN, BN:
			for _, d := range knightD {
				nr, nf := r+d[0], f+d[1]
				if nr < 0 || nr > 7 || nf < 0 || nf > 7 {
					continue
				}
				to := nr*8 + nf
				if p.b[to] == Empty {
					if !capturesOnly {
						moves = append(moves, Move{From: sq, To: to})
					}
				} else if enemy(p.b[to]) {
					moves = append(moves, Move{From: sq, To: to, Flags: fCapture})
				}
			}
		case WK, BK:
			for _, d := range kingD {
				nr, nf := r+d[0], f+d[1]
				if nr < 0 || nr > 7 || nf < 0 || nf > 7 {
					continue
				}
				to := nr*8 + nf
				if p.b[to] == Empty {
					if !capturesOnly {
						moves = append(moves, Move{From: sq, To: to})
					}
				} else if enemy(p.b[to]) {
					moves = append(moves, Move{From: sq, To: to, Flags: fCapture})
				}
			}
			if capturesOnly {
				break
			}
			// castling (king not in check, path empty, transit squares safe)
			if us == White && sq == 4 {
				if p.rights&CastleWK != 0 && p.b[5] == Empty && p.b[6] == Empty &&
					!p.attacked(4, Black) && !p.attacked(5, Black) && !p.attacked(6, Black) {
					moves = append(moves, Move{From: 4, To: 6, Flags: fCastle})
				}
				if p.rights&CastleWQ != 0 && p.b[3] == Empty && p.b[2] == Empty && p.b[1] == Empty &&
					!p.attacked(4, Black) && !p.attacked(3, Black) && !p.attacked(2, Black) {
					moves = append(moves, Move{From: 4, To: 2, Flags: fCastle})
				}
			} else if us == Black && sq == 60 {
				if p.rights&CastleBK != 0 && p.b[61] == Empty && p.b[62] == Empty &&
					!p.attacked(60, White) && !p.attacked(61, White) && !p.attacked(62, White) {
					moves = append(moves, Move{From: 60, To: 62, Flags: fCastle})
				}
				if p.rights&CastleBQ != 0 && p.b[59] == Empty && p.b[58] == Empty && p.b[57] == Empty &&
					!p.attacked(60, White) && !p.attacked(59, White) && !p.attacked(58, White) {
					moves = append(moves, Move{From: 60, To: 58, Flags: fCastle})
				}
			}
		default: // sliders
			var dirs [][2]int
			switch pc {
			case WR, BR:
				dirs = rookD[:]
			case WB, BB:
				dirs = bishD[:]
			case WQ, BQ:
				dirs = append(rookD[:], bishD[:]...)
			}
			for _, d := range dirs {
				nr, nf := r+d[0], f+d[1]
				for nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
					to := nr*8 + nf
					if p.b[to] == Empty {
						if !capturesOnly {
							moves = append(moves, Move{From: sq, To: to})
						}
					} else {
						if enemy(p.b[to]) {
							moves = append(moves, Move{From: sq, To: to, Flags: fCapture})
						}
						break
					}
					nr += d[0]
					nf += d[1]
				}
			}
		}
	}
	return moves
}

// ---------- make / unmake ----------

func (p *Pos) make(m Move) Undo {
	u := Undo{captured: Empty, capSq: m.To, rights: p.rights, ep: p.ep, half: p.half, hash: p.hash, pawnKey: p.pawnKey}
	pc := p.b[m.From]
	mover := p.side
	if m.Flags&fEP != 0 {
		if mover == White {
			u.capSq = m.To - 8
		} else {
			u.capSq = m.To + 8
		}
		u.captured = p.b[u.capSq]
		p.hash ^= zPiece[pidx(u.captured)][u.capSq]
		p.pawnKey ^= zPiece[pidx(u.captured)][u.capSq] // EP victim is a pawn
		p.b[u.capSq] = Empty
	} else if p.b[m.To] != Empty {
		u.captured = p.b[m.To]
		p.hash ^= zPiece[pidx(u.captured)][m.To]
		if u.captured == WP || u.captured == BP {
			p.pawnKey ^= zPiece[pidx(u.captured)][m.To]
		}
	}
	p.hash ^= zPiece[pidx(pc)][m.From]
	if pc == WP || pc == BP {
		p.pawnKey ^= zPiece[pidx(pc)][m.From]
	}
	p.b[m.From] = Empty
	np := pc
	if m.Promo != 0 {
		if mover == White {
			np = m.Promo
		} else {
			np = -m.Promo
		}
	}
	p.b[m.To] = np
	p.hash ^= zPiece[pidx(np)][m.To]
	if np == WP || np == BP {
		p.pawnKey ^= zPiece[pidx(np)][m.To]
	}
	if m.Flags&fCastle != 0 {
		var rf, rt int
		switch m.To {
		case 6:
			rf, rt = 7, 5
		case 2:
			rf, rt = 0, 3
		case 62:
			rf, rt = 63, 61
		case 58:
			rf, rt = 56, 59
		}
		rk := p.b[rf]
		p.hash ^= zPiece[pidx(rk)][rf] ^ zPiece[pidx(rk)][rt]
		p.b[rf] = Empty
		p.b[rt] = rk
	}
	if pc == WK {
		p.kingSq[White] = m.To
	} else if pc == BK {
		p.kingSq[Black] = m.To
	}
	p.hash ^= zCastle[p.rights]
	p.rights &= castleMask[m.From] & castleMask[m.To]
	p.hash ^= zCastle[p.rights]
	if p.ep >= 0 {
		p.hash ^= zEP[p.ep&7]
	}
	p.ep = -1
	if m.Flags&fDouble != 0 {
		if mover == White {
			p.ep = m.From + 8
		} else {
			p.ep = m.From - 8
		}
		p.hash ^= zEP[p.ep&7]
	}
	if pc == WP || pc == BP || u.captured != Empty {
		p.half = 0
	} else {
		p.half++
	}
	p.side ^= 1
	p.hash ^= zSide
	p.hist = append(p.hist, p.hash)
	return u
}

func (p *Pos) unmake(m Move, u Undo) {
	p.hist = p.hist[:len(p.hist)-1]
	p.side ^= 1
	mover := p.side
	pc := p.b[m.To]
	if m.Promo != 0 {
		if mover == White {
			pc = WP
		} else {
			pc = BP
		}
	}
	p.b[m.From] = pc
	p.b[m.To] = Empty
	if u.captured != Empty {
		p.b[u.capSq] = u.captured
	}
	if m.Flags&fCastle != 0 {
		switch m.To {
		case 6:
			p.b[7] = p.b[5]
			p.b[5] = Empty
		case 2:
			p.b[0] = p.b[3]
			p.b[3] = Empty
		case 62:
			p.b[63] = p.b[61]
			p.b[61] = Empty
		case 58:
			p.b[56] = p.b[59]
			p.b[59] = Empty
		}
	}
	if pc == WK {
		p.kingSq[White] = m.From
	} else if pc == BK {
		p.kingSq[Black] = m.From
	}
	p.rights = u.rights
	p.ep = u.ep
	p.half = u.half
	p.hash = u.hash
	p.pawnKey = u.pawnKey
}

// legal move filter: after make, the mover's king must not be attacked.
func (p *Pos) genLegal() []Move {
	pseudo := p.genPseudo(false)
	legal := pseudo[:0]
	for _, m := range pseudo {
		u := p.make(m)
		if !p.attacked(p.kingSq[p.side^1], p.side) {
			legal = append(legal, m)
		}
		p.unmake(m, u)
	}
	return legal
}

// ---------- evaluation: PeSTO tapered PSTs + structure terms ----------

// pieceVal is used for move ordering (MVV-LVA) and pruning margins.
var pieceVal = [7]int{0, 100, 320, 330, 500, 900, 0}

// seeVal is used by static exchange evaluation.
var seeVal = [7]int{0, 100, 325, 335, 500, 975, 20000}

var mgVal = [7]int{0, 82, 337, 365, 477, 1025, 0}
var egVal = [7]int{0, 94, 281, 297, 512, 936, 0}

// PeSTO (Rofchade) piece-square tables, rank 8 first (visual):
// white reads pst[sq^56], black reads pst[sq].
var mgPawnT = [64]int{
	0, 0, 0, 0, 0, 0, 0, 0,
	98, 134, 61, 95, 68, 126, 34, -11,
	-6, 7, 26, 31, 65, 56, 25, -20,
	-14, 13, 6, 21, 23, 12, 17, -23,
	-27, -2, -5, 12, 17, 6, 10, -25,
	-26, -4, -4, -10, 3, 3, 33, -12,
	-35, -1, -20, -23, -15, 24, 38, -22,
	0, 0, 0, 0, 0, 0, 0, 0,
}
var egPawnT = [64]int{
	0, 0, 0, 0, 0, 0, 0, 0,
	178, 173, 158, 134, 147, 132, 165, 187,
	94, 100, 85, 67, 56, 53, 82, 84,
	32, 24, 13, 5, -2, 4, 17, 17,
	13, 9, -3, -7, -7, -8, 3, -1,
	4, 7, -6, 1, 0, -5, -1, -8,
	13, 8, 8, 10, 13, 0, 2, -7,
	0, 0, 0, 0, 0, 0, 0, 0,
}
var mgKnightT = [64]int{
	-167, -89, -34, -49, 61, -97, -15, -107,
	-73, -41, 72, 36, 23, 62, 7, -17,
	-47, 60, 37, 65, 84, 129, 73, 44,
	-9, 17, 19, 53, 37, 69, 18, 22,
	-13, 4, 16, 13, 28, 19, 21, -8,
	-23, -9, 12, 10, 19, 17, 25, -16,
	-29, -53, -12, -3, -1, 18, -14, -19,
	-105, -21, -58, -33, -17, -28, -19, -23,
}
var egKnightT = [64]int{
	-58, -38, -13, -28, -31, -27, -63, -99,
	-25, -8, -25, -2, -9, -25, -24, -52,
	-24, -20, 10, 9, -1, -9, -19, -41,
	-17, 3, 22, 22, 22, 11, 8, -18,
	-18, -6, 16, 25, 16, 17, 4, -18,
	-23, -3, -1, 15, 10, -3, -20, -22,
	-42, -20, -10, -5, -2, -20, -23, -44,
	-29, -51, -23, -15, -22, -18, -50, -64,
}
var mgBishopT = [64]int{
	-29, 4, -82, -37, -25, -42, 7, -8,
	-26, 16, -18, -13, 30, 59, 18, -47,
	-16, 37, 43, 40, 35, 50, 37, -2,
	-4, 5, 19, 50, 37, 37, 7, -2,
	-6, 13, 13, 26, 34, 12, 10, 4,
	0, 15, 15, 15, 14, 27, 18, 10,
	4, 15, 16, 0, 7, 21, 33, 1,
	-33, -3, -14, -21, -13, -12, -39, -21,
}
var egBishopT = [64]int{
	-14, -21, -11, -8, -7, -9, -17, -24,
	-8, -4, 7, -12, -3, -13, -4, -14,
	2, -8, 0, -1, -2, 6, 0, 4,
	-3, 9, 12, 9, 14, 10, 3, 2,
	-6, 3, 13, 19, 7, 10, -3, -9,
	-12, -3, 8, 10, 13, 3, -7, -15,
	-14, -18, -7, -1, 4, -9, -15, -27,
	-23, -9, -23, -5, -9, -16, -5, -17,
}
var mgRookT = [64]int{
	32, 42, 32, 51, 63, 9, 31, 43,
	27, 32, 58, 62, 80, 67, 26, 44,
	-5, 19, 26, 36, 17, 45, 61, 16,
	-24, -11, 7, 26, 24, 35, -8, -20,
	-36, -26, -12, -1, 9, -7, 6, -23,
	-45, -25, -16, -17, 3, 0, -5, -33,
	-44, -16, -20, -9, -1, 11, -6, -71,
	-19, -13, 1, 17, 16, 7, -37, -26,
}
var egRookT = [64]int{
	13, 10, 18, 15, 12, 12, 8, 5,
	11, 13, 13, 11, -3, 3, 8, 3,
	7, 7, 7, 5, 4, -3, -5, -3,
	4, 3, 13, 1, 2, 1, -1, 2,
	3, 5, 8, 4, -5, -6, -8, -11,
	-4, 0, -5, -1, -7, -12, -8, -16,
	-6, -6, 0, 2, -9, -9, -11, -3,
	-9, 2, 3, -1, -5, -13, 4, -20,
}
var mgQueenT = [64]int{
	-28, 0, 29, 12, 59, 44, 43, 45,
	-24, -39, -5, 1, -16, 57, 28, 54,
	-13, -17, 7, 8, 29, 56, 47, 57,
	-27, -27, -16, -16, -1, 17, -2, 1,
	-9, -26, -9, -10, -2, -4, 3, -3,
	-14, 2, -11, -2, -5, 2, 14, 5,
	-35, -8, 11, 2, 8, 15, -3, 1,
	-1, -18, -9, 10, -15, -25, -31, -50,
}
var egQueenT = [64]int{
	-9, 22, 22, 27, 27, 19, 10, 20,
	-17, 20, 32, 41, 58, 25, 30, 0,
	-20, 6, 9, 49, 47, 35, 19, 9,
	3, 22, 24, 45, 57, 40, 57, 36,
	-18, 28, 19, 47, 31, 34, 39, 23,
	-16, -27, 15, 6, 9, 17, 10, 5,
	-22, -23, -30, -16, -16, -23, -36, -32,
	-33, -28, -22, -43, -5, -32, -20, -41,
}
var mgKingT = [64]int{
	-65, 23, 16, -15, -56, -34, 2, 13,
	29, -1, -20, -7, -8, -4, -38, -29,
	-9, 24, 2, -16, -20, 6, 22, -22,
	-17, -20, -12, -27, -30, -25, -14, -36,
	-49, -1, -27, -39, -46, -44, -33, -51,
	-14, -14, -22, -46, -44, -30, -15, -27,
	1, 7, -8, -64, -43, -16, 9, 8,
	-15, 36, 12, -54, 8, -28, 24, 14,
}
var egKingT = [64]int{
	-74, -35, -18, -18, -11, 15, 4, -17,
	-12, 17, 14, 17, 17, 38, 23, 11,
	10, 17, 23, 15, 20, 45, 44, 13,
	-8, 22, 24, 27, 26, 33, 26, 3,
	-18, -4, 21, 24, 27, 23, 9, -11,
	-19, -3, 11, 21, 23, 16, 7, -9,
	-27, -11, 4, 13, 14, 4, -5, -17,
	-53, -34, -21, -11, -28, -14, -24, -43,
}

var mgPST = [7]*[64]int{nil, &mgPawnT, &mgKnightT, &mgBishopT, &mgRookT, &mgQueenT, &mgKingT}
var egPST = [7]*[64]int{nil, &egPawnT, &egKnightT, &egBishopT, &egRookT, &egQueenT, &egKingT}

var phaseVal = [7]int{0, 0, 1, 1, 2, 4, 0}

// passed pawn bonus by relative rank (0 = own back rank)
var passedMG = [8]int{0, 4, 9, 15, 30, 60, 100, 0}
var passedEG = [8]int{0, 14, 24, 40, 70, 125, 195, 0}

// eval returns the score from the side-to-move's perspective (centipawns).
// PeSTO tapered material+PST plus: bishop pair, doubled/isolated/passed
// pawns, rooks on (semi-)open files, king pawn shield, tempo.
// ---------- evaluation: bitboard-assisted (mobility, king safety, threats) ----------

const (
	fileABB = uint64(0x0101010101010101)
	fileHBB = uint64(0x8080808080808080)
)

var queenD = [8][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

var knightBB, kingBB [64]uint64

func init() {
	for sq := 0; sq < 64; sq++ {
		f, r := sq&7, sq>>3
		for _, d := range knightD {
			if nf, nr := f+d[0], r+d[1]; nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
				knightBB[sq] |= 1 << uint(nr*8+nf)
			}
		}
		for _, d := range kingD {
			if nf, nr := f+d[0], r+d[1]; nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
				kingBB[sq] |= 1 << uint(nr*8+nf)
			}
		}
	}
}

func pawnAttWBB(bb uint64) uint64 { return (bb&^fileABB)<<7 | (bb&^fileHBB)<<9 }
func pawnAttBBB(bb uint64) uint64 { return (bb&^fileABB)>>9 | (bb&^fileHBB)>>7 }

// rayAttBB returns sliding attacks from sq over occupancy occ.
func rayAttBB(sq int, occ uint64, dirs [][2]int) uint64 {
	var att uint64
	f0, r0 := sq&7, sq>>3
	for _, d := range dirs {
		f, r := f0+d[0], r0+d[1]
		for f >= 0 && f < 8 && r >= 0 && r < 8 {
			b := uint64(1) << uint(r*8+f)
			att |= b
			if occ&b != 0 {
				break
			}
			f += d[0]
			r += d[1]
		}
	}
	return att
}

func chebDist(a, b int) int {
	df, dr := a&7-b&7, a>>3-b>>3
	if df < 0 {
		df = -df
	}
	if dr < 0 {
		dr = -dr
	}
	if df > dr {
		return df
	}
	return dr
}

// mobility bonuses indexed by number of reachable safe squares
var mobN = [9]int{-50, -22, -8, 0, 7, 13, 18, 22, 26}
var mobB = [14]int{-40, -18, 0, 10, 18, 25, 30, 34, 38, 41, 44, 46, 48, 50}
var mobR = [15]int{-40, -18, -4, 2, 6, 10, 14, 18, 22, 25, 28, 31, 33, 35, 37}
var mobQ = [28]int{-20, -12, -6, -2, 2, 5, 8, 11, 13, 15, 17, 19, 21, 23, 25, 26,
	27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}

// connected pawns bonus by rank (white perspective)
var connPawn = [8]int{0, 3, 5, 8, 14, 24, 40, 0}

// pawn-structure cache: lock-free, XOR-validated (key ^ data)
const pawnTTBits = 20

var pawnTT [1 << pawnTTBits]struct{ xkey, data uint64 }

// pawnStructure scores cacheable pawn terms (white minus black):
// isolated, doubled, connected/phalanx, passed-pawn base bonuses.
func pawnStructure(wp, bp uint64) (mg, eg int) {
	wAtt, bAtt := pawnAttWBB(wp), pawnAttBBB(bp)
	for bb := wp; bb != 0; bb &= bb - 1 {
		sq := bits.TrailingZeros64(bb)
		f, r := sq&7, sq>>3
		var neigh, front uint64
		if f > 0 {
			neigh |= fileABB << uint(f-1)
		}
		if f < 7 {
			neigh |= fileABB << uint(f+1)
		}
		if wp&neigh == 0 {
			mg -= 12
			eg -= 9
		}
		front = (fileABB << uint(f)) << uint(8*(r+1))
		if wp&front != 0 { // doubled (own pawn ahead on same file)
			mg -= 11
			eg -= 17
		}
		span := front | neigh&^((1<<uint(8*(r+1)))-1)
		if bp&span == 0 { // passed
			mg += passedMG[r]
			eg += passedEG[r]
		}
		sup := wAtt&(1<<uint(sq)) != 0
		pha := f > 0 && wp&(1<<uint(sq-1)) != 0 || f < 7 && wp&(1<<uint(sq+1)) != 0
		if sup || pha {
			b := connPawn[r]
			if sup && pha {
				b += b / 2
			}
			mg += b
			eg += b * 2 / 3
		}
	}
	for bb := bp; bb != 0; bb &= bb - 1 {
		sq := bits.TrailingZeros64(bb)
		f, r := sq&7, sq>>3
		var neigh uint64
		if f > 0 {
			neigh |= fileABB << uint(f-1)
		}
		if f < 7 {
			neigh |= fileABB << uint(f+1)
		}
		if bp&neigh == 0 {
			mg += 12
			eg += 9
		}
		low := uint64(1)<<uint(8*r) - 1
		front := (fileABB << uint(f)) & low
		if bp&front != 0 {
			mg += 11
			eg += 17
		}
		span := front | neigh&low
		if wp&span == 0 {
			mg -= passedMG[7-r]
			eg -= passedEG[7-r]
		}
		sup := bAtt&(1<<uint(sq)) != 0
		pha := f > 0 && bp&(1<<uint(sq-1)) != 0 || f < 7 && bp&(1<<uint(sq+1)) != 0
		if sup || pha {
			b := connPawn[7-r]
			if sup && pha {
				b += b / 2
			}
			mg -= b
			eg -= b * 2 / 3
		}
	}
	return
}

func (p *Pos) eval() int {
	var mg, eg, phase int
	var wp, bp, wOcc, bOcc uint64
	var wpF, bpF [10]uint16 // pawn rank-bitmask per file (index = file+1)
	// piece lists (squares) per side for N,B,R,Q
	var wN, wB, wR, wQ, bN, bB, bR, bQ [10]int
	var nwN, nwB, nwR, nwQ, nbN, nbB, nbR, nbQ int
	for sq := 0; sq < 64; sq++ {
		pc := p.b[sq]
		if pc == Empty {
			continue
		}
		bit := uint64(1) << uint(sq)
		if pc > 0 {
			wOcc |= bit
			phase += phaseVal[pc]
			mg += mgVal[pc] + mgPST[pc][sq^56]
			eg += egVal[pc] + egPST[pc][sq^56]
			switch pc {
			case WP:
				wp |= bit
				wpF[sq&7+1] |= 1 << uint(sq>>3)
			case WN:
				wN[nwN] = sq
				nwN++
			case WB:
				wB[nwB] = sq
				nwB++
			case WR:
				wR[nwR] = sq
				nwR++
			case WQ:
				wQ[nwQ] = sq
				nwQ++
			}
		} else {
			bOcc |= bit
			q := -pc
			phase += phaseVal[q]
			mg -= mgVal[q] + mgPST[q][sq]
			eg -= egVal[q] + egPST[q][sq]
			switch q {
			case WP:
				bp |= bit
				bpF[sq&7+1] |= 1 << uint(sq>>3)
			case WN:
				bN[nbN] = sq
				nbN++
			case WB:
				bB[nbB] = sq
				nbB++
			case WR:
				bR[nbR] = sq
				nbR++
			case WQ:
				bQ[nbQ] = sq
				nbQ++
			}
		}
	}
	occ := wOcc | bOcc
	// pawn structure (cached)
	if wp|bp != 0 {
		k := p.pawnKey
		pe := &pawnTT[k&(1<<pawnTTBits-1)]
		xk := atomic.LoadUint64(&pe.xkey)
		d := atomic.LoadUint64(&pe.data)
		if xk^d == k {
			mg += int(int16(uint16(d)))
			eg += int(int16(uint16(d >> 16)))
		} else {
			pmg, peg := pawnStructure(wp, bp)
			mg += pmg
			eg += peg
			nd := uint64(uint16(int16(pmg))) | uint64(uint16(int16(peg)))<<16 | 1<<48
			atomic.StoreUint64(&pe.data, nd)
			atomic.StoreUint64(&pe.xkey, k^nd)
		}
	}
	wk, bk := p.kingSq[White], p.kingSq[Black]
	wPawnAtt, bPawnAtt := pawnAttWBB(wp), pawnAttBBB(bp)
	wMobArea := ^(wOcc | bPawnAtt)
	bMobArea := ^(bOcc | wPawnAtt)
	wRing := kingBB[wk] | 1<<uint(wk)
	bRing := kingBB[bk] | 1<<uint(bk)
	wAtt, bAtt := wPawnAtt|kingBB[wk], bPawnAtt|kingBB[bk]
	wUnits, wAttackers, bUnits, bAttackers := 0, 0, 0, 0 // attacks on ENEMY king
	// white pieces
	for i := 0; i < nwN; i++ {
		a := knightBB[wN[i]]
		wAtt |= a
		c := bits.OnesCount64(a & wMobArea)
		mg += mobN[c]
		eg += mobN[c] * 2 / 3
		if t := a & bRing; t != 0 {
			wUnits += 2 * bits.OnesCount64(t)
			wAttackers++
		}
	}
	for i := 0; i < nwB; i++ {
		a := rayAttBB(wB[i], occ&^(1<<uint(wB[i])), bishD[:])
		wAtt |= a
		c := bits.OnesCount64(a & wMobArea)
		if c > 13 {
			c = 13
		}
		mg += mobB[c]
		eg += mobB[c] * 2 / 3
		if t := a & bRing; t != 0 {
			wUnits += 2 * bits.OnesCount64(t)
			wAttackers++
		}
	}
	for i := 0; i < nwR; i++ {
		sq := wR[i]
		a := rayAttBB(sq, occ&^(1<<uint(sq)), rookD[:])
		wAtt |= a
		c := bits.OnesCount64(a & wMobArea)
		if c > 14 {
			c = 14
		}
		mg += mobR[c]
		eg += mobR[c]
		if t := a & bRing; t != 0 {
			wUnits += 3 * bits.OnesCount64(t)
			wAttackers++
		}
		f := sq & 7
		if wpF[f+1] == 0 { // (semi-)open file
			if bpF[f+1] == 0 {
				mg += 25
				eg += 12
			} else {
				mg += 12
				eg += 6
			}
		}
	}
	for i := 0; i < nwQ; i++ {
		a := rayAttBB(wQ[i], occ&^(1<<uint(wQ[i])), queenD[:])
		wAtt |= a
		c := bits.OnesCount64(a & wMobArea)
		if c > 27 {
			c = 27
		}
		mg += mobQ[c]
		eg += mobQ[c]
		if t := a & bRing; t != 0 {
			wUnits += 5 * bits.OnesCount64(t)
			wAttackers++
		}
	}
	// black pieces
	for i := 0; i < nbN; i++ {
		a := knightBB[bN[i]]
		bAtt |= a
		c := bits.OnesCount64(a & bMobArea)
		mg -= mobN[c]
		eg -= mobN[c] * 2 / 3
		if t := a & wRing; t != 0 {
			bUnits += 2 * bits.OnesCount64(t)
			bAttackers++
		}
	}
	for i := 0; i < nbB; i++ {
		a := rayAttBB(bB[i], occ&^(1<<uint(bB[i])), bishD[:])
		bAtt |= a
		c := bits.OnesCount64(a & bMobArea)
		if c > 13 {
			c = 13
		}
		mg -= mobB[c]
		eg -= mobB[c] * 2 / 3
		if t := a & wRing; t != 0 {
			bUnits += 2 * bits.OnesCount64(t)
			bAttackers++
		}
	}
	for i := 0; i < nbR; i++ {
		sq := bR[i]
		a := rayAttBB(sq, occ&^(1<<uint(sq)), rookD[:])
		bAtt |= a
		c := bits.OnesCount64(a & bMobArea)
		if c > 14 {
			c = 14
		}
		mg -= mobR[c]
		eg -= mobR[c]
		if t := a & wRing; t != 0 {
			bUnits += 3 * bits.OnesCount64(t)
			bAttackers++
		}
		f := sq & 7
		if bpF[f+1] == 0 {
			if wpF[f+1] == 0 {
				mg -= 25
				eg -= 12
			} else {
				mg -= 12
				eg -= 6
			}
		}
	}
	for i := 0; i < nbQ; i++ {
		a := rayAttBB(bQ[i], occ&^(1<<uint(bQ[i])), queenD[:])
		bAtt |= a
		c := bits.OnesCount64(a & bMobArea)
		if c > 27 {
			c = 27
		}
		mg -= mobQ[c]
		eg -= mobQ[c]
		if t := a & wRing; t != 0 {
			bUnits += 5 * bits.OnesCount64(t)
			bAttackers++
		}
	}
	// king safety: weighted attack units on the enemy king ring
	// (needs at least two attacking pieces and the queen still on the board)
	if wAttackers >= 2 && nwQ > 0 {
		u := wUnits * wUnits / 6
		if u > 450 {
			u = 450
		}
		mg += u
		eg += u / 4
	}
	if bAttackers >= 2 && nbQ > 0 {
		u := bUnits * bUnits / 6
		if u > 450 {
			u = 450
		}
		mg -= u
		eg -= u / 4
	}
	// threats
	bPieces := bOcc &^ bp
	wPieces := wOcc &^ wp
	if t := wPawnAtt & bPieces; t != 0 { // pawns attacking pieces
		n := bits.OnesCount64(t)
		mg += 50 * n
		eg += 30 * n
	}
	if t := bPawnAtt & wPieces; t != 0 {
		n := bits.OnesCount64(t)
		mg -= 50 * n
		eg -= 30 * n
	}
	if t := bOcc & wAtt &^ bAtt; t != 0 { // hanging enemy men
		n := bits.OnesCount64(t)
		mg += 18 * n
		eg += 12 * n
	}
	if t := wOcc & bAtt &^ wAtt; t != 0 {
		n := bits.OnesCount64(t)
		mg -= 18 * n
		eg -= 12 * n
	}
	// passed-pawn extras: blockade and king proximity (endgame)
	for bb := wp; bb != 0; bb &= bb - 1 {
		sq := bits.TrailingZeros64(bb)
		f, r := sq&7, sq>>3
		if (bpF[f]|bpF[f+1]|bpF[f+2])&(^uint16(0)<<uint(r+1)) == 0 {
			stop := sq + 8
			if occ&(1<<uint(stop)) != 0 {
				mg -= 8
				eg -= passedEG[r] / 3
			}
			eg += 6*chebDist(bk, stop)*r/4 - 3*chebDist(wk, stop)*r/4
		}
	}
	for bb := bp; bb != 0; bb &= bb - 1 {
		sq := bits.TrailingZeros64(bb)
		f, r := sq&7, sq>>3
		if (wpF[f]|wpF[f+1]|wpF[f+2])&((1<<uint(r))-1) == 0 {
			rr := 7 - r
			stop := sq - 8
			if occ&(1<<uint(stop)) != 0 {
				mg += 8
				eg += passedEG[rr] / 3
			}
			eg -= 6*chebDist(wk, stop)*rr/4 - 3*chebDist(bk, stop)*rr/4
		}
	}
	// bishop pair
	if nwB >= 2 {
		mg += 24
		eg += 45
	}
	if nbB >= 2 {
		mg -= 24
		eg -= 45
	}
	// king pawn shield (middlegame): own pawns 1-2 ranks ahead of the king
	if kf, kr := wk&7, wk>>3; kr <= 1 {
		for i := kf; i <= kf+2; i++ {
			if i == 0 || i == 9 {
				continue
			}
			if wpF[i]&(6<<uint(kr)) != 0 {
				mg += 8
			} else {
				mg -= 14
			}
			if wpF[i] == 0 && bpF[i] == 0 {
				mg -= 10
			}
		}
	}
	if kf, kr := bk&7, bk>>3; kr >= 6 {
		for i := kf; i <= kf+2; i++ {
			if i == 0 || i == 9 {
				continue
			}
			if bpF[i]&(3<<uint(kr-2)) != 0 {
				mg -= 8
			} else {
				mg += 14
			}
			if wpF[i] == 0 && bpF[i] == 0 {
				mg += 10
			}
		}
	}
	if phase > 24 {
		phase = 24
	}
	score := (mg*phase + eg*(24-phase)) / 24
	// scale down drawish endgames
	if phase <= 8 {
		strongerPawns := wp
		if score < 0 {
			strongerPawns = bp
		}
		if nwB == 1 && nbB == 1 && nwN+nbN+nwR+nbR+nwQ+nbQ == 0 {
			// opposite-colored bishops: strong draw tendency
			light := uint64(0x55AA55AA55AA55AA)
			wLight := light>>uint(wB[0])&1 != 0
			bLight := light>>uint(bB[0])&1 != 0
			if wLight != bLight {
				score = score * 55 / 100
			}
		}
		if strongerPawns == 0 { // winning side has no pawns: hard to convert
			score = score * 60 / 100
		}
	}
	if p.side == Black {
		score = -score
	}
	return score + 15 // side-to-move tempo
}

// non-pawn material present for the side to move (null-move guard)
func (p *Pos) hasNonPawnMaterial() bool {
	for sq := 0; sq < 64; sq++ {
		pc := p.b[sq]
		if pc == Empty {
			continue
		}
		if p.side == White && pc > WP && pc < WK {
			return true
		}
		if p.side == Black && pc < BP && pc > BK {
			return true
		}
	}
	return false
}

// ---------- transposition table (lock-free, XOR-validated) ----------

const (
	ttAlpha = 1
	ttExact = 2
	ttBeta  = 3
)

// Three atomically-accessed words per entry with xkey = key ^ data ^ extra:
// a torn or racing write fails validation at probe time and is ignored,
// making the shared table safe for Lazy SMP without locks. Entries live in
// 2-way buckets (slots idx and idx^1) with generation-based aging so deep
// results survive while stale ones get recycled.
// data:  move 20 | gen 6 <<20 | flag 2 <<26 | depth 8 <<28 | score 16 <<36
// extra: static eval int16 (evalNone = unknown)
type ttEntry struct {
	xkey  uint64
	data  uint64
	extra uint64
}

const ttEntryBytes = 24

const evalNone = -32000

var ttSize uint64 // entries, always a power of two
var ttTable []ttEntry
var ttGen atomic.Uint32 // bumped once per bestMove call, 6-bit wrap

// availableRAM estimates currently available memory in bytes.
func availableRAM() uint64 {
	// Linux
	if data, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, ln := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(ln, "MemAvailable:") {
				if f := strings.Fields(ln); len(f) >= 2 {
					if kb, err := strconv.ParseUint(f[1], 10, 64); err == nil {
						return kb << 10
					}
				}
			}
		}
	}
	// FreeBSD (and similar): free + inactive pages
	if out, err := exec.Command("sysctl", "-n", "hw.pagesize",
		"vm.stats.vm.v_free_count", "vm.stats.vm.v_inact_count").Output(); err == nil {
		if f := strings.Fields(string(out)); len(f) == 3 {
			ps, e1 := strconv.ParseUint(f[0], 10, 64)
			fr, e2 := strconv.ParseUint(f[1], 10, 64)
			in, e3 := strconv.ParseUint(f[2], 10, 64)
			if e1 == nil && e2 == nil && e3 == nil {
				return (fr + in) * ps
			}
		}
	}
	// last resort: half of physical RAM
	if out, err := exec.Command("sysctl", "-n", "hw.physmem").Output(); err == nil {
		if v, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); err == nil {
			return v / 2
		}
	}
	return 1 << 30 // conservative 1 GiB assumption
}

// initTT sizes the hash table from the RAM the machine can actually spare
// (~80% of available, floor 64 MB, cap 64 GB, CHESS_HASH_MB env overrides)
// and sets a Go runtime memory limit (table + GC headroom) so the process
// can never OOM in any mode: all other engine data is fixed-size, so total
// memory stays flat at this ceiling no matter how long it thinks. In
// RAM-limited (0) mode this table is also the search budget itself.
func initTT() {
	var budget uint64
	if s := os.Getenv("CHESS_HASH_MB"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			budget = uint64(v) << 20
		}
	}
	if budget == 0 {
		budget = availableRAM() * 4 / 5
		if max := uint64(64) << 30; budget > max {
			budget = max
		}
		if min := uint64(64) << 20; budget < min {
			budget = min
		}
	}
	entries := uint64(1)
	for entries*2*ttEntryBytes <= budget {
		entries *= 2
	}
	ttSize = entries
	ttTable = make([]ttEntry, ttSize)
	head := ttSize * ttEntryBytes / 4
	if min := uint64(96) << 20; head < min {
		head = min
	}
	debug.SetMemoryLimit(int64(ttSize*ttEntryBytes + head))
	if os.Getenv("CHESS_VERBOSE") == "1" || verbose {
		fmt.Fprintf(os.Stderr, "hash: %d MB | memory limit: %d MB | threads: %d\n",
			ttSize*ttEntryBytes>>20, (ttSize*ttEntryBytes+head)>>20, runtime.NumCPU())
	}
}

func encMove(m Move) uint64 {
	return uint64(m.From) | uint64(m.To)<<6 | uint64(m.Promo)<<12 | uint64(m.Flags)<<16
}

func decMove(v uint64) Move {
	return Move{From: int(v & 63), To: int(v >> 6 & 63), Promo: int(v >> 12 & 15), Flags: int(v >> 16 & 15)}
}

// ttProbe returns (move, score, depth, flag, staticEval, hit). staticEval
// is evalNone when the entry carries none.
func ttProbe(hash uint64) (Move, int, int, int, int, bool) {
	idx := hash & (ttSize - 1)
	for s := uint64(0); s < 2; s++ {
		e := &ttTable[idx^s]
		d := atomic.LoadUint64(&e.data)
		x := atomic.LoadUint64(&e.extra)
		if atomic.LoadUint64(&e.xkey)^d^x == hash {
			return decMove(d), int(int16(uint16(d >> 36))), int(uint8(d >> 28)),
				int(d >> 26 & 3), int(int16(uint16(x))), true
		}
	}
	return NoMove, 0, 0, 0, evalNone, false
}

var ttStores atomic.Uint64 // stores since the current 0-mode search began

func ttStore(hash uint64, m Move, score, depth, flag, ply, se int) {
	if score > MateScore-2000 {
		score += ply
	} else if score < -(MateScore - 2000) {
		score -= ply
	}
	gen := uint64(ttGen.Load()) & 63
	d := encMove(m) | gen<<20 | uint64(flag)<<26 | uint64(uint8(depth))<<28 |
		uint64(uint16(int16(score)))<<36
	x := uint64(uint16(int16(se)))
	idx := hash & (ttSize - 1)
	// bucket choice: same-position slot first, else the shallower/older slot
	tgt := idx
	bestQ := int64(1) << 62
	for s := uint64(0); s < 2; s++ {
		e := &ttTable[idx^s]
		ed := atomic.LoadUint64(&e.data)
		ex := atomic.LoadUint64(&e.extra)
		if atomic.LoadUint64(&e.xkey)^ed^ex == hash {
			// protect deep results from shallow same-key overwrites
			// (quiescence stores depth-0 entries constantly)
			if flag != ttExact && depth+3 < int(uint8(ed>>28)) {
				nd := ed&^(uint64(63)<<20) | gen<<20 // refresh age only
				atomic.StoreUint64(&e.data, nd)
				atomic.StoreUint64(&e.xkey, hash^nd^ex)
				return
			}
			tgt = idx ^ s
			break
		}
		q := int64(uint8(ed>>28)) - 8*int64((gen-(ed>>20))&63)
		if q < bestQ {
			bestQ = q
			tgt = idx ^ s
		}
	}
	e := &ttTable[tgt]
	atomic.StoreUint64(&e.xkey, hash^d^x)
	atomic.StoreUint64(&e.data, d)
	atomic.StoreUint64(&e.extra, x)
	ttStores.Add(1)
}

// ---------- verbose / debug instrumentation (VERBOSE=1) ----------

// verbose enables a full search/debug trace on stderr: per-depth results
// with score + PV, aspiration re-searches, 0-mode RAM-budget progress,
// per-move summaries (nodes, nps, seldepth, TT fill, heap) and every
// position as FEN + zobrist key. stdout stays a clean move protocol.
var verbose = os.Getenv("VERBOSE") != "" && os.Getenv("VERBOSE") != "0"

var startTime = time.Now()

// vlog prints one timestamped (seconds since program start) stderr line.
func vlog(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "[%8.3f] "+format+"\n",
		append([]any{time.Since(startTime).Seconds()}, a...)...)
}

func fmtN(n uint64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.2fG", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	}
	return strconv.FormatUint(n, 10)
}

func scoreStr(s int) string {
	if s > MateScore-2000 {
		return fmt.Sprintf("mate %d", (MateScore-s+1)/2)
	}
	if s < -(MateScore - 2000) {
		return fmt.Sprintf("mate -%d", (MateScore+s+1)/2)
	}
	return fmt.Sprintf("cp %d", s)
}

// ttFullPct estimates hash-table fill by sampling up to 8192 entries.
func ttFullPct() float64 {
	n := uint64(8192)
	if n > ttSize {
		n = ttSize
	}
	step := ttSize / n
	used := 0
	for i := uint64(0); i < n; i++ {
		e := &ttTable[i*step]
		if atomic.LoadUint64(&e.xkey) != 0 || atomic.LoadUint64(&e.data) != 0 {
			used++
		}
	}
	return 100 * float64(used) / float64(n)
}

func toFEN(p *Pos) string {
	var sb strings.Builder
	for r := 7; r >= 0; r-- {
		empty := 0
		for f := 0; f < 8; f++ {
			pc := p.b[r*8+f]
			if pc == Empty {
				empty++
				continue
			}
			if empty > 0 {
				sb.WriteByte(byte('0' + empty))
				empty = 0
			}
			if pc > 0 {
				sb.WriteByte("PNBRQK"[pc-1])
			} else {
				sb.WriteByte("pnbrqk"[-pc-1])
			}
		}
		if empty > 0 {
			sb.WriteByte(byte('0' + empty))
		}
		if r > 0 {
			sb.WriteByte('/')
		}
	}
	if p.side == White {
		sb.WriteString(" w ")
	} else {
		sb.WriteString(" b ")
	}
	cr := ""
	if p.rights&CastleWK != 0 {
		cr += "K"
	}
	if p.rights&CastleWQ != 0 {
		cr += "Q"
	}
	if p.rights&CastleBK != 0 {
		cr += "k"
	}
	if p.rights&CastleBQ != 0 {
		cr += "q"
	}
	if cr == "" {
		cr = "-"
	}
	sb.WriteString(cr)
	if p.ep >= 0 {
		sb.WriteString(" " + sqName(p.ep))
	} else {
		sb.WriteString(" -")
	}
	full := 1
	if n := len(p.hist); n > 0 {
		full = (n-1)/2 + 1
	}
	sb.WriteString(fmt.Sprintf(" %d %d", p.half, full))
	return sb.String()
}

// pvString reconstructs the principal variation by walking the TT from the
// root, validating every move against the legal list (races can only
// truncate the line, never corrupt it).
func pvString(root *Pos, first Move, maxLen int) string {
	p := root.clone()
	var sb strings.Builder
	seen := make(map[uint64]bool, maxLen)
	m := first
	for i := 0; i < maxLen; i++ {
		ok := false
		for _, lm := range p.genLegal() {
			if lm == m {
				ok = true
				break
			}
		}
		if !ok {
			break
		}
		if i > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(moveStr(m))
		p.make(m)
		if seen[p.hash] {
			break
		}
		seen[p.hash] = true
		tm, _, _, _, _, hit := ttProbe(p.hash)
		if !hit || tm == NoMove {
			break
		}
		m = tm
	}
	return sb.String()
}

// ---------- search ----------

// searchCtl is shared by all parallel search workers.
type searchCtl struct {
	stop        atomic.Bool
	hasDeadline bool
	deadline    time.Time     // hard stop, enforced inside the search
	soft        time.Time     // don't start another iteration after this
	memLimited  bool          // 0-mode: stop when the hash capacity is used up
	nodes       atomic.Uint64 // ~nodes this move (flushed every 1024/worker)
	nextMark    atomic.Uint64 // verbose: next 0-mode progress checkpoint
}

// search-shape tuning constants (bisect: pass1 regression)
const (
	razorOn     = true
	rfpBase     = 80
	rfpImp      = 20
	nullEvalDiv = 200
	lmpBase     = 3
	lmpDepth    = 6
	lmpHalve    = true
	futImp      = 40
	lmrNotImp   = 1
)

var enginePool []*Engine
var poolOnce sync.Once

type Engine struct {
	p         *Pos
	ctl       *searchCtl
	stop      bool
	nodes     uint64
	seldepth  int // deepest ply touched (incl. quiescence)
	killers   [MaxPly][2]Move
	excluded  [MaxPly]Move
	corrHist  [2][16384]int32
	history   [2][64][64]int32 // butterfly quiet history (gravity-bounded)
	counter   [2][64][64]Move  // countermove heuristic, indexed by prev move
	capHist   [12][64][7]int32 // capture history: [attacker][to][victimType]
	contHist  [2][768][768]int16
	moveStack [MaxPly]Move // move that led to each ply (for countermoves)
	evalStack [MaxPly]int  // static eval per ply (evalNone in check)
	pcStack   [MaxPly]int  // pieceIdx*64+to of the move at each ply, -1 none
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// pieceIdx maps a signed mailbox piece to 0..11 (white P..K, black P..K).
func pieceIdx(pc int) int {
	if pc > 0 {
		return pc - 1
	}
	return -pc + 5
}

// victimType maps a capture target to 1..5 (P..Q), EP handled by caller.
func victimType(pc int) int {
	if pc < 0 {
		pc = -pc
	}
	return pc
}

// quietHist is the combined quiet-move history used for ordering and LMR.
func (e *Engine) quietHist(side int, m Move, ply int) int {
	pc := pieceIdx(e.p.b[m.From])*64 + m.To
	h := int(e.history[side][m.From][m.To])
	if ply >= 1 && e.pcStack[ply-1] >= 0 {
		h += int(e.contHist[0][e.pcStack[ply-1]][pc])
	}
	if ply >= 2 && e.pcStack[ply-2] >= 0 {
		h += int(e.contHist[1][e.pcStack[ply-2]][pc])
	}
	return h
}

// gravity-bounded history update: |value| stays <= 16384
func histUp32(h *int32, bonus int) {
	b := bonus
	if b < 0 {
		b = -b
	}
	*h += int32(bonus) - *h*int32(b)/16384
}

func histUp16(h *int16, bonus int) {
	b := bonus
	if b < 0 {
		b = -b
	}
	*h += int16(bonus - int(*h)*b/16384)
}

// updateQuiets rewards the cutoff quiet and punishes the tried quiets.
func (e *Engine) updateQuiets(cut Move, tried []Move, side, depth, ply int) {
	bonus := imin(32*depth*depth+64*depth, 1280)
	apply := func(m Move, b int) {
		histUp32(&e.history[side][m.From][m.To], b)
		pc := pieceIdx(e.p.b[m.From])*64 + m.To
		if ply >= 1 && e.pcStack[ply-1] >= 0 {
			histUp16(&e.contHist[0][e.pcStack[ply-1]][pc], b)
		}
		if ply >= 2 && e.pcStack[ply-2] >= 0 {
			histUp16(&e.contHist[1][e.pcStack[ply-2]][pc], b)
		}
	}
	apply(cut, bonus)
	for _, m := range tried {
		if m != cut {
			apply(m, -bonus)
		}
	}
}

func (e *Engine) capHistOf(m Move) *int32 {
	p := e.p
	v := WP
	if m.Flags&fEP == 0 {
		v = victimType(p.b[m.To])
	}
	return &e.capHist[pieceIdx(p.b[m.From])][m.To][v]
}

func (e *Engine) checkTime() {
	if e.nodes&1023 != 0 {
		return
	}
	if verbose {
		e.ctl.nodes.Add(1024) // stats only kept in verbose mode
	}
	if e.ctl.stop.Load() {
		e.stop = true
		return
	}
	if e.ctl.hasDeadline && time.Now().After(e.ctl.deadline) {
		e.ctl.stop.Store(true)
		e.stop = true
	}
	// RAM-limited (0) mode: the move's budget is one full hash table's worth
	// of stored positions - the only limit in this mode is memory capacity.
	if e.ctl.memLimited {
		st := ttStores.Load()
		if verbose {
			if mark := e.ctl.nextMark.Load(); mark != 0 && st >= mark &&
				e.ctl.nextMark.CompareAndSwap(mark, mark+(ttSize+9)/10) {
				vlog("ram: %d%% of budget (%s / %s stores) | nodes %s | tt %.1f%%",
					100*st/ttSize, fmtN(st), fmtN(ttSize),
					fmtN(e.ctl.nodes.Load()), ttFullPct())
			}
		}
		if st >= ttSize {
			e.ctl.stop.Store(true)
			e.stop = true
		}
	}
}

// repetition: has the current position occurred before (game + search path)?
func (e *Engine) isRepetition() bool {
	p := e.p
	n := len(p.hist)
	// walk back through positions with the same side to move; a match with any
	// earlier hash is scored as a draw inside the search (2-fold).
	for i := n - 3; i >= 0; i -= 2 {
		if p.hist[i] == p.hash {
			return true
		}
		// halfmove reset bounds how far repetitions can reach, but the clock is
		// not tracked per hist entry; scanning all is correct, just a bit wider.
	}
	return false
}

func mvvLva(p *Pos, m Move) int {
	victim := p.b[m.To]
	if m.Flags&fEP != 0 {
		victim = WP
	}
	if victim < 0 {
		victim = -victim
	}
	attacker := p.b[m.From]
	if attacker < 0 {
		attacker = -attacker
	}
	s := pieceVal[victim]*16 - pieceVal[attacker]/16
	if m.Promo != 0 {
		s += pieceVal[m.Promo]
	}
	return s
}

// sliderAttacker walks rays from (tx,ty) looking for `want`, treating
// removed pieces as transparent (x-ray support for SEE).
func (p *Pos) sliderAttacker(tx, ty int, dirs [][2]int, want int, removed *[64]bool) int {
	for _, d := range dirs {
		x, y := tx+d[0], ty+d[1]
		for x >= 0 && x <= 7 && y >= 0 && y <= 7 {
			sq := y*8 + x
			if !removed[sq] && p.b[sq] != Empty {
				if p.b[sq] == want {
					return sq
				}
				break // blocked
			}
			x += d[0]
			y += d[1]
		}
	}
	return -1
}

// leastAttacker returns the square of the least valuable piece of `side`
// attacking target (removed pieces are transparent); -1 if none.
func (p *Pos) leastAttacker(target, side int, removed *[64]bool) int {
	tx, ty := target&7, target>>3
	dir, pw, sign := 8, WP, 1
	if side == Black {
		dir, pw, sign = -8, BP, -1
	}
	if base := target - dir; base >= 1 && base <= 62 {
		if x := base - 1; tx > 0 && !removed[x] && p.b[x] == pw {
			return x
		}
		if x := base + 1; tx < 7 && !removed[x] && p.b[x] == pw {
			return x
		}
	}
	for _, d := range knightD {
		x, y := tx+d[0], ty+d[1]
		if x < 0 || x > 7 || y < 0 || y > 7 {
			continue
		}
		if sq := y*8 + x; !removed[sq] && p.b[sq] == sign*WN {
			return sq
		}
	}
	if sq := p.sliderAttacker(tx, ty, bishD[:], sign*WB, removed); sq >= 0 {
		return sq
	}
	if sq := p.sliderAttacker(tx, ty, rookD[:], sign*WR, removed); sq >= 0 {
		return sq
	}
	if sq := p.sliderAttacker(tx, ty, bishD[:], sign*WQ, removed); sq >= 0 {
		return sq
	}
	if sq := p.sliderAttacker(tx, ty, rookD[:], sign*WQ, removed); sq >= 0 {
		return sq
	}
	for _, d := range kingD {
		x, y := tx+d[0], ty+d[1]
		if x < 0 || x > 7 || y < 0 || y > 7 {
			continue
		}
		if sq := y*8 + x; !removed[sq] && p.b[sq] == sign*WK {
			return sq
		}
	}
	return -1
}

// see estimates the material outcome of the capture sequence on m.To
// (static exchange evaluation; pins are ignored - it's a pruning heuristic).
func (p *Pos) see(m Move) int {
	target := m.To
	var removed [64]bool
	victim := p.b[target]
	if m.Flags&fEP != 0 {
		victim = WP
		if p.side == White {
			removed[target-8] = true
		} else {
			removed[target+8] = true
		}
	}
	if victim < 0 {
		victim = -victim
	}
	var gain [40]int
	d := 0
	gain[0] = seeVal[victim]
	att := p.b[m.From]
	if att < 0 {
		att = -att
	}
	if m.Promo != 0 {
		gain[0] += seeVal[m.Promo] - seeVal[WP]
		att = m.Promo
	}
	removed[m.From] = true
	side := p.side ^ 1
	for d < 38 {
		sq := p.leastAttacker(target, side, &removed)
		if sq < 0 {
			break
		}
		d++
		gain[d] = seeVal[att] - gain[d-1]
		if gain[d] < 0 && gain[d-1] > 0 {
			break // neither continuation helps the side to move
		}
		att = p.b[sq]
		if att < 0 {
			att = -att
		}
		removed[sq] = true
		side ^= 1
	}
	for ; d > 0; d-- {
		if -gain[d] < gain[d-1] {
			gain[d-1] = -gain[d]
		}
	}
	return gain[0]
}

func (e *Engine) orderMoves(moves []Move, ttMove Move, ply int) {
	p := e.p
	prev := NoMove
	if ply >= 1 {
		prev = e.moveStack[ply-1]
	}
	counter := NoMove
	if prev != NoMove {
		counter = e.counter[p.side][prev.From][prev.To]
	}
	scores := make([]int, len(moves))
	for i, m := range moves {
		switch {
		case m == ttMove:
			scores[i] = 1 << 30
		case m.Flags&fCapture != 0:
			s := mvvLva(p, m) + int(*e.capHistOf(m))/16
			if m.Promo != 0 || p.see(m) >= 0 {
				scores[i] = (1 << 28) + s // good captures right after TT move
			} else {
				scores[i] = -(1 << 27) + s // bad captures dead last
			}
		case m.Promo != 0:
			scores[i] = (1 << 28) + pieceVal[m.Promo]
		case m == e.killers[ply][0]:
			scores[i] = 1 << 26
		case m == e.killers[ply][1]:
			scores[i] = (1 << 26) - 1
		case m == counter:
			scores[i] = 1 << 25
		default:
			scores[i] = e.quietHist(p.side, m, ply)
		}
	}
	// insertion sort by score desc (move lists are short)
	for i := 1; i < len(moves); i++ {
		m, s := moves[i], scores[i]
		j := i - 1
		for j >= 0 && scores[j] < s {
			moves[j+1], scores[j+1] = moves[j], scores[j]
			j--
		}
		moves[j+1], scores[j+1] = m, s
	}
}

func (e *Engine) quiesce(alpha, beta, ply int) int {
	e.nodes++
	if verbose && ply > e.seldepth {
		e.seldepth = ply
	}
	e.checkTime()
	if e.stop {
		return 0
	}
	p := e.p
	pvNode := beta-alpha > 1
	ttMove, ttScore, _, ttFlag, ttEval, ttHit := ttProbe(p.hash)
	if ttHit && !pvNode {
		s := ttScore
		if s > MateScore-2000 {
			s -= ply
		} else if s < -(MateScore - 2000) {
			s += ply
		}
		switch ttFlag {
		case ttExact:
			return s
		case ttAlpha:
			if s <= alpha {
				return s
			}
		case ttBeta:
			if s >= beta {
				return s
			}
		}
	}
	inChk := p.inCheck(p.side)
	stand := -Inf
	if !inChk { // no stand-pat while in check: must search all evasions
		if ttHit && ttEval != evalNone {
			stand = ttEval
		} else {
			stand = p.eval()
		}
		if stand >= beta {
			if !ttHit {
				ttStore(p.hash, NoMove, stand, 0, ttBeta, ply, stand)
			}
			return stand
		}
		if stand > alpha {
			alpha = stand
		}
	}
	if ply >= MaxPly-1 {
		if inChk {
			return 0
		}
		return stand
	}
	moves := p.genPseudo(!inChk) // in check: all evasions; else captures/promos
	e.orderMoves(moves, ttMove, ply)
	origAlpha := alpha
	best := stand
	bestM := NoMove
	legal := 0
	se := stand
	if inChk {
		se = evalNone
	}
	for _, m := range moves {
		if !inChk && m.Promo == 0 {
			victim := p.b[m.To]
			if m.Flags&fEP != 0 {
				victim = WP
			}
			if victim < 0 {
				victim = -victim
			}
			if stand+pieceVal[victim]+200 <= alpha {
				continue // delta pruning
			}
			if p.see(m) < 0 {
				continue // losing capture
			}
		}
		u := p.make(m)
		if p.attacked(p.kingSq[p.side^1], p.side) {
			p.unmake(m, u)
			continue
		}
		legal++
		e.moveStack[ply] = m
		e.pcStack[ply] = pieceIdx(p.b[m.To])*64 + m.To
		score := -e.quiesce(-beta, -alpha, ply+1)
		p.unmake(m, u)
		if e.stop {
			return 0
		}
		if score > best {
			best = score
			bestM = m
		}
		if score >= beta {
			ttStore(p.hash, bestM, best, 0, ttBeta, ply, se)
			return score
		}
		if score > alpha {
			alpha = score
		}
	}
	if inChk && legal == 0 {
		return -MateScore + ply // checkmated in quiescence
	}
	flag := ttAlpha
	if best > origAlpha {
		flag = ttExact
	}
	ttStore(p.hash, bestM, best, 0, flag, ply, se)
	return best
}

// lmrTable[depth][moveNumber] = log-based late-move reduction
var lmrTable [64][64]int

func initLMR() {
	for d := 1; d < 64; d++ {
		for i := 1; i < 64; i++ {
			lmrTable[d][i] = int(0.4 + math.Log(float64(d))*math.Log(float64(i))/2.2)
		}
	}
}

func imin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (e *Engine) search(depth, ply, alpha, beta int, nullOK bool) int {
	e.nodes++
	e.checkTime()
	if e.stop {
		return 0
	}
	p := e.p
	if ply > 0 {
		if p.half >= 100 || e.isRepetition() {
			return 0
		}
		// mate-distance pruning
		if a := -MateScore + ply; a > alpha {
			alpha = a
		}
		if b := MateScore - ply - 1; b < beta {
			beta = b
		}
		if alpha >= beta {
			return alpha
		}
	}
	inChk := p.inCheck(p.side)
	if inChk {
		depth++ // check extension
	}
	if depth <= 0 {
		return e.quiesce(alpha, beta, ply)
	}
	if ply >= MaxPly-1 {
		return p.eval()
	}
	pvNode := beta-alpha > 1
	excl := e.excluded[ply]
	ttMove, ttScore, ttDepth, ttFlag, ttEval, ttHit := ttProbe(p.hash)
	if ttHit && ply > 0 && !pvNode && excl == NoMove && ttDepth >= depth {
		s := ttScore
		if s > MateScore-2000 {
			s -= ply
		} else if s < -(MateScore - 2000) {
			s += ply
		}
		switch ttFlag {
		case ttExact:
			return s
		case ttAlpha:
			if s <= alpha {
				return s
			}
		case ttBeta:
			if s >= beta {
				return s
			}
		}
	}
	// internal iterative reduction: deep node without a TT move
	if ttMove == NoMove && depth >= 6 {
		depth--
	}
	rawEval := evalNone
	staticEval := evalNone
	improving := false
	if !inChk {
		if ttHit && ttEval != evalNone {
			rawEval = ttEval
		} else {
			rawEval = p.eval()
		}
		// correction history: pawn-structure-keyed eval adjustment
		staticEval = rawEval + int(e.corrHist[p.side][p.pawnKey&16383])/256
		e.evalStack[ply] = staticEval
		if ply >= 2 && e.evalStack[ply-2] != evalNone {
			improving = staticEval > e.evalStack[ply-2]
		} else {
			improving = true
		}
	} else {
		e.evalStack[ply] = evalNone
	}
	if !inChk && !pvNode && excl == NoMove && beta > -(MateScore-2000) {
		// razoring: hopeless nodes drop straight into quiescence
		if razorOn && depth <= 2 && staticEval+200+250*depth <= alpha {
			v := e.quiesce(alpha, beta, ply)
			if v <= alpha || e.stop {
				return v
			}
		}
		// reverse futility pruning
		if depth <= 8 && staticEval-(rfpBase-rfpImp*b2i(improving))*depth >= beta {
			return staticEval
		}
		// null-move pruning (adaptive reduction + high-depth verification)
		if nullOK && depth >= 3 && staticEval >= beta && p.hasNonPawnMaterial() {
			R := 3 + depth/6 + imin((staticEval-beta)/nullEvalDiv, 2)
			if R > depth-1 {
				R = depth - 1
			}
			savedEP := p.ep
			if p.ep >= 0 {
				p.hash ^= zEP[p.ep&7]
			}
			p.ep = -1
			p.side ^= 1
			p.hash ^= zSide
			p.hist = append(p.hist, p.hash)
			e.moveStack[ply] = NoMove
			e.pcStack[ply] = -1
			score := -e.search(depth-1-R, ply+1, -beta, -beta+1, false)
			p.hist = p.hist[:len(p.hist)-1]
			p.side ^= 1
			p.hash ^= zSide
			p.ep = savedEP
			if p.ep >= 0 {
				p.hash ^= zEP[p.ep&7]
			}
			if e.stop {
				return 0
			}
			if score >= beta {
				if score > MateScore-2000 {
					score = beta // don't trust null-move mate scores
				}
				// zugzwang guard: verify deep null cutoffs with a real search
				if depth >= 12 {
					v := e.search(depth-1-R, ply, beta-1, beta, false)
					if e.stop {
						return 0
					}
					if v >= beta {
						return score
					}
				} else {
					return score
				}
			}
		}
		// probcut: a good capture that beats beta by a margin at reduced
		// depth almost certainly beats beta at full depth
		if depth >= 5 && beta < MateScore-2000 {
			pcBeta := beta + 170
			caps := p.genPseudo(true)
			e.orderMoves(caps, NoMove, ply)
			tried := 0
			for _, m := range caps {
				if tried >= 3 {
					break
				}
				if m.Flags&fCapture != 0 && p.see(m) < 0 {
					continue
				}
				u := p.make(m)
				if p.attacked(p.kingSq[p.side^1], p.side) {
					p.unmake(m, u)
					continue
				}
				tried++
				e.moveStack[ply] = m
				e.pcStack[ply] = pieceIdx(p.b[m.To])*64 + m.To
				v := -e.quiesce(-pcBeta, -pcBeta+1, ply+1)
				if v >= pcBeta {
					v = -e.search(depth-4, ply+1, -pcBeta, -pcBeta+1, true)
				}
				p.unmake(m, u)
				if e.stop {
					return 0
				}
				if v >= pcBeta {
					ttStore(p.hash, m, v, depth-3, ttBeta, ply, rawEval)
					return v
				}
			}
		}
	}
	// singular extension probe: is the TT move the only good move here?
	ext := 0
	if ply > 0 && depth >= 8 && excl == NoMove && ttMove != NoMove &&
		ttFlag != ttAlpha && ttDepth >= depth-3 &&
		ttScore > -(MateScore-2000) && ttScore < MateScore-2000 {
		sBeta := ttScore - 2*depth
		e.excluded[ply] = ttMove
		v := e.search((depth-1)/2, ply, sBeta-1, sBeta, false)
		e.excluded[ply] = NoMove
		if e.stop {
			return 0
		}
		if v < sBeta {
			ext = 1 // every other move failed low: extend the singular move
		} else if sBeta >= beta {
			return sBeta // multicut: several moves already beat beta
		}
	}
	moves := p.genPseudo(false)
	e.orderMoves(moves, ttMove, ply)
	origAlpha := alpha
	legal, quiets := 0, 0
	bestScore := -Inf
	bestM := NoMove
	var triedQuiets [32]Move
	var triedCaps [32]Move
	nTQ, nTC := 0, 0
	for _, m := range moves {
		if m == excl {
			continue
		}
		isQuiet := m.Flags&fCapture == 0 && m.Promo == 0
		mHist := 0
		if isQuiet {
			mHist = e.quietHist(p.side, m, ply) // must read before make()
		}
		// prune late/hopeless moves (only after one legal move is confirmed)
		if !pvNode && !inChk && legal > 0 && bestScore > -(MateScore-2000) {
			if isQuiet {
				lmpLimit := lmpBase + depth*depth
				if lmpHalve && !improving {
					lmpLimit /= 2
				}
				if depth <= lmpDepth && quiets > lmpLimit {
					continue // late move pruning
				}
				if depth <= 8 && staticEval+90+100*depth+futImp*b2i(improving) <= alpha {
					continue // futility pruning
				}
				if depth <= 8 && p.see(m) < -25*depth*depth {
					continue // SEE pruning: quiet loses material badly
				}
			} else if depth <= 6 && m.Promo == 0 && p.see(m) < -100*depth {
				continue // SEE pruning: clearly losing capture
			}
		}
		u := p.make(m)
		if p.attacked(p.kingSq[p.side^1], p.side) {
			p.unmake(m, u)
			continue
		}
		legal++
		if isQuiet {
			quiets++
			if nTQ < 32 {
				triedQuiets[nTQ] = m
				nTQ++
			}
		} else if m.Flags&fCapture != 0 && nTC < 32 {
			triedCaps[nTC] = m
			nTC++
		}
		givesCheck := p.inCheck(p.side)
		e.moveStack[ply] = m
		e.pcStack[ply] = pieceIdx(p.b[m.To])*64 + m.To
		nd := depth - 1
		if m == ttMove {
			nd += ext
		}
		var score int
		if legal == 1 {
			score = -e.search(nd, ply+1, -beta, -alpha, true)
		} else {
			// late move reductions + principal variation search
			r := 0
			if isQuiet && depth >= 3 && !inChk && !givesCheck {
				r = lmrTable[imin(depth, 63)][imin(legal, 63)]
				r += lmrNotImp * b2i(!improving)
				if pvNode {
					r--
				}
				r -= mHist / 8192 // good history -> reduce less
				if r > depth-2 {
					r = depth - 2
				}
				if r < 0 {
					r = 0
				}
			}
			score = -e.search(nd-r, ply+1, -alpha-1, -alpha, true)
			if score > alpha && r > 0 {
				score = -e.search(nd, ply+1, -alpha-1, -alpha, true)
			}
			if score > alpha && score < beta {
				score = -e.search(nd, ply+1, -beta, -alpha, true)
			}
		}
		p.unmake(m, u)
		if e.stop {
			return 0
		}
		if score > bestScore {
			bestScore = score
			bestM = m
		}
		if score > alpha {
			alpha = score
		}
		if alpha >= beta {
			if isQuiet {
				if e.killers[ply][0] != m {
					e.killers[ply][1] = e.killers[ply][0]
					e.killers[ply][0] = m
				}
				e.updateQuiets(m, triedQuiets[:nTQ], p.side, depth, ply)
				if ply >= 1 && e.moveStack[ply-1] != NoMove {
					prev := e.moveStack[ply-1]
					e.counter[p.side][prev.From][prev.To] = m
				}
			} else if m.Flags&fCapture != 0 {
				bonus := imin(32*depth*depth+64*depth, 1280)
				histUp32(e.capHistOf(m), bonus)
				for i := 0; i < nTC; i++ {
					if triedCaps[i] != m {
						histUp32(e.capHistOf(triedCaps[i]), -bonus)
					}
				}
			}
			break
		}
	}
	if legal == 0 {
		if inChk {
			return -MateScore + ply
		}
		return 0
	}
	flag := ttExact
	if bestScore <= origAlpha {
		flag = ttAlpha
	} else if bestScore >= beta {
		flag = ttBeta
	}
	// correction history: teach the eval how far off it was here
	if !inChk && excl == NoMove && bestScore > -(MateScore-2000) &&
		bestScore < MateScore-2000 &&
		!(flag == ttBeta && bestScore <= staticEval) &&
		!(flag == ttAlpha && bestScore >= staticEval) {
		c := &e.corrHist[p.side][p.pawnKey&16383]
		w := int32(imin(depth, 15) + 1)
		nv := *c + (int32((bestScore-staticEval)*256)-*c)*w/256
		if nv > 16384 {
			nv = 16384
		} else if nv < -16384 {
			nv = -16384
		}
		*c = nv
	}
	if excl == NoMove {
		ttStore(p.hash, bestM, bestScore, depth, flag, ply, rawEval)
	}
	return bestScore
}

// clone makes an independent copy for a parallel search worker.
func (p *Pos) clone() *Pos {
	q := *p
	q.hist = append([]uint64(nil), p.hist...)
	return &q
}

// rootSearch scans the root move list with the given window (PVS).
func (e *Engine) rootSearch(moves []Move, depth, alpha, beta int) (int, Move) {
	bestScore := -Inf
	bestM := NoMove
	for i, m := range moves {
		u := e.p.make(m)
		e.moveStack[0] = m
		e.pcStack[0] = pieceIdx(e.p.b[m.To])*64 + m.To
		var score int
		if i == 0 {
			score = -e.search(depth-1, 1, -beta, -alpha, true)
		} else {
			score = -e.search(depth-1, 1, -alpha-1, -alpha, true)
			if score > alpha && score < beta {
				score = -e.search(depth-1, 1, -beta, -alpha, true)
			}
		}
		e.p.unmake(m, u)
		if e.stop {
			return bestScore, bestM
		}
		if score > bestScore {
			bestScore = score
			bestM = m
		}
		if score > alpha {
			alpha = score
			if alpha >= beta {
				break // fail high; aspiration loop widens and re-searches
			}
		}
	}
	return bestScore, bestM
}

// bestMove runs parallel (Lazy SMP) iterative deepening on all CPU cores
// with aspiration windows. Exactly one limit applies: budget > 0 =
// wall-clock seconds (enforced); depthCap > 0 = fixed depth; both zero =
// RAM-limited (thinks until one full hash table's worth of positions has
// been stored, then plays). Workers share the XOR-validated lock-free TT;
// played moves always come from the legal list, so races can never yield an
// illegal move.
func bestMove(p *Pos, budget, depthCap int) Move {
	searchStart := time.Now()
	legal := p.genLegal()
	if len(legal) == 0 {
		return NoMove
	}
	workers := runtime.NumCPU()
	if v := os.Getenv("CHESS_THREADS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= workers {
			workers = n
		}
	}
	if len(legal) == 1 {
		if verbose {
			vlog("forced: only one legal move -> %s", moveStr(legal[0]))
		}
		return legal[0]
	}
	maxD := MaxDepth
	if depthCap > 0 && depthCap < maxD {
		maxD = depthCap
	}
	c := &searchCtl{}
	if budget > 0 {
		c.hasDeadline = true
		c.deadline = searchStart.Add(time.Duration(budget) * time.Second)
		c.soft = searchStart.Add(time.Duration(budget) * time.Second * 55 / 100)
	} else if depthCap == 0 {
		// 0 = RAM-limited: no clock, no depth cap; think until this move has
		// stored one full hash table's worth of positions, then play.
		c.memLimited = true
		ttStores.Store(0)
		if verbose {
			c.nextMark.Store((ttSize + 9) / 10)
		}
	}
	if verbose {
		var mode string
		switch {
		case budget > 0:
			mode = fmt.Sprintf("TIME %d s (soft-stop %.1f s)", budget, float64(budget)*0.55)
		case depthCap > 0:
			mode = fmt.Sprintf("DEPTH %d", depthCap)
		default:
			mode = fmt.Sprintf("RAM %s stores / %d MB hash", fmtN(ttSize), ttSize*ttEntryBytes>>20)
		}
		side := "white"
		if p.side == Black {
			side = "black"
		}
		vlog("search: %s to move | limit: %s | legal: %d | threads: %d",
			side, mode, len(legal), workers)
	}
	var mu sync.Mutex
	best := legal[0]
	bestDepth, bestScore := 0, 0
	ttGen.Add(1) // age the shared hash table once per move
	// engines persist across moves (one per thread): history, countermove
	// and continuation-history knowledge carries over into the next search.
	poolOnce.Do(func() { enginePool = make([]*Engine, workers) })
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			e := enginePool[w]
			if e == nil {
				e = &Engine{}
				enginePool[w] = e
			}
			e.p = p.clone()
			e.ctl = c
			e.stop = false
			e.nodes = 0
			e.seldepth = 0
			e.killers = [MaxPly][2]Move{}
			e.moveStack = [MaxPly]Move{}
			e.evalStack = [MaxPly]int{}
			e.pcStack = [MaxPly]int{}
			moves := append([]Move(nil), legal...)
			myBest := moves[0]
			prevScore := 0
			if e.p.inCheck(e.p.side) {
				e.evalStack[0] = evalNone
			} else {
				e.evalStack[0] = e.p.eval()
			}
			startD := 1 + (w & 1) // odd helpers start one ply deeper
			if startD > maxD {
				startD = maxD
			}
			for depth := startD; depth <= maxD; depth++ {
				if c.hasDeadline && depth > startD && time.Now().After(c.soft) {
					break // a new iteration would very likely not finish
				}
				e.orderMoves(moves, myBest, 0)
				window := 30
				alpha, beta := -Inf, Inf
				if depth >= 5 {
					alpha, beta = prevScore-window, prevScore+window
				}
				var iterScore int
				var iterBest Move
				for { // aspiration window re-search loop
					iterScore, iterBest = e.rootSearch(moves, depth, alpha, beta)
					if e.stop || (iterScore > alpha && iterScore < beta) {
						break
					}
					window *= 3
					failLow := iterScore <= alpha
					if failLow {
						alpha = iterScore - window
					} else {
						beta = iterScore + window
					}
					if verbose {
						dir := "high"
						if failLow {
							dir = "low"
						}
						vlog("[w%02d] depth %d fail-%s (%s) -> re-search window (%d, %d)",
							w, depth, dir, scoreStr(iterScore), alpha, beta)
					}
					if alpha < -MateScore {
						alpha = -Inf
					}
					if beta > MateScore {
						beta = Inf
					}
				}
				if e.stop {
					break // discard partial iteration
				}
				if iterBest != NoMove {
					myBest = iterBest
					prevScore = iterScore
					ttStore(e.p.hash, myBest, iterScore, depth, ttExact, 0, e.evalStack[0])
					mu.Lock()
					if depth > bestDepth {
						bestDepth = depth
						best = myBest
						bestScore = iterScore
						if verbose {
							el := time.Since(searchStart).Seconds()
							nodes := c.nodes.Load()
							var nps uint64
							if el > 0 {
								nps = uint64(float64(nodes) / el)
							}
							vlog("depth %2d sel %2d | %s | best %s | nodes %s | nps %s | tt %.1f%% | pv %s",
								depth, e.seldepth, scoreStr(iterScore), moveStr(myBest),
								fmtN(nodes), fmtN(nps), ttFullPct(),
								pvString(p, myBest, imin(depth, 12)))
						}
					}
					mu.Unlock()
				}
				if iterScore > MateScore-2000 || iterScore < -(MateScore-2000) {
					break // proven mate either way; deeper search can't change it
				}
			}
			// first worker to finish (mate, depth cap or deadline) stops the rest
			c.stop.Store(true)
		}(w)
	}
	wg.Wait()
	if verbose {
		var nodes uint64
		sel := 0
		for _, e := range enginePool {
			if e != nil {
				nodes += e.nodes
				if e.seldepth > sel {
					sel = e.seldepth
				}
			}
		}
		el := time.Since(searchStart).Seconds()
		var nps uint64
		if el > 0 {
			nps = uint64(float64(nodes) / el)
		}
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		vlog("played %s | depth %d sel %d | %s | nodes %s | nps %s | time %.2f s | tt %.1f%% | heap %d MB (sys %d MB)",
			moveStr(best), bestDepth, sel, scoreStr(bestScore), fmtN(nodes),
			fmtN(nps), el, ttFullPct(), ms.HeapAlloc>>20, ms.Sys>>20)
	}
	return best
}

// ---------- move I/O ----------

func sqName(sq int) string {
	return string([]byte{byte('a' + sq&7), byte('1' + sq>>3)})
}

func moveStr(m Move) string {
	s := sqName(m.From) + sqName(m.To)
	if m.Promo != 0 {
		s += string("  nbrq"[m.Promo])
	}
	return s
}

// parseMove matches user input against the legal move list.
func parseMove(p *Pos, in string, legal []Move) (Move, string) {
	if len(in) != 4 && len(in) != 5 {
		return NoMove, "cannot parse (use coordinate notation like e2e4, e7e8q)"
	}
	if in[0] < 'a' || in[0] > 'h' || in[1] < '1' || in[1] > '8' ||
		in[2] < 'a' || in[2] > 'h' || in[3] < '1' || in[3] > '8' {
		return NoMove, "cannot parse (use coordinate notation like e2e4, e7e8q)"
	}
	from := int(in[1]-'1')*8 + int(in[0]-'a')
	to := int(in[3]-'1')*8 + int(in[2]-'a')
	promo := 0
	if len(in) == 5 {
		switch in[4] {
		case 'q':
			promo = WQ
		case 'r':
			promo = WR
		case 'b':
			promo = WB
		case 'n':
			promo = WN
		default:
			return NoMove, "bad promotion piece (use q, r, b or n)"
		}
	}
	needPromo := false
	for _, m := range legal {
		if m.From == from && m.To == to {
			if m.Promo == promo {
				return m, ""
			}
			if m.Promo != 0 && promo == 0 {
				needPromo = true
			}
		}
	}
	if needPromo {
		return NoMove, "promotion move: append q, r, b or n (e.g. e7e8q)"
	}
	return NoMove, "illegal move"
}

// ---------- board display ----------

func drawBoard(p *Pos, w *bufio.Writer) {
	for r := 7; r >= 0; r-- {
		row := make([]byte, 8)
		for f := 0; f < 8; f++ {
			pc := p.b[r*8+f]
			switch {
			case pc == Empty:
				row[f] = ' '
			case pc > 0:
				row[f] = "PNBRQK"[pc-1]
			default:
				row[f] = "pnbrqk"[-pc-1]
			}
		}
		w.Write(row)
		w.WriteByte('\n')
	}
	w.Flush()
}

// ---------- game state ----------

func insufficientMaterial(p *Pos) bool {
	var bishops []int
	knights := 0
	for sq := 0; sq < 64; sq++ {
		switch p.b[sq] {
		case Empty, WK, BK:
		case WB, BB:
			bishops = append(bishops, sq)
		case WN, BN:
			knights++
		default:
			return false // pawn, rook or queen on the board
		}
	}
	pieces := len(bishops) + knights
	if pieces == 0 || pieces == 1 {
		return true // K vs K, K+minor vs K
	}
	if knights == 0 {
		// only bishops: draw if all are on the same square color
		color := (bishops[0]>>3 + bishops[0]&7) & 1
		for _, sq := range bishops[1:] {
			if (sq>>3+sq&7)&1 != color {
				return false
			}
		}
		return true
	}
	return false
}

func repetitionCount(p *Pos) int {
	n := 0
	for _, h := range p.hist {
		if h == p.hash {
			n++
		}
	}
	return n
}

// gameResult returns ("", "") while the game goes on, otherwise result+reason.
func gameResult(p *Pos) (string, string) {
	if len(p.genLegal()) == 0 {
		if p.inCheck(p.side) {
			if p.side == White {
				return "0-1", "checkmate"
			}
			return "1-0", "checkmate"
		}
		return "1/2-1/2", "stalemate"
	}
	if p.half >= 100 {
		return "1/2-1/2", "fifty-move"
	}
	if repetitionCount(p) >= 3 {
		return "1/2-1/2", "repetition"
	}
	if insufficientMaterial(p) {
		return "1/2-1/2", "material"
	}
	return "", ""
}

// ---------- main game loop ----------

func main() {
	initTT()
	initZobrist()
	initMasks()
	initLMR()
	if os.Getenv("CHESS_SELFTEST") == "1" {
		selfTest()
		return
	}
	budget, depthCap := 600, 0
	demo, silent, gui := false, false, false
	args := os.Args[1:]
	for len(args) > 0 { // leading mode keywords combine: e.g. "gui demo"
		if strings.EqualFold(args[0], "demo") {
			demo = true
		} else if strings.EqualFold(args[0], "silentdemo") || strings.EqualFold(args[0], "sdemo") {
			demo, silent = true, true
		} else if strings.EqualFold(args[0], "gui") {
			gui = true
		} else {
			break
		}
		args = args[1:]
	}
	usage := func() {
		fmt.Fprintln(os.Stderr, "usage: chess [demo|silentdemo] [limit]  (exactly one limit per mode)")
		fmt.Fprintln(os.Stderr, "  limit > 0: TIME-limited - max seconds per move (default 600)")
		fmt.Fprintln(os.Stderr, "  limit < 0: DEPTH-limited - fixed search depth (-10 = depth 10)")
		fmt.Fprintln(os.Stderr, "  limit = 0: RAM-limited - hash auto-sized from available RAM")
		fmt.Fprintln(os.Stderr, "             (CHESS_HASH_MB overrides); moves when it's used up")
		fmt.Fprintln(os.Stderr, "  demo: computer plays both sides, board redrawn after every move")
		fmt.Fprintln(os.Stderr, "  silentdemo: same but without the boards (moves + result only)")
		fmt.Fprintln(os.Stderr, "  gui: X11 board window - drag & drop your moves (same limits)")
		fmt.Fprintln(os.Stderr, "  gui demo: X11 window where the engine plays both sides")
		fmt.Fprintln(os.Stderr, "  env: VERBOSE=1 full search/debug trace on stderr")
		fmt.Fprintln(os.Stderr, "       CHESS_HASH_MB=<mb> hash size | CHESS_FEN=<fen> start position")
		os.Exit(1)
	}
	if len(args) > 1 {
		usage()
	}
	if len(args) == 1 {
		v, err := strconv.Atoi(args[0])
		if err != nil {
			usage()
		}
		switch {
		case v > 0:
			budget = v
		case v < 0:
			budget, depthCap = 0, -v
		default:
			budget = 0 // truly unlimited
		}
	}
	p := startPos()
	if fen := os.Getenv("CHESS_FEN"); fen != "" {
		p = fromFEN(fen) // testing aid: start from an arbitrary position
	}
	if gui {
		guiLoop(p, budget, depthCap, demo)
		return
	}
	out := bufio.NewWriter(os.Stdout)
	scanner := bufio.NewScanner(os.Stdin)

	// a CHESS_FEN start may already be game over
	if res, why := gameResult(p); res != "" {
		fmt.Fprintln(out, res+" "+why)
		out.Flush()
		return
	}

	// play makes the move, prints it, announces check/result.
	// Returns false when the game is over (result already printed).
	play := func(m Move, announce bool) bool {
		p.make(m)
		if verbose {
			vlog("pos after %s: %s | key %016x", moveStr(m), toFEN(p), p.hash)
		}
		if announce {
			fmt.Fprintln(out, moveStr(m))
			out.Flush()
		}
		if res, why := gameResult(p); res != "" {
			fmt.Fprintln(out, res+" "+why)
			out.Flush()
			return false
		}
		if p.inCheck(p.side) {
			fmt.Fprintln(out, "check")
			out.Flush()
		}
		return true
	}

	// DEMO mode: the computer plays both sides (as if the human typed "c"
	// forever); board redrawn after every move ("d" after each) unless
	// silent (silentdemo), which only skips the board printouts.
	// Same limit semantics as normal mode.
	if demo {
		oneshot := os.Getenv("CHESS_ONESHOT") == "1" // testing aid: one move, then exit
		for {
			m := bestMove(p, budget, depthCap)
			p.make(m)
			if verbose {
				vlog("pos after %s: %s | key %016x", moveStr(m), toFEN(p), p.hash)
			}
			fmt.Fprintln(out, moveStr(m))
			if silent {
				out.Flush() // drawBoard would otherwise flush
			} else {
				drawBoard(p, out)
			}
			if res, why := gameResult(p); res != "" {
				fmt.Fprintln(out, res+" "+why)
				out.Flush()
				return
			}
			if p.inCheck(p.side) {
				fmt.Fprintln(out, "check")
				out.Flush()
			}
			if oneshot {
				out.Flush()
				return
			}
		}
	}

	for scanner.Scan() {
		line := strings.ToLower(strings.TrimSpace(scanner.Text()))
		switch line {
		case "":
			continue
		case "d":
			drawBoard(p, out)
			continue
		case "c":
			// computer plays the user's (side to move) move, then replies
			m := bestMove(p, budget, depthCap)
			if !play(m, true) {
				return
			}
			rm := bestMove(p, budget, depthCap)
			if !play(rm, true) {
				return
			}
			continue
		}
		legal := p.genLegal()
		m, errMsg := parseMove(p, line, legal)
		if errMsg != "" {
			fmt.Fprintln(os.Stderr, errMsg)
			continue
		}
		if !play(m, false) {
			return
		}
		rm := bestMove(p, budget, depthCap)
		if !play(rm, true) {
			return
		}
	}
}

// ---------- self test: FEN + perft + zobrist consistency ----------

func fromFEN(fen string) *Pos {
	p := &Pos{ep: -1}
	parts := strings.Fields(fen)
	sq := 56
	for _, c := range parts[0] {
		switch {
		case c == '/':
			sq -= 16
		case c >= '1' && c <= '8':
			sq += int(c - '0')
		default:
			pc := 0
			switch c {
			case 'P':
				pc = WP
			case 'N':
				pc = WN
			case 'B':
				pc = WB
			case 'R':
				pc = WR
			case 'Q':
				pc = WQ
			case 'K':
				pc = WK
			case 'p':
				pc = BP
			case 'n':
				pc = BN
			case 'b':
				pc = BB
			case 'r':
				pc = BR
			case 'q':
				pc = BQ
			case 'k':
				pc = BK
			}
			p.b[sq] = pc
			if pc == WK {
				p.kingSq[White] = sq
			}
			if pc == BK {
				p.kingSq[Black] = sq
			}
			sq++
		}
	}
	if len(parts) > 1 && parts[1] == "b" {
		p.side = Black
	}
	if len(parts) > 2 {
		for _, c := range parts[2] {
			switch c {
			case 'K':
				p.rights |= CastleWK
			case 'Q':
				p.rights |= CastleWQ
			case 'k':
				p.rights |= CastleBK
			case 'q':
				p.rights |= CastleBQ
			}
		}
	}
	if len(parts) > 3 && parts[3] != "-" {
		p.ep = int(parts[3][1]-'1')*8 + int(parts[3][0]-'a')
	}
	p.hash = p.computeHash()
	p.pawnKey = p.computePawnKey()
	p.hist = append(p.hist, p.hash)
	return p
}

func perft(p *Pos, d int) uint64 {
	if d == 0 {
		return 1
	}
	var n uint64
	for _, m := range p.genPseudo(false) {
		u := p.make(m)
		if !p.attacked(p.kingSq[p.side^1], p.side) {
			if p.pawnKey != p.computePawnKey() {
				panic("pawnKey mismatch")
			}
			if p.hash != p.computeHash() {
				fmt.Println("ZOBRIST MISMATCH after", moveStr(m))
				os.Exit(1)
			}
			n += perft(p, d-1)
		}
		p.unmake(m, u)
	}
	return n
}

func selfTest() {
	type tc struct {
		name  string
		fen   string
		depth int
		want  uint64
	}
	start := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	cases := []tc{
		{"start-d1", start, 1, 20},
		{"start-d2", start, 2, 400},
		{"start-d3", start, 3, 8902},
		{"start-d4", start, 4, 197281},
		{"start-d5", start, 5, 4865609},
		{"kiwipete-d1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
		{"kiwipete-d2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
		{"kiwipete-d3", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
		{"kiwipete-d4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
		{"pos3-d5", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
		{"pos4-d4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
		{"pos5-d3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
		{"pos6-d3", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890},
	}
	fail := 0
	for _, c := range cases {
		p := fromFEN(c.fen)
		t0 := time.Now()
		got := perft(p, c.depth)
		dt := time.Since(t0)
		status := "PASS"
		if got != c.want {
			status = "FAIL"
			fail++
		}
		fmt.Printf("%-12s want %-10d got %-10d %-4s (%.2fs)\n", c.name, c.want, got, status, dt.Seconds())
	}
	if fail > 0 {
		fmt.Printf("%d FAILURES\n", fail)
		os.Exit(1)
	}
	fmt.Println("ALL PERFT TESTS PASSED (zobrist verified at every node)")
}

// ---------- optional GUI mode: a raw-X11 (wire protocol) chessboard ----------
// Zero dependencies kept: this speaks the core X11 protocol directly over the
// unix socket (works with any X server: real, XWayland, Xvfb, ...).

type x11Conn struct {
	c         net.Conn
	buf       *bufio.Writer
	seq       uint16
	ridBase   uint32
	ridNext   uint32
	root      uint32
	rootDepth byte
	visual    uint32
	cmap      uint32
	white     uint32
	black     uint32
	events    chan []byte
	replies   chan []byte
	readErr   error
}

func le16(b []byte, v uint16) { b[0] = byte(v); b[1] = byte(v >> 8) }
func le32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}
func rd16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }
func rd32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
func pad4(n int) int { return (4 - n&3) & 3 }

// xAuthCookie finds an MIT-MAGIC-COOKIE-1 for this display in ~/.Xauthority.
func xAuthCookie(disp string) (name string, data []byte) {
	path := os.Getenv("XAUTHORITY")
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", nil
		}
		path = home + "/.Xauthority"
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", nil
	}
	dispNum := strings.TrimPrefix(disp, ":")
	if i := strings.IndexByte(dispNum, '.'); i >= 0 {
		dispNum = dispNum[:i]
	}
	var fbName string
	var fbData []byte
	for off := 0; off+2 <= len(raw); {
		rdBE := func() ([]byte, bool) {
			if off+2 > len(raw) {
				return nil, false
			}
			n := int(raw[off])<<8 | int(raw[off+1])
			off += 2
			if off+n > len(raw) {
				return nil, false
			}
			s := raw[off : off+n]
			off += n
			return s, true
		}
		off += 2                  // family
		if _, ok := rdBE(); !ok { // address
			break
		}
		dpy, ok := rdBE()
		if !ok {
			break
		}
		nm, ok := rdBE()
		if !ok {
			break
		}
		dt, ok := rdBE()
		if !ok {
			break
		}
		if string(nm) != "MIT-MAGIC-COOKIE-1" {
			continue
		}
		if string(dpy) == dispNum || len(dpy) == 0 {
			return string(nm), dt
		}
		if fbName == "" {
			fbName, fbData = string(nm), dt
		}
	}
	return fbName, fbData
}

func x11Dial() (*x11Conn, error) {
	disp := os.Getenv("DISPLAY")
	if disp == "" {
		disp = ":0"
	}
	num := disp
	if i := strings.LastIndexByte(num, ':'); i >= 0 {
		num = num[i+1:]
	}
	if i := strings.IndexByte(num, '.'); i >= 0 {
		num = num[:i]
	}
	var c net.Conn
	var err error
	if strings.HasPrefix(disp, ":") || strings.HasPrefix(disp, "unix:") {
		c, err = net.Dial("unix", "/tmp/.X11-unix/X"+num)
	} else { // host:N
		host := disp[:strings.LastIndexByte(disp, ':')]
		port := 6000
		if v, e := strconv.Atoi(num); e == nil {
			port += v
		}
		c, err = net.DialTimeout("tcp", net.JoinHostPort(host, strconv.Itoa(port)), 5*time.Second)
	}
	if err != nil {
		return nil, err
	}
	authName, authData := xAuthCookie(disp)
	req := make([]byte, 12+len(authName)+pad4(len(authName))+len(authData)+pad4(len(authData)))
	req[0] = 'l'
	le16(req[2:], 11)
	le16(req[6:], uint16(len(authName)))
	le16(req[8:], uint16(len(authData)))
	copy(req[12:], authName)
	copy(req[12+len(authName)+pad4(len(authName)):], authData)
	if _, err = c.Write(req); err != nil {
		return nil, err
	}
	head := make([]byte, 8)
	if _, err = io.ReadFull(c, head); err != nil {
		return nil, err
	}
	body := make([]byte, int(rd16(head[6:]))*4)
	if _, err = io.ReadFull(c, body); err != nil {
		return nil, err
	}
	if head[0] != 1 {
		n := int(head[1])
		msg := ""
		if 8+n <= len(body)+8 && n <= len(body) {
			msg = string(body[:n])
		}
		return nil, fmt.Errorf("X11 connection refused: %s", msg)
	}
	x := &x11Conn{c: c, buf: bufio.NewWriter(c)}
	x.ridBase = rd32(body[4:])
	vlen := int(rd16(body[16:]))
	nform := int(body[21])
	off := 32 + vlen + pad4(vlen) + 8*nform
	// first screen
	x.root = rd32(body[off:])
	x.cmap = rd32(body[off+4:])
	x.white = rd32(body[off+8:])
	x.black = rd32(body[off+12:])
	x.visual = rd32(body[off+32:])
	x.rootDepth = body[off+38]
	x.events = make(chan []byte, 64)
	x.replies = make(chan []byte, 4)
	go x.reader()
	return x, nil
}

func (x *x11Conn) newID() uint32 {
	id := x.ridBase | x.ridNext
	x.ridNext++
	return id
}

// reader pumps server messages: replies to x.replies, events to x.events.
func (x *x11Conn) reader() {
	r := bufio.NewReader(x.c)
	for {
		m := make([]byte, 32)
		if _, err := io.ReadFull(r, m); err != nil {
			x.readErr = err
			close(x.events)
			return
		}
		switch m[0] {
		case 1: // reply, possibly with extra data
			if extra := rd32(m[4:]); extra > 0 {
				rest := make([]byte, extra*4)
				if _, err := io.ReadFull(r, rest); err != nil {
					x.readErr = err
					close(x.events)
					return
				}
				m = append(m, rest...)
			}
			x.replies <- m
		case 0: // protocol error: report and continue
			fmt.Fprintf(os.Stderr, "X11 error: code=%d major=%d\n", m[1], m[10])
		default:
			x.events <- m
		}
	}
}

func (x *x11Conn) send(b []byte) {
	x.seq++
	x.buf.Write(b)
}

func (x *x11Conn) flush() { x.buf.Flush() }

func (x *x11Conn) roundTrip(b []byte) []byte {
	x.send(b)
	x.flush()
	return <-x.replies
}

func (x *x11Conn) internAtom(name string) uint32 {
	n := len(name)
	b := make([]byte, 8+n+pad4(n))
	b[0] = 16
	le16(b[2:], uint16(len(b)/4))
	le16(b[4:], uint16(n))
	copy(b[8:], name)
	r := x.roundTrip(b)
	return rd32(r[8:])
}

func (x *x11Conn) allocColor(rr, gg, bb uint16) (uint32, bool) {
	b := make([]byte, 16)
	b[0] = 84
	le16(b[2:], 4)
	le32(b[4:], x.cmap)
	le16(b[8:], rr)
	le16(b[10:], gg)
	le16(b[12:], bb)
	r := x.roundTrip(b)
	if r[0] != 1 {
		return 0, false
	}
	return rd32(r[8:]), true
}

func (x *x11Conn) createWindow(w, h int, evMask uint32) uint32 {
	wid := x.newID()
	b := make([]byte, 32+8)
	b[0] = 1 // CreateWindow, depth CopyFromParent
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], wid)
	le32(b[8:], x.root)
	le16(b[16:], uint16(w))
	le16(b[18:], uint16(h))
	le16(b[20:], 0)         // border width
	le16(b[22:], 1)         // InputOutput
	le32(b[24:], 0)         // visual CopyFromParent
	le32(b[28:], 0x2|0x800) // background-pixel | event-mask
	le32(b[32:], x.white)   // background = white
	le32(b[36:], evMask)
	x.send(b)
	return wid
}

func (x *x11Conn) mapWindow(wid uint32) {
	b := make([]byte, 8)
	b[0] = 8
	le16(b[2:], 2)
	le32(b[4:], wid)
	x.send(b)
}

func (x *x11Conn) changeProp32(wid, prop, typ uint32, vals []uint32) {
	b := make([]byte, 24+4*len(vals))
	b[0] = 18
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], wid)
	le32(b[8:], prop)
	le32(b[12:], typ)
	b[16] = 32
	le32(b[20:], uint32(len(vals)))
	for i, v := range vals {
		le32(b[24+4*i:], v)
	}
	x.send(b)
}

func (x *x11Conn) changeProp8(wid, prop, typ uint32, data string) {
	n := len(data)
	b := make([]byte, 24+n+pad4(n))
	b[0] = 18
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], wid)
	le32(b[8:], prop)
	le32(b[12:], typ)
	b[16] = 8
	le32(b[20:], uint32(n))
	copy(b[24:], data)
	x.send(b)
}

func (x *x11Conn) openFont(name string) uint32 {
	fid := x.newID()
	n := len(name)
	b := make([]byte, 12+n+pad4(n))
	b[0] = 45
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], fid)
	le16(b[8:], uint16(n))
	copy(b[12:], name)
	x.send(b)
	return fid
}

func (x *x11Conn) createGC(drawable, fg, bg, font uint32) uint32 {
	gc := x.newID()
	mask := uint32(0x4 | 0x8) // foreground | background
	vals := []uint32{fg, bg}
	if font != 0 {
		mask |= 0x4000
		vals = append(vals, font)
	}
	b := make([]byte, 16+4*len(vals))
	b[0] = 55
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], gc)
	le32(b[8:], drawable)
	le32(b[12:], mask)
	for i, v := range vals {
		le32(b[16+4*i:], v)
	}
	x.send(b)
	return gc
}

func (x *x11Conn) setFG(gc, fg uint32) {
	b := make([]byte, 16)
	b[0] = 56
	le16(b[2:], 4)
	le32(b[4:], gc)
	le32(b[8:], 0x4)
	le32(b[12:], fg)
	x.send(b)
}

func (x *x11Conn) setBG(gc, bg uint32) {
	b := make([]byte, 16)
	b[0] = 56
	le16(b[2:], 4)
	le32(b[4:], gc)
	le32(b[8:], 0x8)
	le32(b[12:], bg)
	x.send(b)
}

func (x *x11Conn) fillRect(d, gc uint32, xx, yy, w, h int) {
	b := make([]byte, 20)
	b[0] = 70
	le16(b[2:], 5)
	le32(b[4:], d)
	le32(b[8:], gc)
	le16(b[12:], uint16(int16(xx)))
	le16(b[14:], uint16(int16(yy)))
	le16(b[16:], uint16(w))
	le16(b[18:], uint16(h))
	x.send(b)
}

func (x *x11Conn) drawRect(d, gc uint32, xx, yy, w, h int) {
	b := make([]byte, 20)
	b[0] = 67
	le16(b[2:], 5)
	le32(b[4:], d)
	le32(b[8:], gc)
	le16(b[12:], uint16(int16(xx)))
	le16(b[14:], uint16(int16(yy)))
	le16(b[16:], uint16(w))
	le16(b[18:], uint16(h))
	x.send(b)
}

func (x *x11Conn) text(d, gc uint32, xx, yy int, s string) {
	if len(s) > 255 {
		s = s[:255]
	}
	n := len(s)
	b := make([]byte, 16+n+pad4(n))
	b[0] = 76
	b[1] = byte(n)
	le16(b[2:], uint16(len(b)/4))
	le32(b[4:], d)
	le32(b[8:], gc)
	le16(b[12:], uint16(int16(xx)))
	le16(b[14:], uint16(int16(yy)))
	copy(b[16:], s)
	x.send(b)
}

// guiLoop runs the chess GUI: an 8x8 board window where the human drags (or
// click-clicks) White's moves, with a button that makes the computer move for
// the human (the text-mode `c` command). Moves, checks and the result are
// still printed on stdout exactly like text mode.
func guiLoop(p *Pos, budget, depthCap int, demo bool) {
	x, err := x11Dial()
	if err != nil {
		fmt.Fprintln(os.Stderr, "gui: cannot open X display:", err)
		os.Exit(1)
	}
	const (
		SQ      = 64
		BW      = 8 * SQ
		statusY = BW
		statusH = 40
		btnY    = BW + statusH
		btnH    = 44
		WinW    = BW
		WinH    = BW + statusH + btnH
	)
	evMask := uint32(1<<2 | 1<<3 | 1<<15 | 1<<17) // press, release, exposure, structure
	win := x.createWindow(WinW, WinH, evMask)
	x.changeProp8(win, 39, 31, "chess-fable") // WM_NAME
	wmProtocols := x.internAtom("WM_PROTOCOLS")
	wmDelete := x.internAtom("WM_DELETE_WINDOW")
	x.changeProp32(win, wmProtocols, 4, []uint32{wmDelete})
	// lock the window size (PMinSize|PMaxSize)
	hints := make([]uint32, 18)
	hints[0] = 16 | 32
	hints[5], hints[6] = WinW, WinH
	hints[7], hints[8] = WinW, WinH
	x.changeProp32(win, 40, 41, hints) // WM_NORMAL_HINTS / WM_SIZE_HINTS
	gray := x.white
	if px, ok := x.allocColor(0xb8b8, 0xb8b8, 0xb8b8); ok {
		gray = px
	}
	font := x.openFont("10x20")
	gc := x.createGC(win, x.black, x.white, font)
	x.mapWindow(win)
	x.flush()

	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	sel := -1
	lastFrom, lastTo := -1, -1
	thinking := false
	pendingCPU := 0 // engine moves left to play in a row (1 reply, 2 for CPU btn)
	over := false
	status := "You are White: drag or click-click a move."
	if demo {
		status = "Demo: engine plays both sides."
	}
	promoFrom, promoTo := -1, -1
	pressSq, pressBtn := -1, false
	engCh := make(chan Move, 1)

	if res, why := gameResult(p); res != "" {
		fmt.Fprintln(out, res+" "+why)
		out.Flush()
		status = res + " " + why
		over = true
	}

	sqAt := func(px, py int) int {
		if px < 0 || px >= BW || py < 0 || py >= BW {
			return -1
		}
		return (7-py/SQ)*8 + px/SQ
	}
	inBtn := func(px, py int) bool {
		return px >= 8 && px < WinW-8 && py >= btnY+2 && py < btnY+btnH-6
	}

	pieceGlyph := func(pc int) (byte, bool) { // letter, isWhite
		if pc > 0 {
			return "PNBRQK"[pc-1], true
		}
		return "PNBRQK"[-pc-1], false
	}

	draw := func() {
		// board squares
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				xx, yy := f*SQ, (7-r)*SQ
				if (f+r)&1 == 0 {
					x.setFG(gc, gray)
				} else {
					x.setFG(gc, x.white)
				}
				x.fillRect(win, gc, xx, yy, SQ, SQ)
			}
		}
		// grid + last-move + selection markers
		x.setFG(gc, x.black)
		for i := 0; i <= 8; i++ {
			x.fillRect(win, gc, i*SQ-1, 0, 1, BW)
			x.fillRect(win, gc, 0, i*SQ-1, BW, 1)
		}
		mark := func(sq int, w int) {
			if sq < 0 {
				return
			}
			f, r := sq&7, sq>>3
			for i := 0; i < w; i++ {
				x.drawRect(win, gc, f*SQ+i, (7-r)*SQ+i, SQ-1-2*i, SQ-1-2*i)
			}
		}
		x.setFG(gc, x.black)
		mark(lastFrom, 2)
		mark(lastTo, 2)
		mark(sel, 4)
		// pieces: letter on a round-ish badge - white badge/black ink for
		// White, black badge/white ink for Black (readable on any square)
		for sq := 0; sq < 64; sq++ {
			pc := p.b[sq]
			if pc == Empty {
				continue
			}
			g, isW := pieceGlyph(pc)
			s := string(g)
			f, r := sq&7, sq>>3
			bx, by := f*SQ+SQ/2-14, (7-r)*SQ+SQ/2-16
			fill, ink := x.black, x.white
			if isW {
				fill, ink = x.white, x.black
			}
			x.setFG(gc, fill)
			x.fillRect(win, gc, bx, by, 28, 32)
			x.setFG(gc, ink)
			x.drawRect(win, gc, bx, by, 27, 31)
			x.drawRect(win, gc, bx+1, by+1, 25, 29)
			x.setBG(gc, fill)
			cx, cy := f*SQ+SQ/2-5, (7-r)*SQ+SQ/2+7
			x.text(win, gc, cx, cy, s)
			x.text(win, gc, cx+1, cy, s) // fake bold
			x.setBG(gc, x.white)
		}
		// status bar
		x.setFG(gc, x.white)
		x.fillRect(win, gc, 0, statusY, WinW, statusH+btnH)
		x.setFG(gc, x.black)
		x.text(win, gc, 8, statusY+26, status)
		// CPU button (hidden in demo mode - the engine plays itself)
		if !demo {
			label := "CPU MOVE (plays for you)"
			if thinking {
				label = "THINKING..."
			}
			x.drawRect(win, gc, 8, btnY+2, WinW-16, btnH-8)
			x.drawRect(win, gc, 9, btnY+3, WinW-18, btnH-10)
			x.text(win, gc, WinW/2-len(label)*10/2, btnY+btnH/2+4, label)
		} else if thinking {
			x.text(win, gc, WinW/2-55, btnY+btnH/2+4, "THINKING...")
		}
		// promotion chooser overlay
		if promoFrom >= 0 {
			x.setFG(gc, x.white)
			x.fillRect(win, gc, 2*SQ-8, 3*SQ+SQ/2-8, 4*SQ+16, SQ+16)
			x.setFG(gc, x.black)
			x.drawRect(win, gc, 2*SQ-8, 3*SQ+SQ/2-8, 4*SQ+15, SQ+15)
			for i, c := range "QRBN" {
				xx, yy := (2+i)*SQ, 3*SQ+SQ/2
				x.drawRect(win, gc, xx, yy, SQ-1, SQ-1)
				x.text(win, gc, xx+SQ/2-5, yy+SQ/2+7, string(c))
			}
		}
		x.flush()
	}

	applyMove := func(m Move) {
		p.make(m)
		lastFrom, lastTo = m.From, m.To
		fmt.Fprintln(out, moveStr(m))
		if res, why := gameResult(p); res != "" {
			fmt.Fprintln(out, res+" "+why)
			status = res + " " + why
			over = true
		} else if p.inCheck(p.side) {
			fmt.Fprintln(out, "check")
			status = "check!"
		} else if p.side == White && !demo {
			status = "Your move."
		}
		out.Flush()
	}

	startEngine := func(n int) { // n engine moves in a row
		if over || thinking {
			return
		}
		thinking = true
		pendingCPU = n
		status = "Thinking..."
		go func() { engCh <- bestMove(p, budget, depthCap) }()
	}

	tryHuman := func(from, to int) {
		if over || thinking || p.side != White {
			return
		}
		var cands []Move
		for _, m := range p.genLegal() {
			if m.From == from && m.To == to {
				cands = append(cands, m)
			}
		}
		if len(cands) == 0 {
			status = "Illegal move."
			return
		}
		if cands[0].Promo != 0 {
			promoFrom, promoTo = from, to
			status = "Choose promotion piece."
			return
		}
		sel = -1
		applyMove(cands[0])
		if !over {
			startEngine(1)
		}
	}

	promoPick := func(px, py int) {
		if py < 3*SQ+SQ/2 || py >= 3*SQ+SQ/2+SQ || px < 2*SQ || px >= 6*SQ {
			return // click outside: keep chooser open
		}
		promos := [4]int{WQ, WR, WB, WN}
		pr := promos[(px-2*SQ)/SQ]
		for _, m := range p.genLegal() {
			if m.From == promoFrom && m.To == promoTo && m.Promo == pr {
				promoFrom, promoTo = -1, -1
				sel = -1
				applyMove(m)
				if !over {
					startEngine(1)
				}
				return
			}
		}
	}

	if demo && !over {
		startEngine(1) // demo: the engine opens as White and never stops
	}

	for {
		select {
		case m := <-engCh:
			thinking = false
			pendingCPU--
			applyMove(m)
			if !over && (demo || pendingCPU > 0) {
				startEngine(max(pendingCPU, 1))
			} else {
				pendingCPU = 0
			}
			draw()
		case ev, ok := <-x.events:
			if !ok {
				return // display gone
			}
			switch ev[0] & 0x7f {
			case 12: // Expose
				if rd16(ev[16:]) == 0 {
					draw()
				}
			case 4: // ButtonPress
				if demo {
					break // demo: board is not interactive
				}
				px, py := int(int16(rd16(ev[24:]))), int(int16(rd16(ev[26:])))
				if promoFrom >= 0 {
					promoPick(px, py)
					draw()
					break
				}
				pressSq = sqAt(px, py)
				pressBtn = inBtn(px, py)
			case 5: // ButtonRelease
				if demo {
					break
				}
				px, py := int(int16(rd16(ev[24:]))), int(int16(rd16(ev[26:])))
				if promoFrom >= 0 {
					break
				}
				if pressBtn && inBtn(px, py) {
					pressBtn = false
					if !over && !thinking && p.side == White {
						startEngine(2) // your move + the reply
					}
					draw()
					break
				}
				pressBtn = false
				rel := sqAt(px, py)
				if rel < 0 || pressSq < 0 {
					break
				}
				if rel == pressSq { // click-click mode
					if sel >= 0 && rel != sel {
						tryHuman(sel, rel)
					} else if p.b[rel] > 0 && p.side == White && !thinking && !over {
						sel = rel
						status = "Now click the destination square."
					} else {
						sel = -1
					}
				} else { // drag
					tryHuman(pressSq, rel)
					sel = -1
				}
				pressSq = -1
				draw()
			case 33: // ClientMessage (window close)
				if rd32(ev[12:]) == wmDelete {
					return
				}
			}
		}
	}
}
