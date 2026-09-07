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
// Build: go build -o chess chess.go
// Self-test (perft + zobrist): CHESS_SELFTEST=1 ./chess
package main

import (
	"bufio"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
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
}

type Pos struct {
	b      [64]int
	side   int
	rights int
	ep     int // -1 = none
	half   int // halfmove clock
	hash   uint64
	kingSq [2]int
	hist   []uint64 // hashes after every move (incl. start position)
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
	u := Undo{captured: Empty, capSq: m.To, rights: p.rights, ep: p.ep, half: p.half, hash: p.hash}
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
		p.b[u.capSq] = Empty
	} else if p.b[m.To] != Empty {
		u.captured = p.b[m.To]
		p.hash ^= zPiece[pidx(u.captured)][m.To]
	}
	p.hash ^= zPiece[pidx(pc)][m.From]
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
func (p *Pos) eval() int {
	var mg, eg, phase int
	var wpF, bpF [10]uint16 // pawn rank-bitmask per file, index = file+1
	for sq := 0; sq < 64; sq++ {
		switch p.b[sq] {
		case WP:
			wpF[sq&7+1] |= 1 << uint(sq>>3)
		case BP:
			bpF[sq&7+1] |= 1 << uint(sq>>3)
		}
	}
	var bishops [2]int
	for sq := 0; sq < 64; sq++ {
		pc := p.b[sq]
		if pc == Empty {
			continue
		}
		f, r := sq&7, sq>>3
		if pc > 0 {
			phase += phaseVal[pc]
			mg += mgVal[pc] + mgPST[pc][sq^56]
			eg += egVal[pc] + egPST[pc][sq^56]
			switch pc {
			case WP:
				if wpF[f]|wpF[f+2] == 0 { // isolated (neighbors are idx f, f+2)
					mg -= 12
					eg -= 9
				}
				if (bpF[f]|bpF[f+1]|bpF[f+2])&(^uint16(0)<<uint(r+1)) == 0 { // passed
					mg += passedMG[r]
					eg += passedEG[r]
				}
			case WB:
				bishops[0]++
			case WR:
				if wpF[f+1] == 0 {
					if bpF[f+1] == 0 {
						mg += 25
						eg += 12
					} else {
						mg += 12
						eg += 6
					}
				}
			}
		} else {
			pc = -pc
			phase += phaseVal[pc]
			mg -= mgVal[pc] + mgPST[pc][sq]
			eg -= egVal[pc] + egPST[pc][sq]
			switch pc {
			case WP:
				if bpF[f]|bpF[f+2] == 0 {
					mg += 12
					eg += 9
				}
				if (wpF[f]|wpF[f+1]|wpF[f+2])&((1<<uint(r))-1) == 0 {
					mg -= passedMG[7-r]
					eg -= passedEG[7-r]
				}
			case WB:
				bishops[1]++
			case WR:
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
		}
	}
	for f := 1; f <= 8; f++ { // doubled pawns (per extra pawn on a file)
		if c := bits.OnesCount16(wpF[f]); c > 1 {
			mg -= 11 * (c - 1)
			eg -= 17 * (c - 1)
		}
		if c := bits.OnesCount16(bpF[f]); c > 1 {
			mg += 11 * (c - 1)
			eg += 17 * (c - 1)
		}
	}
	if bishops[0] >= 2 {
		mg += 24
		eg += 45
	}
	if bishops[1] >= 2 {
		mg -= 24
		eg -= 45
	}
	// king pawn shield (middlegame only): own pawns 1-2 ranks ahead of the
	// king on his and adjacent files; extra penalty for fully open files.
	wk, bk := p.kingSq[White], p.kingSq[Black]
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

// Two atomically-accessed words per entry with xkey = key ^ data: a torn or
// racing write fails validation at probe time and is simply ignored, making
// the shared table safe for Lazy SMP without locks (Stockfish-style).
type ttEntry struct {
	xkey uint64
	data uint64
}

const ttEntryBytes = 16

var ttSize uint64 // entries, always a power of two
var ttTable []ttEntry

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
// (~40% of available, floor 64 MB, cap 16 GB, CHESS_HASH_MB env overrides)
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
		budget = availableRAM() * 2 / 5
		if max := uint64(16) << 30; budget > max {
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
	if os.Getenv("CHESS_VERBOSE") == "1" {
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

func ttProbe(hash uint64) (Move, int, int, int, bool) {
	e := &ttTable[hash&(ttSize-1)]
	d := atomic.LoadUint64(&e.data)
	if atomic.LoadUint64(&e.xkey)^d != hash {
		return NoMove, 0, 0, 0, false
	}
	return decMove(d), int(int16(uint16(d >> 32))), int(uint8(d >> 48)), int(d >> 56 & 3), true
}

var ttStores atomic.Uint64 // stores since the current 0-mode search began

func ttStore(hash uint64, m Move, score, depth, flag, ply int) {
	if score > MateScore-2000 {
		score += ply
	} else if score < -(MateScore - 2000) {
		score -= ply
	}
	d := encMove(m) | uint64(uint16(int16(score)))<<32 | uint64(uint8(depth))<<48 | uint64(flag)<<56
	e := &ttTable[hash&(ttSize-1)]
	atomic.StoreUint64(&e.xkey, hash^d)
	atomic.StoreUint64(&e.data, d)
	ttStores.Add(1)
}

// ---------- search ----------

// searchCtl is shared by all parallel search workers.
type searchCtl struct {
	stop        atomic.Bool
	hasDeadline bool
	deadline    time.Time // hard stop, enforced inside the search
	soft        time.Time // don't start another iteration after this
	memLimited  bool      // 0-mode: stop when the hash capacity is used up
}

type Engine struct {
	p         *Pos
	ctl       *searchCtl
	stop      bool
	nodes     uint64
	killers   [MaxPly][2]Move
	history   [2][64][64]int32
	counter   [2][64][64]Move // countermove heuristic, indexed by prev move
	moveStack [MaxPly]Move    // move that led to each ply (for countermoves)
}

func (e *Engine) checkTime() {
	if e.nodes&1023 != 0 {
		return
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
	if e.ctl.memLimited && ttStores.Load() >= ttSize {
		e.ctl.stop.Store(true)
		e.stop = true
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

func (e *Engine) orderMoves(moves []Move, ttMove Move, ply int, prev Move) {
	p := e.p
	counter := NoMove
	if prev != NoMove {
		counter = e.counter[p.side][prev.From][prev.To]
	}
	scores := make([]int, len(moves))
	for i, m := range moves {
		switch {
		case m == ttMove:
			scores[i] = 1 << 30
		case m.Flags&fCapture != 0 || m.Promo != 0:
			scores[i] = (1 << 28) + mvvLva(p, m)
		case m == e.killers[ply][0]:
			scores[i] = 1 << 26
		case m == e.killers[ply][1]:
			scores[i] = (1 << 26) - 1
		case m == counter:
			scores[i] = 1 << 25
		default:
			scores[i] = int(e.history[p.side][m.From][m.To])
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
	e.checkTime()
	if e.stop {
		return 0
	}
	p := e.p
	inChk := p.inCheck(p.side)
	stand := -Inf
	if !inChk { // no stand-pat while in check: must search all evasions
		stand = p.eval()
		if stand >= beta {
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
	e.orderMoves(moves, NoMove, ply, NoMove)
	best := stand
	legal := 0
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
		score := -e.quiesce(-beta, -alpha, ply+1)
		p.unmake(m, u)
		if e.stop {
			return 0
		}
		if score > best {
			best = score
		}
		if score >= beta {
			return score
		}
		if score > alpha {
			alpha = score
		}
	}
	if inChk && legal == 0 {
		return -MateScore + ply // checkmated in quiescence
	}
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
	ttMove, ttScore, ttDepth, ttFlag, ttHit := ttProbe(p.hash)
	if ttHit && ply > 0 && !pvNode && ttDepth >= depth {
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
	if !ttHit && depth >= 6 {
		depth--
	}
	staticEval := 0
	if !inChk {
		staticEval = p.eval()
		if !pvNode && beta > -(MateScore-2000) {
			// reverse futility pruning
			if depth <= 8 && staticEval-75*depth >= beta {
				return staticEval
			}
			// null-move pruning (adaptive reduction)
			if nullOK && depth >= 3 && staticEval >= beta && p.hasNonPawnMaterial() {
				R := 3 + depth/6
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
					return score
				}
			}
		}
	}
	moves := p.genPseudo(false)
	prev := NoMove
	if ply > 0 {
		prev = e.moveStack[ply-1]
	}
	e.orderMoves(moves, ttMove, ply, prev)
	origAlpha := alpha
	legal, quiets := 0, 0
	bestScore := -Inf
	bestM := NoMove
	for _, m := range moves {
		isQuiet := m.Flags&fCapture == 0 && m.Promo == 0
		// prune late/hopeless quiets (only after one legal move is confirmed)
		if isQuiet && !pvNode && !inChk && legal > 0 && bestScore > -(MateScore-2000) {
			if depth <= 5 && quiets > 4+depth*depth {
				continue // late move pruning
			}
			if depth <= 8 && staticEval+90+100*depth <= alpha {
				continue // futility pruning
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
		}
		givesCheck := p.inCheck(p.side)
		e.moveStack[ply] = m
		var score int
		if legal == 1 {
			score = -e.search(depth-1, ply+1, -beta, -alpha, true)
		} else {
			// late move reductions + principal variation search
			r := 0
			if isQuiet && depth >= 3 && !inChk && !givesCheck {
				r = lmrTable[imin(depth, 63)][imin(legal, 63)]
				if pvNode && r > 0 {
					r--
				}
				if r > depth-2 {
					r = depth - 2
				}
				if r < 0 {
					r = 0
				}
			}
			score = -e.search(depth-1-r, ply+1, -alpha-1, -alpha, true)
			if score > alpha && r > 0 {
				score = -e.search(depth-1, ply+1, -alpha-1, -alpha, true)
			}
			if score > alpha && score < beta {
				score = -e.search(depth-1, ply+1, -beta, -alpha, true)
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
				h := &e.history[p.side][m.From][m.To]
				*h += int32(depth * depth)
				if *h > 1<<20 {
					for f := 0; f < 64; f++ {
						for t := 0; t < 64; t++ {
							e.history[p.side][f][t] /= 2
						}
					}
				}
				if prev != NoMove {
					e.counter[p.side][prev.From][prev.To] = m
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
	ttStore(p.hash, bestM, bestScore, depth, flag, ply)
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
	legal := p.genLegal()
	if len(legal) == 0 {
		return NoMove
	}
	if len(legal) == 1 {
		return legal[0]
	}
	maxD := MaxDepth
	if depthCap > 0 && depthCap < maxD {
		maxD = depthCap
	}
	c := &searchCtl{}
	if budget > 0 {
		start := time.Now()
		c.hasDeadline = true
		c.deadline = start.Add(time.Duration(budget) * time.Second)
		c.soft = start.Add(time.Duration(budget) * time.Second * 55 / 100)
	} else if depthCap == 0 {
		// 0 = RAM-limited: no clock, no depth cap; think until this move has
		// stored one full hash table's worth of positions, then play.
		c.memLimited = true
		ttStores.Store(0)
	}
	var mu sync.Mutex
	best := legal[0]
	bestDepth := 0
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			e := &Engine{p: p.clone(), ctl: c}
			moves := append([]Move(nil), legal...)
			myBest := moves[0]
			prevScore := 0
			startD := 1 + (w & 1) // odd helpers start one ply deeper
			if startD > maxD {
				startD = maxD
			}
			for depth := startD; depth <= maxD; depth++ {
				if c.hasDeadline && depth > startD && time.Now().After(c.soft) {
					break // a new iteration would very likely not finish
				}
				e.orderMoves(moves, myBest, 0, NoMove)
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
					if iterScore <= alpha {
						alpha = iterScore - window
					} else {
						beta = iterScore + window
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
					ttStore(e.p.hash, myBest, iterScore, depth, ttExact, 0)
					mu.Lock()
					if depth > bestDepth {
						bestDepth = depth
						best = myBest
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
	demo := false
	args := os.Args[1:]
	if len(args) > 0 && strings.EqualFold(args[0], "demo") {
		demo = true
		args = args[1:]
	}
	usage := func() {
		fmt.Fprintln(os.Stderr, "usage: chess [demo] [limit]  (exactly one limit per mode)")
		fmt.Fprintln(os.Stderr, "  limit > 0: TIME-limited - max seconds per move (default 600)")
		fmt.Fprintln(os.Stderr, "  limit < 0: DEPTH-limited - fixed search depth (-10 = depth 10)")
		fmt.Fprintln(os.Stderr, "  limit = 0: RAM-limited - hash auto-sized from available RAM")
		fmt.Fprintln(os.Stderr, "             (CHESS_HASH_MB overrides); moves when it's used up")
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
	// forever) and the board is redrawn after every move ("d" after each).
	// Same think-time semantics as normal mode.
	if demo {
		for {
			m := bestMove(p, budget, depthCap)
			p.make(m)
			fmt.Fprintln(out, moveStr(m))
			drawBoard(p, out)
			if res, why := gameResult(p); res != "" {
				fmt.Fprintln(out, res+" "+why)
				out.Flush()
				return
			}
			if p.inCheck(p.side) {
				fmt.Fprintln(out, "check")
				out.Flush()
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
