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
//	            auto-sized to ~80% of available RAM (CHESS_HASH_MB env
//	            overrides) and the move is played once one full table's
//	            worth of positions has been stored. More RAM = deeper.
//
// Memory can never OOM: all data is fixed-size + a Go runtime memory limit.
// Searches in parallel on all CPU cores (Lazy SMP, shared lock-free TT);
// the evaluation is a small neural network (its weights end this file).
// Extra modes: "demo" = self-play with the board after every move,
// "silentdemo" = the same without boards, "gui" = an X11 board window
// (--help lists its options); VERBOSE=1 env = full search/debug trace on
// stderr (zero overhead when off; stdout stays protocol).
// Build: go build -o chess chess.go
// Self-test (perft + zobrist): CHESS_SELFTEST=1 ./chess
// Tests: go test chess.go chess_test.go
package main

import (
	"bufio"
	"bytes"
	"compress/flate"
	"encoding/base64"
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

// Move packs from (6 bits), to (6), the promotion piece (4: 0 or WN..WQ)
// and the flags (4), the layout the hash table stores.
type Move uint32

const NoMove Move = 0

func mkMove(from, to, promo, flags int) Move {
	return Move(from | to<<6 | promo<<12 | flags<<16)
}

func (m Move) From() int  { return int(m & 63) }
func (m Move) To() int    { return int(m >> 6 & 63) }
func (m Move) Promo() int { return int(m >> 12 & 15) }
func (m Move) Flags() int { return int(m >> 16 & 15) }

type Undo struct {
	captured int
	capSq    int
	rights   int
	ep       int
	half     int
	rev      int
	hash     uint64
	pawnKey  uint64
}

type Pos struct {
	b       [64]int
	bb      [2][7]uint64 // piece sets by color and type (WP..WK)
	occ     [2]uint64    // all men of each color
	side    int
	rights  int
	ep      int // -1 = none
	half    int // halfmove clock
	rev     int // positions in hist since the last irreversible move
	hash    uint64
	pawnKey uint64
	kingSq  [2]int
	hist    []uint64 // hashes after every move (incl. start position)
	nn      *nnStack // accumulators of the evaluation network (search only)
}

// set puts piece pc on the empty square sq (board and piece sets); clr
// takes it off again.
func (p *Pos) set(pc, sq int) {
	c, t := White, pc
	if pc < 0 {
		c, t = Black, -pc
	}
	bit := uint64(1) << uint(sq)
	p.b[sq] = pc
	p.bb[c][t] |= bit
	p.occ[c] |= bit
}

func (p *Pos) clr(pc, sq int) {
	c, t := White, pc
	if pc < 0 {
		c, t = Black, -pc
	}
	bit := uint64(1) << uint(sq)
	p.b[sq] = Empty
	p.bb[c][t] &^= bit
	p.occ[c] &^= bit
}

// consistent reports whether the piece sets match the board.
func (p *Pos) consistent() bool {
	var q Pos
	for sq, pc := range p.b {
		if pc != Empty {
			q.set(pc, sq)
		}
	}
	return q.bb == p.bb && q.occ == p.occ
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
		p.set(back[f], f)
		p.set(WP, 8+f)
		p.set(BP, 48+f)
		p.set(-back[f], 56+f)
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

// attacked reports whether square sq is attacked by side "by".
func (p *Pos) attacked(sq, by int) bool {
	bb := &p.bb[by]
	if pawnAttBB[by^1][sq]&bb[WP] != 0 || knightBB[sq]&bb[WN] != 0 || kingBB[sq]&bb[WK] != 0 {
		return true
	}
	occ := p.occ[White] | p.occ[Black]
	if d := bb[WB] | bb[WQ]; d&bishopMask[sq] != 0 && bishopAtt(sq, occ)&d != 0 {
		return true
	}
	l := bb[WR] | bb[WQ]
	return l&rookMask[sq] != 0 && rookAtt(sq, occ)&l != 0
}

func (p *Pos) inCheck(side int) bool {
	return p.attacked(p.kingSq[side], side^1)
}

// ---------- move generation (pseudo-legal) ----------

func (p *Pos) genPseudo(capturesOnly bool) []Move {
	return p.gen(make([]Move, 0, 64), capturesOnly)
}

// knightSeq and kingSeq list the squares a knight or king on sq can reach,
// in the order of knightD and kingD.
var knightSeq, kingSeq [64][]int8

// rayUp tells whether direction d of rayBB runs towards higher squares.
var rayUp = [8]bool{true, false, true, false, true, true, false, false}

func init() {
	for sq := 0; sq < 64; sq++ {
		r, f := sq>>3, sq&7
		for _, d := range knightD {
			if nr, nf := r+d[0], f+d[1]; nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7 {
				knightSeq[sq] = append(knightSeq[sq], int8(nr*8+nf))
			}
		}
		for _, d := range kingD {
			if nr, nf := r+d[0], f+d[1]; nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7 {
				kingSeq[sq] = append(kingSeq[sq], int8(nr*8+nf))
			}
		}
	}
}

// gen appends the pseudo-legal moves of the side to move to moves; with
// capturesOnly just the captures and queen promotions. The order is fixed
// (and part of the search): own men by square, each in its direction order.
func (p *Pos) gen(moves []Move, capturesOnly bool) []Move {
	us := p.side
	enemy := p.occ[us^1]
	occ := p.occ[us] | enemy
	for own := p.occ[us]; own != 0; own &= own - 1 {
		sq := bits.TrailingZeros64(own)
		pc := p.b[sq]
		if pc < 0 {
			pc = -pc
		}
		switch pc {
		case WP:
			r, f := sq>>3, sq&7
			dir, startR, promoR := 8, 1, 6
			if us == Black {
				dir, startR, promoR = -8, 6, 1
			}
			// pushes
			if to := sq + dir; to >= 0 && to < 64 && occ>>uint(to)&1 == 0 {
				if r == promoR {
					if !capturesOnly {
						moves = append(moves, mkMove(sq, to, WQ, 0), mkMove(sq, to, WR, 0),
							mkMove(sq, to, WB, 0), mkMove(sq, to, WN, 0))
					} else {
						// queen promotion counts as a "loud" move in quiescence
						moves = append(moves, mkMove(sq, to, WQ, 0))
					}
				} else if !capturesOnly {
					moves = append(moves, mkMove(sq, to, 0, 0))
					if r == startR && occ>>uint(sq+2*dir)&1 == 0 {
						moves = append(moves, mkMove(sq, sq+2*dir, 0, fDouble))
					}
				}
			}
			// captures
			for df := -1; df <= 1; df += 2 {
				to := sq + dir + df
				if f+df < 0 || f+df > 7 || to < 0 || to > 63 {
					continue
				}
				if enemy>>uint(to)&1 != 0 {
					if r == promoR {
						moves = append(moves, mkMove(sq, to, WQ, fCapture), mkMove(sq, to, WR, fCapture),
							mkMove(sq, to, WB, fCapture), mkMove(sq, to, WN, fCapture))
					} else {
						moves = append(moves, mkMove(sq, to, 0, fCapture))
					}
				} else if to == p.ep {
					moves = append(moves, mkMove(sq, to, 0, fCapture|fEP))
				}
			}
		case WN, WK:
			seq, reach := knightSeq[sq], knightBB[sq]
			if pc == WK {
				seq, reach = kingSeq[sq], kingBB[sq]
			}
			if !capturesOnly || reach&enemy != 0 {
				for _, t := range seq {
					to := int(t)
					if occ>>uint(to)&1 == 0 {
						if !capturesOnly {
							moves = append(moves, mkMove(sq, to, 0, 0))
						}
					} else if enemy>>uint(to)&1 != 0 {
						moves = append(moves, mkMove(sq, to, 0, fCapture))
					}
				}
			}
			if pc == WN || capturesOnly {
				break
			}
			// castling (king not in check, path empty, transit squares safe)
			if us == White && sq == 4 {
				if p.rights&CastleWK != 0 && p.b[5] == Empty && p.b[6] == Empty &&
					!p.attacked(4, Black) && !p.attacked(5, Black) && !p.attacked(6, Black) {
					moves = append(moves, mkMove(4, 6, 0, fCastle))
				}
				if p.rights&CastleWQ != 0 && p.b[3] == Empty && p.b[2] == Empty && p.b[1] == Empty &&
					!p.attacked(4, Black) && !p.attacked(3, Black) && !p.attacked(2, Black) {
					moves = append(moves, mkMove(4, 2, 0, fCastle))
				}
			} else if us == Black && sq == 60 {
				if p.rights&CastleBK != 0 && p.b[61] == Empty && p.b[62] == Empty &&
					!p.attacked(60, White) && !p.attacked(61, White) && !p.attacked(62, White) {
					moves = append(moves, mkMove(60, 62, 0, fCastle))
				}
				if p.rights&CastleBQ != 0 && p.b[59] == Empty && p.b[58] == Empty && p.b[57] == Empty &&
					!p.attacked(60, White) && !p.attacked(59, White) && !p.attacked(58, White) {
					moves = append(moves, mkMove(60, 58, 0, fCastle))
				}
			}
		default: // sliders: rook rays 0..3, bishop rays 4..7, queen all
			d0, d1 := 0, 8
			if pc == WR {
				d1 = 4
			} else if pc == WB {
				d0 = 4
			}
			for d := d0; d < d1; d++ {
				quiet := rayBB[d][sq]
				blockers := quiet & occ
				if blockers == 0 && capturesOnly {
					continue
				}
				up, hit := rayUp[d], -1
				if blockers != 0 {
					if up {
						hit = bits.TrailingZeros64(blockers)
					} else {
						hit = 63 - bits.LeadingZeros64(blockers)
					}
					quiet ^= rayBB[d][hit] | 1<<uint(hit)
				}
				if !capturesOnly { // outwards from the piece
					if up {
						for ; quiet != 0; quiet &= quiet - 1 {
							moves = append(moves, mkMove(sq, bits.TrailingZeros64(quiet), 0, 0))
						}
					} else {
						for quiet != 0 {
							to := 63 - bits.LeadingZeros64(quiet)
							quiet &^= 1 << uint(to)
							moves = append(moves, mkMove(sq, to, 0, 0))
						}
					}
				}
				if hit >= 0 && enemy>>uint(hit)&1 != 0 {
					moves = append(moves, mkMove(sq, hit, 0, fCapture))
				}
			}
		}
	}
	return moves
}

// ---------- make / unmake ----------

// epLegal reports whether side by can legally capture en passant on square
// ep, the pawn that just made a double step standing behind it. Only then
// is ep part of the position (its hash, so threefold repetition, and the
// move generation), as the rules of chess have it.
func (p *Pos) epLegal(ep, by int) bool {
	victim := ep + 8
	if by == White {
		victim = ep - 8
	}
	them := &p.bb[by^1]
	k := p.kingSq[by]
	all := p.occ[White] | p.occ[Black]
	for from := pawnAttBB[by^1][ep] & p.bb[by][WP]; from != 0; from &= from - 1 {
		occ := all&^(from&-from|1<<uint(victim)) | 1<<uint(ep)
		if pawnAttBB[by][k]&them[WP]&^(1<<uint(victim)) == 0 && knightBB[k]&them[WN] == 0 &&
			kingBB[k]&them[WK] == 0 && bishopAtt(k, occ)&(them[WB]|them[WQ]) == 0 &&
			rookAtt(k, occ)&(them[WR]|them[WQ]) == 0 {
			return true
		}
	}
	return false
}

// castleRook gives the rook's squares for a castling king arriving on to.
func castleRook(to int) (from, dest int) {
	switch to {
	case 6:
		return 7, 5
	case 2:
		return 0, 3
	case 62:
		return 63, 61
	}
	return 56, 59
}

func (p *Pos) make(m Move) Undo {
	from, to := m.From(), m.To()
	u := Undo{captured: Empty, capSq: to, rights: p.rights, ep: p.ep, half: p.half, rev: p.rev,
		hash: p.hash, pawnKey: p.pawnKey}
	pc := p.b[from]
	mover := p.side
	if m.Flags()&fEP != 0 {
		if mover == White {
			u.capSq = to - 8
		} else {
			u.capSq = to + 8
		}
		u.captured = p.b[u.capSq]
		p.hash ^= zPiece[pidx(u.captured)][u.capSq]
		p.pawnKey ^= zPiece[pidx(u.captured)][u.capSq] // EP victim is a pawn
		p.clr(u.captured, u.capSq)
	} else if p.b[to] != Empty {
		u.captured = p.b[to]
		p.hash ^= zPiece[pidx(u.captured)][to]
		if u.captured == WP || u.captured == BP {
			p.pawnKey ^= zPiece[pidx(u.captured)][to]
		}
		p.clr(u.captured, to)
	}
	p.hash ^= zPiece[pidx(pc)][from]
	if pc == WP || pc == BP {
		p.pawnKey ^= zPiece[pidx(pc)][from]
	}
	p.clr(pc, from)
	np := pc
	if promo := m.Promo(); promo != 0 {
		if mover == White {
			np = promo
		} else {
			np = -promo
		}
	}
	p.set(np, to)
	p.hash ^= zPiece[pidx(np)][to]
	if np == WP || np == BP {
		p.pawnKey ^= zPiece[pidx(np)][to]
	}
	if m.Flags()&fCastle != 0 {
		rf, rt := castleRook(to)
		rk := p.b[rf]
		p.hash ^= zPiece[pidx(rk)][rf] ^ zPiece[pidx(rk)][rt]
		p.clr(rk, rf)
		p.set(rk, rt)
	}
	if pc == WK {
		p.kingSq[White] = to
	} else if pc == BK {
		p.kingSq[Black] = to
	}
	p.hash ^= zCastle[p.rights]
	p.rights &= castleMask[from] & castleMask[to]
	p.hash ^= zCastle[p.rights]
	if p.ep >= 0 {
		p.hash ^= zEP[p.ep&7]
	}
	p.ep = -1
	if m.Flags()&fDouble != 0 {
		if ep := (from + to) / 2; p.epLegal(ep, mover^1) {
			p.ep = ep
			p.hash ^= zEP[ep&7]
		}
	}
	if pc == WP || pc == BP || u.captured != Empty {
		p.half, p.rev = 0, 0
	} else {
		p.half++
		p.rev++
	}
	p.side ^= 1
	p.hash ^= zSide
	p.hist = append(p.hist, p.hash)
	if p.nn != nil {
		p.nn.push(p, pc, np, from, to, u.captured, u.capSq, m.Flags()&fCastle != 0)
	}
	return u
}

func (p *Pos) unmake(m Move, u Undo) {
	if s := p.nn; s != nil {
		if s.over > 0 {
			s.over--
		} else {
			s.at--
		}
	}
	p.hist = p.hist[:len(p.hist)-1]
	p.side ^= 1
	mover := p.side
	from, to := m.From(), m.To()
	np := p.b[to]
	pc := np
	if m.Promo() != 0 {
		if mover == White {
			pc = WP
		} else {
			pc = BP
		}
	}
	p.clr(np, to)
	p.set(pc, from)
	if u.captured != Empty {
		p.set(u.captured, u.capSq)
	}
	if m.Flags()&fCastle != 0 {
		rf, rt := castleRook(to)
		rk := p.b[rt]
		p.clr(rk, rt)
		p.set(rk, rf)
	}
	if pc == WK {
		p.kingSq[White] = from
	} else if pc == BK {
		p.kingSq[Black] = from
	}
	p.rights = u.rights
	p.ep = u.ep
	p.half = u.half
	p.rev = u.rev
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

// ---------- piece values ----------

// pieceVal is used for move ordering (MVV-LVA) and pruning margins.
var pieceVal = [7]int{0, 100, 320, 330, 500, 900, 0}

// seeVal is used by static exchange evaluation.
var seeVal = [7]int{0, 100, 325, 335, 500, 975, 20000}

// ---------- bitboards ----------

const (
	fileABB      = uint64(0x0101010101010101)
	fileHBB      = uint64(0x8080808080808080)
	lightSquares = uint64(0x55AA55AA55AA55AA)
)

var queenD = [8][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}

var knightBB, kingBB [64]uint64

// rayBB[d][sq] are the squares from sq to the edge in direction d, in the
// order of queenD (rook lines, then diagonals): 0 N, 1 S, 2 E, 3 W, 4 NE,
// 5 NW, 6 SE, 7 SW; 0, 2, 4 and 5 run towards higher square numbers.
var rayBB [8][64]uint64
var rookMask, bishopMask [64]uint64 // attacks on an empty board
var pawnAttBB [2][64]uint64         // squares a pawn of that color on sq attacks

func init() {
	for sq := 0; sq < 64; sq++ {
		f, r := sq&7, sq>>3
		for d, dd := range queenD {
			for nr, nf := r+dd[0], f+dd[1]; nr >= 0 && nr < 8 && nf >= 0 && nf < 8; nr, nf = nr+dd[0], nf+dd[1] {
				rayBB[d][sq] |= 1 << uint(nr*8+nf)
			}
		}
		rookMask[sq] = rayBB[0][sq] | rayBB[1][sq] | rayBB[2][sq] | rayBB[3][sq]
		bishopMask[sq] = rayBB[4][sq] | rayBB[5][sq] | rayBB[6][sq] | rayBB[7][sq]
		pawnAttBB[White][sq] = pawnAttWBB(1 << uint(sq))
		pawnAttBB[Black][sq] = pawnAttBBB(1 << uint(sq))
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

// rookAtt and bishopAtt are the squares a rook or a bishop on sq attacks
// with the men of occ on the board (the first man on each ray included).
func rookAtt(sq int, occ uint64) uint64 {
	n, s, e, w := rayBB[0][sq], rayBB[1][sq], rayBB[2][sq], rayBB[3][sq]
	if b := n & occ; b != 0 {
		n ^= rayBB[0][bits.TrailingZeros64(b)]
	}
	if b := s & occ; b != 0 {
		s ^= rayBB[1][63-bits.LeadingZeros64(b)]
	}
	if b := e & occ; b != 0 {
		e ^= rayBB[2][bits.TrailingZeros64(b)]
	}
	if b := w & occ; b != 0 {
		w ^= rayBB[3][63-bits.LeadingZeros64(b)]
	}
	return n | s | e | w
}

func bishopAtt(sq int, occ uint64) uint64 {
	ne, nw, se, sw := rayBB[4][sq], rayBB[5][sq], rayBB[6][sq], rayBB[7][sq]
	if b := ne & occ; b != 0 {
		ne ^= rayBB[4][bits.TrailingZeros64(b)]
	}
	if b := nw & occ; b != 0 {
		nw ^= rayBB[5][bits.TrailingZeros64(b)]
	}
	if b := se & occ; b != 0 {
		se ^= rayBB[6][63-bits.LeadingZeros64(b)]
	}
	if b := sw & occ; b != 0 {
		sw ^= rayBB[7][63-bits.LeadingZeros64(b)]
	}
	return ne | nw | se | sw
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

// ---------- evaluation: neural network ----------

// The network sees the board twice, once from each side: 768 inputs (own or
// enemy man, its type, its square as that side sees it, mirrored left to
// right when its own king is on the files e-h) feed nn.nw*4 hidden units
// per side; the output adds up the clipped hidden units of the side to move
// and of the other side with the weights of one of nn.nb buckets (by the
// number of men), plus a value of every man in that bucket (as the side to
// move sees it minus as the other side sees it). Hidden units are 16-bit
// lanes, four to a word, stored with an offset so that adding and
// subtracting whole words adds and subtracts the lanes: no lane ever leaves
// 0..32767.
const (
	nnZero  = 1 << 14 // lane of a hidden unit at zero
	nnWZero = 512     // lane of a weight of zero
	nnOnes  = 0x0001000100010001
)

var nn struct {
	nw, nb, k int
	ft        []uint64         // input weights: 768 rows of nw words
	fb        []uint64         // hidden biases
	ow        []uint64         // output weights + 128: nb rows of 2*nw words, lanes reversed
	ob        []int32          // output biases
	pq        []int32          // values of the men: 768 rows of nb
	feat      [2][2][768]int32 // input of man*64+square: by view, mirrored or not
}

// nnLoad reads a network: "CNN3", the hidden units per view, the buckets,
// the centipawns per unit of output, 0 (16 bits each), then the input
// weights (all low bytes, then all high bytes), the hidden biases (16 bits),
// the output weights (8 bits), the output biases (32 bits) and the values
// of the men (low bytes, then high bytes).
func nnLoad(d []byte) error {
	if len(d) < 12 || string(d[:4]) != "CNN3" {
		return fmt.Errorf("network: bad header")
	}
	h, nb, k := int(rd16(d[4:])), int(rd16(d[6:])), int(rd16(d[8:]))
	if h == 0 || h%4 != 0 || nb == 0 || len(d) != 12+2*768*h+2*h+nb*2*h+4*nb+2*768*nb {
		return fmt.Errorf("network: bad size")
	}
	nw := h / 4
	ft, fb := make([]uint64, 768*nw), make([]uint64, nw)
	ow, ob, pq := make([]uint64, nb*2*nw), make([]int32, nb), make([]int32, 768*nb)
	w1, d := d[12:12+2*768*h], d[12+2*768*h:]
	b1, w2, b2, pv := d[:2*h], d[2*h:2*h+nb*2*h], d[2*h+nb*2*h:2*h+nb*2*h+4*nb], d[2*h+nb*2*h+4*nb:]
	for i := 0; i < 768*h; i++ {
		w := int(int16(uint16(w1[i]) | uint16(w1[768*h+i])<<8))
		if w < -480 || w > 480 {
			return fmt.Errorf("network: weight out of range")
		}
		ft[i/4] |= uint64(w+nnWZero) << uint(16*(i&3))
	}
	for i := 0; i < h; i++ {
		b := int(int16(rd16(b1[2*i:])))
		if b < -1000 || b > 1000 {
			return fmt.Errorf("network: bias out of range")
		}
		fb[i/4] |= uint64(b+nnZero+2) << uint(16*(i&3))
	}
	for i := 0; i < nb*2*h; i++ {
		ow[i/4] |= uint64(int(int8(w2[i]))+128) << uint(16*(3-i&3))
	}
	for i := range ob {
		ob[i] = int32(rd32(b2[4*i:]))
	}
	for i := range pq {
		pq[i] = int32(int16(uint16(pv[i]) | uint16(pv[768*nb+i])<<8))
	}
	nn.nw, nn.nb, nn.k, nn.ft, nn.fb, nn.ow, nn.ob, nn.pq = nw, nb, k, ft, fb, ow, ob, pq
	for pi := 0; pi < 12; pi++ {
		for sq := 0; sq < 64; sq++ {
			black, t := pi/6, pi%6
			for m := 0; m < 2; m++ {
				nn.feat[White][m][pi*64+sq] = int32(black*384 + t*64 + (sq ^ 7*m))
				nn.feat[Black][m][pi*64+sq] = int32((1-black)*384 + t*64 + (sq ^ 56 ^ 7*m))
			}
		}
	}
	return nil
}

var nnOnce sync.Once

// nnInit loads the network: nnData at the end of this file, or the file
// that CHESS_NET names. main does it at the start; a search or an
// evaluation that comes before does it too.
func nnInit() {
	var d []byte
	var err error
	if f := os.Getenv("CHESS_NET"); f != "" {
		d, err = os.ReadFile(f)
	} else if d, err = base64.StdEncoding.DecodeString(nnData); err == nil {
		d, err = io.ReadAll(flate.NewReader(bytes.NewReader(d)))
	}
	if err == nil {
		err = nnLoad(d)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "chess:", err)
		os.Exit(1)
	}
}

// nnStack holds, for the positions on the search path, the hidden units
// before clipping (v) and the sums of the values of the men (ps), each for
// both views. An entry is computed when an evaluation needs it, from the
// entry before it and the men its move put down and took off; a view whose
// king has crossed the middle of the board is computed from the board at
// once.
type nnStack struct {
	at, over int
	ok       [MaxPly + 2]bool
	mir      [MaxPly + 2][2]int8  // views mirrored
	na, ns   [MaxPly + 2]int8     // men put down, men taken off
	add, sub [MaxPly + 2][2]int16 // man*64+square
	v        []uint64             // entries of 2*nw words; the last one is scratch
	ps       []int32              // entries of 2*nb sums
}

func nnMirrored(p *Pos) [2]int8 {
	return [2]int8{int8(p.kingSq[White] >> 2 & 1), int8(p.kingSq[Black] >> 2 & 1)}
}

func (p *Pos) nnAttach(s *nnStack) {
	nnOnce.Do(nnInit)
	if len(s.v) != (MaxPly+2)*2*nn.nw {
		s.v = make([]uint64, (MaxPly+2)*2*nn.nw)
		s.ps = make([]int32, (MaxPly+2)*2*nn.nb)
	}
	s.at, s.over = 0, 0
	p.nn = s
	s.mir[0] = nnMirrored(p)
	s.refresh(p, 0, White)
	s.refresh(p, 0, Black)
	s.ok[0] = true
}

func (s *nnStack) push(p *Pos, pc, np, from, to, captured, capSq int, castle bool) {
	if s.at >= MaxPly {
		s.over++
		return
	}
	s.at++
	i := s.at
	s.ok[i] = false
	s.sub[i][0] = int16(pieceIdx(pc)*64 + from)
	s.add[i][0] = int16(pieceIdx(np)*64 + to)
	s.na[i], s.ns[i] = 1, 1
	if captured != Empty {
		s.sub[i][1] = int16(pieceIdx(captured)*64 + capSq)
		s.ns[i] = 2
	} else if castle {
		rf, rt := castleRook(to)
		rook := pieceIdx(pc) - 2
		s.sub[i][1] = int16(rook*64 + rf)
		s.add[i][1] = int16(rook*64 + rt)
		s.na[i], s.ns[i] = 2, 2
	}
	s.mir[i] = nnMirrored(p)
	if s.mir[i] != s.mir[i-1] {
		s.upTo(i - 1)
		for c := 0; c < 2; c++ {
			if s.mir[i][c] != s.mir[i-1][c] {
				s.refresh(p, i, c)
			} else {
				s.step(i, c)
			}
		}
		s.ok[i] = true
	}
}

// refresh computes view c of entry j from the board.
func (s *nnStack) refresh(p *Pos, j, c int) {
	nw, nb := nn.nw, nn.nb
	feat := &nn.feat[c][p.kingSq[c]>>2&1]
	dst := s.v[(2*j+c)*nw : (2*j+c+1)*nw]
	pd := s.ps[(2*j+c)*nb : (2*j+c+1)*nb]
	copy(dst, nn.fb)
	clear(pd)
	for b := p.occ[White] | p.occ[Black]; b != 0; b &= b - 1 {
		sq := bits.TrailingZeros64(b)
		f := int(feat[pieceIdx(p.b[sq])*64+sq])
		row := nn.ft[f*nw:][:len(dst)]
		for i := range dst {
			dst[i] += row[i] - nnWZero*nnOnes
		}
		val := nn.pq[f*nb:][:len(pd)]
		for i := range pd {
			pd[i] += val[i]
		}
	}
}

// step computes view c of entry j from entry j-1.
func (s *nnStack) step(j, c int) {
	nw, nb := nn.nw, nn.nb
	feat := &nn.feat[c][s.mir[j][c]]
	dst := s.v[(2*j+c)*nw : (2*j+c+1)*nw]
	src := s.v[(2*j-2+c)*nw:][:len(dst)]
	pd := s.ps[(2*j+c)*nb : (2*j+c+1)*nb]
	pp := s.ps[(2*j-2+c)*nb:][:len(pd)]
	fa, fs := int(feat[s.add[j][0]]), int(feat[s.sub[j][0]])
	a0, s0 := nn.ft[fa*nw:][:len(dst)], nn.ft[fs*nw:][:len(dst)]
	pa, pz := nn.pq[fa*nb:][:len(pd)], nn.pq[fs*nb:][:len(pd)]
	switch {
	case s.ns[j] == 1:
		for i := range dst {
			dst[i] = src[i] + a0[i] - s0[i]
		}
		for i := range pd {
			pd[i] = pp[i] + pa[i] - pz[i]
		}
	case s.na[j] == 1:
		f1 := int(feat[s.sub[j][1]])
		s1, p1 := nn.ft[f1*nw:][:len(dst)], nn.pq[f1*nb:][:len(pd)]
		for i := range dst {
			dst[i] = src[i] + a0[i] + nnWZero*nnOnes - s0[i] - s1[i]
		}
		for i := range pd {
			pd[i] = pp[i] + pa[i] - pz[i] - p1[i]
		}
	default:
		f1, f2 := int(feat[s.add[j][1]]), int(feat[s.sub[j][1]])
		a1, s1 := nn.ft[f1*nw:][:len(dst)], nn.ft[f2*nw:][:len(dst)]
		p1, p2 := nn.pq[f1*nb:][:len(pd)], nn.pq[f2*nb:][:len(pd)]
		for i := range dst {
			dst[i] = src[i] + a0[i] + a1[i] - s0[i] - s1[i]
		}
		for i := range pd {
			pd[i] = pp[i] + pa[i] + p1[i] - pz[i] - p2[i]
		}
	}
}

// upTo computes the entries up to entry n that are still missing.
func (s *nnStack) upTo(n int) {
	j := n
	for !s.ok[j] {
		j--
	}
	for j++; j <= n; j++ {
		s.step(j, White)
		s.step(j, Black)
		s.ok[j] = true
	}
}

// cur returns the entry of the current position.
func (s *nnStack) cur(p *Pos) int {
	if s.over > 0 {
		s.refresh(p, MaxPly+1, White)
		s.refresh(p, MaxPly+1, Black)
		return MaxPly + 1
	}
	s.upTo(s.at)
	return s.at
}

// nnAct clips the four hidden units of a word to 0..255 and keeps their
// upper six bits.
func nnAct(x uint64) uint64 {
	pos := x >> 14 & nnOnes
	big := (x&0x3F003F003F003F00 + 0x3F003F003F003F00) >> 14 & nnOnes
	return (x>>2&0x003F003F003F003F | big*63) & (pos * 63)
}

// nnEval is the output of the network for the side to move, in centipawns.
// A word of four hidden units times the word of their weights in reverse
// order has their dot product in its upper 16 bits.
func (p *Pos) nnEval() int {
	s, j := p.nn, 0
	if s == nil {
		nnOnce.Do(nnInit)
	}
	nw, nb := nn.nw, nn.nb
	if s != nil {
		j = s.cur(p)
	} else {
		s = &nnStack{v: make([]uint64, 2*nw), ps: make([]int32, 2*nb)}
		s.refresh(p, 0, White)
		s.refresh(p, 0, Black)
	}
	us, them := s.v[(2*j+p.side)*nw:][:nw], s.v[(2*j+(p.side^1))*nw:][:nw]
	bkt := min((bits.OnesCount64(p.occ[White]|p.occ[Black])-2)/4, nb-1)
	w := nn.ow[bkt*2*nw:][:2*nw]
	var dot, sum uint64
	for i, x := range us {
		a, b := nnAct(x), nnAct(them[i])
		dot += a*w[i]>>48 + b*w[nw+i]>>48
		sum += a + b
	}
	sum = sum&0xFFFF + sum>>16&0xFFFF + sum>>32&0xFFFF + sum>>48
	v := int(dot) - 128*int(sum) + int(nn.ob[bkt]) +
		int(s.ps[(2*j+p.side)*nb+bkt]) - int(s.ps[(2*j+(p.side^1))*nb+bkt])
	return v * nn.k / 4096
}

// ---------- evaluation: what is known without the network ----------

// mopUp is the bonus for driving a bare king (on lone) to the edge with the
// own king close by.
func mopUp(lone, king int) int {
	f, r := lone&7, lone>>3
	d := abs(f-king&7) + abs(r-king>>3)
	return 47*(max(3-f, f-4)+max(3-r, r-4)) + 16*(14-d)
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// kbnk is the bonus for driving the bare king (on lone) into a corner of
// the bishop's color with the own king close by: bishop and knight can
// mate nowhere else.
func kbnk(bishop uint64, lone, king int) int {
	f, r := lone&7, lone>>3
	d := min(f+r, 14-f-r) // steps to a1 or h8, the dark corners
	if bishop&lightSquares != 0 {
		d = min(7-f+r, f+7-r) // to h1 or a8
	}
	return 40*(7-d) + 10*(7-chebDist(lone, king))
}

// kpkWins has a bit for every position of a white king, a white pawn on the
// files a-d and a black king, with either side to move, that White wins:
// index side to move | white king<<1 | black king<<7 | file<<13 | (6-rank)<<15.
// It is worked out once: when the program starts, or else at its first use.
var kpkWins [2 * 64 * 64 * 24 / 64]uint64
var kpkOnce sync.Once

func kpkIndex(us, wk, bk, psq int) int {
	return us | wk<<1 | bk<<7 | (psq&7)<<13 | (6-psq>>3)<<15
}

func kpkInit() {
	const unknown, draw, win = 1, 2, 4 // 0: not a position
	res := make([]uint8, 2*64*64*24)
	sq := func(i int) (us, wk, bk, psq int) {
		return i & 1, i >> 1 & 63, i >> 7 & 63, i>>13&3 + 8*(6-i>>15)
	}
	for i := range res {
		us, wk, bk, psq := sq(i)
		stop := psq + 8
		switch {
		case chebDist(wk, bk) <= 1 || wk == psq || bk == psq || us == White && pawnAttBB[White][psq]>>uint(bk)&1 != 0:
		case us == White && psq>>3 == 6 && wk != stop && (chebDist(bk, stop) > 1 || chebDist(wk, stop) == 1):
			res[i] = win // promotes
		case us == Black && (kingBB[bk]&^(kingBB[wk]|pawnAttBB[White][psq]) == 0 || kingBB[bk]&^kingBB[wk]>>uint(psq)&1 != 0):
			res[i] = draw // stalemate, or the pawn falls
		default:
			res[i] = unknown
		}
	}
	// White wins when one of his moves wins, Black draws when one of his draws
	for changed := true; changed; {
		changed = false
		for i := range res {
			if res[i] != unknown {
				continue
			}
			us, wk, bk, psq := sq(i)
			var r uint8
			if us == White {
				for t := kingBB[wk]; t != 0; t &= t - 1 {
					r |= res[kpkIndex(Black, bits.TrailingZeros64(t), bk, psq)]
				}
				if psq>>3 < 6 {
					r |= res[kpkIndex(Black, wk, bk, psq+8)]
					if psq>>3 == 1 && psq+8 != wk && psq+8 != bk {
						r |= res[kpkIndex(Black, wk, bk, psq+16)]
					}
				}
				if r&win != 0 {
					res[i], changed = win, true
				} else if r&unknown == 0 {
					res[i], changed = draw, true
				}
			} else {
				for t := kingBB[bk]; t != 0; t &= t - 1 {
					r |= res[kpkIndex(White, wk, bits.TrailingZeros64(t), psq)]
				}
				if r&draw != 0 {
					res[i], changed = draw, true
				} else if r&unknown == 0 {
					res[i], changed = win, true
				}
			}
		}
	}
	for i, r := range res {
		if r == win {
			kpkWins[i/64] |= 1 << uint(i&63)
		}
	}
}

// kpkScore scores king and pawn against king for side c, which has the
// pawn: nothing when the pawn cannot win, else more the further it has come.
func (p *Pos) kpkScore(c int) int {
	psq, wk, bk := bits.TrailingZeros64(p.bb[c][WP]), p.kingSq[c], p.kingSq[c^1]
	if c == Black {
		psq, wk, bk = psq^56, wk^56, bk^56
	}
	if psq&7 > 3 {
		psq, wk, bk = psq^7, wk^7, bk^7
	}
	kpkOnce.Do(kpkInit)
	i := kpkIndex(p.side^c, wk, bk, psq)
	if kpkWins[i/64]>>uint(i&63)&1 == 0 {
		return 0
	}
	return nnWon + 50*(psq>>3) + 5*chebDist(bk, psq) - 5*chebDist(wk, psq)
}

// rookPawnDraw reports the known draw against the bare king of side c:
// pawns on one rook file, no piece that controls their corner, and the
// king gets there in time.
func (p *Pos) rookPawnDraw(c int) bool {
	s := &p.bb[c^1]
	corner := 0
	switch {
	case s[WQ]|s[WR]|s[WN] != 0:
		return false
	case s[WP]&^fileABB == 0:
	case s[WP]&^fileHBB == 0:
		corner = 7
	default:
		return false
	}
	togo := bits.TrailingZeros64(s[WP]) >> 3 // moves of the first pawn to the corner
	if c == Black {
		corner, togo = corner+56, bits.LeadingZeros64(s[WP])>>3
	}
	if light := lightSquares>>uint(corner)&1 != 0; light && s[WB]&lightSquares != 0 || !light && s[WB]&^lightSquares != 0 {
		return false
	}
	d := chebDist(p.kingSq[c], corner)
	return d <= 1 || d-b2i(c == p.side) < togo
}

// material is the value of the men of side c but its king.
func (p *Pos) material(c int) int {
	b := &p.bb[c]
	return 100*bits.OnesCount64(b[WP]) + seeVal[WN]*bits.OnesCount64(b[WN]) + seeVal[WB]*bits.OnesCount64(b[WB]) +
		seeVal[WR]*bits.OnesCount64(b[WR]) + seeVal[WQ]*bits.OnesCount64(b[WQ])
}

// nnWon is the score from which a position counts as won. The network
// tells such positions apart poorly, so they are scored by what brings the
// win closer: wonScore.
const nnWon = 1000

// wonScore scores a position that side c wins: the material ahead, the
// pawns advanced and, against a bare king, the way to the mate.
func (p *Pos) wonScore(c int) int {
	s := &p.bb[c]
	w := nnWon + max(p.material(c)-p.material(c^1), 0)
	bare := p.occ[c^1] == p.bb[c^1][WK]
	step := 10
	if bare {
		step = 40 // a pawn's step must count for more than a king's step of the mop-up
	}
	for b := s[WP]; b != 0; b &= b - 1 {
		if r := bits.TrailingZeros64(b) >> 3; c == White {
			w += step * r
		} else {
			w += step * (7 - r)
		}
	}
	if bare {
		if s[WQ]|s[WR] == 0 && s[WB] != 0 && (s[WB]&lightSquares == 0 || s[WB]&^lightSquares == 0) {
			w += kbnk(s[WB], p.kingSq[c^1], p.kingSq[c]) // bishops of one color: their corner
		} else {
			w += mopUp(p.kingSq[c^1], p.kingSq[c])
		}
	}
	return w
}

// eval returns the score from the side-to-move's perspective (centipawns):
// the network's, but for the won positions and the endings that are known
// exactly.
func (p *Pos) eval() int {
	for c := 0; c < 2; c++ {
		if p.occ[c] != p.bb[c][WK] {
			continue
		}
		// the bare king of side c: what wins and how is known, no network
		s := &p.bb[c^1]
		knights := bits.OnesCount64(s[WN])
		v := 10 * knights // one or two knights cannot force a mate
		switch {
		case s[WP] != 0 && s[WP]&(s[WP]-1) == 0 && p.occ[c^1] == s[WP]|s[WK]:
			v = p.kpkScore(c ^ 1)
		case s[WP] != 0 && p.rookPawnDraw(c):
			v = 20
		case s[WP]|s[WQ]|s[WR] != 0 || s[WB]&lightSquares != 0 && s[WB]&^lightSquares != 0 || s[WB] != 0 && knights > 0 || knights > 2:
			v = p.wonScore(c ^ 1)
		}
		if c == p.side {
			return -v
		}
		return v
	}
	v := max(-20000, min(20000, p.nnEval()))
	if v >= nnWon {
		return p.wonScore(p.side) + min(v-nnWon, 1600)/8
	}
	if v <= -nnWon {
		return -p.wonScore(p.side^1) - min(-v-nnWon, 1600)/8
	}
	return v
}

// non-pawn material present for the side to move (null-move guard)
func (p *Pos) hasNonPawnMaterial() bool {
	bb := &p.bb[p.side]
	return bb[WN]|bb[WB]|bb[WR]|bb[WQ] != 0
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

// uints parses numbers separated by white space (a final dot is ignored);
// nil when one of them is not a number.
func uints(s string) (v []uint64) {
	for _, f := range strings.Fields(s) {
		n, err := strconv.ParseUint(strings.TrimSuffix(f, "."), 10, 64)
		if err != nil {
			return nil
		}
		v = append(v, n)
	}
	return v
}

// freeBSDAvail counts the free pages, and the inactive ones only when there
// is swap to move them to: without swap they cannot be reclaimed.
func freeBSDAvail(pageSize, free, inactive, swap uint64) uint64 {
	if swap > 0 {
		free += inactive
	}
	return free * pageSize
}

// vmStatAvail reads the output of the macOS vm_stat command: the free,
// inactive, speculative and purgeable pages.
func vmStatAvail(out string) uint64 {
	var pageSize, pages uint64
	for _, ln := range strings.Split(out, "\n") {
		if _, rest, ok := strings.Cut(ln, "page size of "); ok {
			if v := uints(strings.TrimSuffix(rest, " bytes)")); len(v) == 1 {
				pageSize = v[0]
			}
		}
		switch name, val, _ := strings.Cut(ln, ":"); name {
		case "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable":
			if v := uints(val); len(v) == 1 {
				pages += v[0]
			}
		}
	}
	return pageSize * pages
}

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
	// FreeBSD
	if out, err := exec.Command("sysctl", "-n", "hw.pagesize", "vm.stats.vm.v_free_count",
		"vm.stats.vm.v_inactive_count", "vm.swap_total").Output(); err == nil {
		if v := uints(string(out)); len(v) == 4 {
			return freeBSDAvail(v[0], v[1], v[2], v[3])
		}
	}
	// macOS
	if out, err := exec.Command("vm_stat").Output(); err == nil {
		if n := vmStatAvail(string(out)); n > 0 {
			return n
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

// searchThreads is the number of search workers: all CPUs, or CHESS_THREADS.
func searchThreads() int {
	n := runtime.NumCPU()
	if v, err := strconv.Atoi(os.Getenv("CHESS_THREADS")); err == nil && v >= 1 && v <= n {
		n = v
	}
	return n
}

// hashBytes is the memory to give the hash table: 80% of the available RAM
// (floor 64 MB, cap 64 GB) and, with a time limit per move, no more than
// the search can fill within it: about 48 MB per thread and second.
func hashBytes(avail uint64, limit time.Duration, threads int) uint64 {
	budget := avail * 4 / 5
	if fill := limit.Seconds() * float64(threads) * (48 << 20); limit > 0 && fill < float64(budget) {
		budget = uint64(fill)
	}
	return min(max(budget, 64<<20), 64<<30)
}

// initTT sizes the hash table from the RAM the machine can actually spare
// (hashBytes, CHESS_HASH_MB env overrides) and sets a Go runtime memory
// limit (table + GC headroom) so the process can never OOM in any mode: all
// other engine data is fixed-size, so total memory stays flat at this
// ceiling no matter how long it thinks. In RAM-limited (0) mode this table
// is also the search budget itself.
func initTT(limit time.Duration) {
	var budget uint64
	if s := os.Getenv("CHESS_HASH_MB"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			budget = uint64(v) << 20
		}
	}
	if budget == 0 {
		budget = hashBytes(availableRAM(), limit, searchThreads())
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
			ttSize*ttEntryBytes>>20, (ttSize*ttEntryBytes+head)>>20, searchThreads())
	}
}

func encMove(m Move) uint64 { return uint64(m) }

func decMove(v uint64) Move { return Move(v & 0xFFFFF) }

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

// ttStore counts its stores in the engine: a counter shared by all threads
// would be written from every core on every node.
func (e *Engine) ttStore(hash uint64, m Move, score, depth, flag, ply, se int) {
	if ttStore(hash, m, score, depth, flag, ply, se) {
		e.stores++
	}
}

// ttStore reports whether it wrote an entry (false: it only refreshed the
// age of a deeper one).
func ttStore(hash uint64, m Move, score, depth, flag, ply, se int) bool {
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
				return false
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
	return true
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
	deadline    time.Time                  // hard stop, enforced inside the search
	memLimited  bool                       // 0-mode: stop when the hash capacity is used up
	nodes       atomic.Uint64              // ~nodes this move (flushed every 1024/worker)
	nextMark    atomic.Uint64              // verbose: next 0-mode progress checkpoint
	atDepth     [MaxDepth + 1]atomic.Int32 // workers searching each depth
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
	stores    uint64 // hash stores not yet added to ttStores
	seldepth  int    // deepest ply touched (incl. quiescence)
	killers   [MaxPly][2]Move
	excluded  [MaxPly]Move
	corrHist  [2][16384]int32
	history   [2][64][64]int32 // butterfly quiet history (gravity-bounded)
	counter   [2][64][64]Move  // countermove heuristic, indexed by prev move
	capHist   [12][64][7]int32 // capture history: [attacker][to][victimType]
	contHist  [2][768][768]int16
	moveStack [MaxPly]Move      // move that led to each ply (for countermoves)
	evalStack [MaxPly]int       // static eval per ply (evalNone in check)
	pcStack   [MaxPly]int       // pieceIdx*64+to of the move at each ply, -1 none
	mbuf      [MaxPly][256]Move // move list of each ply
	sbuf      [256]int          // move ordering scores
	qbuf      [256]Move         // move ordering scratch
	nn        nnStack
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
	pc := pieceIdx(e.p.b[m.From()])*64 + m.To()
	h := int(e.history[side][m.From()][m.To()])
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
		histUp32(&e.history[side][m.From()][m.To()], b)
		pc := pieceIdx(e.p.b[m.From()])*64 + m.To()
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
	if m.Flags()&fEP == 0 {
		v = victimType(p.b[m.To()])
	}
	return &e.capHist[pieceIdx(p.b[m.From()])][m.To()][v]
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
		st := ttStores.Add(e.stores)
		e.stores = 0
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

// isRepetition tells whether the current position, ply moves into the
// search and depth more to go, is a draw by repetition: it has occurred
// before in the search, or twice in the game. Once in the game is not
// enough, because the opponent, who chose a move there, may choose a
// better one now - unless the search ends here and will not see it.
func (e *Engine) isRepetition(ply, depth int) bool {
	p := e.p
	n := len(p.hist)
	// the positions with the same side to move since the last irreversible move
	seen := depth <= 0
	for i, first := n-3, max(n-1-p.rev, 0); i >= first; i -= 2 {
		if p.hist[i] == p.hash {
			if n-1-i < ply || seen {
				return true
			}
			seen = true
		}
	}
	return false
}

func mvvLva(p *Pos, m Move) int {
	victim := p.b[m.To()]
	if m.Flags()&fEP != 0 {
		victim = WP
	}
	if victim < 0 {
		victim = -victim
	}
	attacker := p.b[m.From()]
	if attacker < 0 {
		attacker = -attacker
	}
	s := pieceVal[victim]*16 - pieceVal[attacker]/16
	if m.Promo() != 0 {
		s += pieceVal[m.Promo()]
	}
	return s
}

// The order in which SEE looks along the rays from the target for a bishop
// or queen (NE, SE, NW, SW) and for a rook or queen (E, W, N, S).
var seeDiag, seeLine = [4]int{4, 6, 5, 7}, [4]int{2, 3, 0, 1}

// rayFirst returns the first square of occ in direction d from sq, or -1.
func rayFirst(d, sq int, occ uint64) int {
	b := rayBB[d][sq] & occ
	if b == 0 {
		return -1
	}
	if rayUp[d] {
		return bits.TrailingZeros64(b)
	}
	return 63 - bits.LeadingZeros64(b)
}

// leastAttacker returns the square of the least valuable man of side that
// attacks target with only the men of occ on the board; -1 if none. Among
// equal attackers the choice is fixed: it decides which man behind them
// joins the exchange next.
func (p *Pos) leastAttacker(target, side int, occ uint64) int {
	bb := &p.bb[side]
	if b := pawnAttBB[side^1][target] & bb[WP] & occ; b != 0 {
		return bits.TrailingZeros64(b)
	}
	if b := knightBB[target] & bb[WN] & occ; b != 0 {
		return bits.TrailingZeros64(b)
	}
	if b := bb[WB] & occ; b&bishopMask[target] != 0 {
		for _, d := range seeDiag {
			if sq := rayFirst(d, target, occ); sq >= 0 && b>>uint(sq)&1 != 0 {
				return sq
			}
		}
	}
	if b := bb[WR] & occ; b&rookMask[target] != 0 {
		for _, d := range seeLine {
			if sq := rayFirst(d, target, occ); sq >= 0 && b>>uint(sq)&1 != 0 {
				return sq
			}
		}
	}
	if b := bb[WQ] & occ; b&(bishopMask[target]|rookMask[target]) != 0 {
		for _, d := range seeDiag {
			if sq := rayFirst(d, target, occ); sq >= 0 && b>>uint(sq)&1 != 0 {
				return sq
			}
		}
		for _, d := range seeLine {
			if sq := rayFirst(d, target, occ); sq >= 0 && b>>uint(sq)&1 != 0 {
				return sq
			}
		}
	}
	if b := kingBB[target] & bb[WK] & occ; b != 0 {
		return bits.TrailingZeros64(b)
	}
	return -1
}

// seeBelow reports whether see(m) < bound, for a bound <= 0. Capturing a
// man worth at least the capturing one never loses material, so only the
// other exchanges are played out.
func (p *Pos) seeBelow(m Move, bound int) bool {
	if m.Flags()&fCapture != 0 && m.Promo() == 0 {
		victim, att := p.b[m.To()], p.b[m.From()]
		if victim < 0 {
			victim = -victim
		}
		if att < 0 {
			att = -att
		}
		if m.Flags()&fEP != 0 {
			victim = WP
		}
		if seeVal[victim] >= seeVal[att] {
			return false
		}
	}
	return p.see(m) < bound
}

// see estimates the material outcome of the capture sequence on m.To()
// (static exchange evaluation; pins are ignored - it's a pruning heuristic).
func (p *Pos) see(m Move) int {
	target := m.To()
	occ := p.occ[White] | p.occ[Black]
	victim := p.b[target]
	if m.Flags()&fEP != 0 {
		victim = WP
		if p.side == White {
			occ &^= 1 << uint(target-8)
		} else {
			occ &^= 1 << uint(target+8)
		}
	}
	if victim < 0 {
		victim = -victim
	}
	var gain [40]int
	d := 0
	gain[0] = seeVal[victim]
	att := p.b[m.From()]
	if att < 0 {
		att = -att
	}
	if m.Promo() != 0 {
		gain[0] += seeVal[m.Promo()] - seeVal[WP]
		att = m.Promo()
	}
	occ &^= 1 << uint(m.From())
	side := p.side ^ 1
	for d < 38 {
		sq := p.leastAttacker(target, side, occ)
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
		occ &^= 1 << uint(sq)
		side ^= 1
	}
	for ; d > 0; d-- {
		if -gain[d] < gain[d-1] {
			gain[d-1] = -gain[d]
		}
	}
	return gain[0]
}

// hashFirst moves the hash move to the front of moves, the others keeping
// their order, and reports whether it is there. The search tries it before
// the other moves are scored and sorted: it mostly cuts off.
func hashFirst(moves []Move, ttMove Move) bool {
	if ttMove != NoMove {
		for i, m := range moves {
			if m == ttMove {
				copy(moves[1:i+1], moves[:i])
				moves[0] = m
				return true
			}
		}
	}
	return false
}

// orderMoves sorts a whole move list: the hash move, the good captures and
// promotions, the killers, the countermove, the other quiet moves by their
// history and last the losing captures.
func (e *Engine) orderMoves(moves []Move, ttMove Move, ply int) {
	if hashFirst(moves, ttMove) {
		moves = moves[1:]
	}
	qs, qe := e.orderCaptures(moves)
	e.orderQuiets(moves[qs:qe], ply)
}

// sortMoves sorts moves by score, best first, equals keeping their order
// (insertion sort: move lists are short).
func sortMoves(moves []Move, scores []int) {
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

func (e *Engine) scores(n int) []int {
	if n > len(e.sbuf) {
		return make([]int, n)
	}
	return e.sbuf[:n]
}

// orderCaptures rearranges moves into the good captures and promotions
// (best first), the quiet moves (as generated) and the losing captures
// (best first); the quiet moves are moves[qs:qe].
func (e *Engine) orderCaptures(moves []Move) (qs, qe int) {
	p := e.p
	quiets, n := e.qbuf[:0], 0
	for _, m := range moves {
		if m.Flags()&fCapture != 0 || m.Promo() != 0 {
			moves[n] = m
			n++
		} else {
			quiets = append(quiets, m)
		}
	}
	scores := e.scores(n)
	for i, m := range moves[:n] {
		if m.Flags()&fCapture == 0 {
			scores[i] = (1 << 28) + pieceVal[m.Promo()]
		} else if s := mvvLva(p, m) + int(*e.capHistOf(m))/16; m.Promo() != 0 || !p.seeBelow(m, 0) {
			scores[i] = (1 << 28) + s
		} else {
			scores[i] = -(1 << 27) + s
		}
	}
	sortMoves(moves[:n], scores)
	for qs = n; qs > 0 && scores[qs-1] < 0; qs-- {
	}
	qe = qs + len(quiets)
	copy(moves[qe:], moves[qs:n])
	copy(moves[qs:], quiets)
	return qs, qe
}

// orderQuiets sorts quiet moves: killers, the countermove, then by history.
func (e *Engine) orderQuiets(moves []Move, ply int) {
	p := e.p
	counter := NoMove
	if ply >= 1 && e.moveStack[ply-1] != NoMove {
		prev := e.moveStack[ply-1]
		counter = e.counter[p.side][prev.From()][prev.To()]
	}
	scores := e.scores(len(moves))
	for i, m := range moves {
		switch {
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
	sortMoves(moves, scores)
}

// stalemated tells whether the side to move, left with a king and pawns
// and not in check, has no move at all. The quiescence search, which tries
// captures only, would take such a position for a won one. A stalemate
// with a pinned pawn is not seen, none is reported wrongly.
func (p *Pos) stalemated() bool {
	c := p.side
	pawns := p.bb[c][WP]
	if p.occ[c] != p.bb[c][WK]|pawns {
		return false
	}
	if pawns != 0 {
		push, att := pawns<<8, pawnAttWBB(pawns)
		if c == Black {
			push, att = pawns>>8, pawnAttBBB(pawns)
		}
		if push&^(p.occ[White]|p.occ[Black]) != 0 || att&p.occ[c^1] != 0 || p.ep >= 0 {
			return false
		}
	}
	for t := kingBB[p.kingSq[c]] &^ p.occ[c]; t != 0; t &= t - 1 {
		if !p.attacked(bits.TrailingZeros64(t), c^1) {
			return false
		}
	}
	return true
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
	if insufficientMaterial(p) {
		return 0
	}
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
	if !inChk && p.stalemated() {
		return 0
	}
	stand := -Inf
	if !inChk { // no stand-pat while in check: must search all evasions
		if ttHit && ttEval != evalNone {
			stand = ttEval
		} else {
			stand = p.eval()
		}
		if stand >= beta {
			if !ttHit {
				e.ttStore(p.hash, NoMove, stand, 0, ttBeta, ply, stand)
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
	moves := p.gen(e.mbuf[ply][:0], !inChk) // in check: all evasions; else captures/promos
	lazy := hashFirst(moves, ttMove)
	if !lazy {
		e.orderMoves(moves, NoMove, ply)
	}
	origAlpha := alpha
	best := stand
	bestM := NoMove
	legal := 0
	se := stand
	if inChk {
		se = evalNone
	}
	for i := 0; i < len(moves); i++ {
		if i == 1 && lazy {
			e.orderMoves(moves[1:], NoMove, ply)
		}
		m := moves[i]
		if !inChk && m.Promo() == 0 {
			victim := p.b[m.To()]
			if m.Flags()&fEP != 0 {
				victim = WP
			}
			if victim < 0 {
				victim = -victim
			}
			if stand+pieceVal[victim]+200 <= alpha {
				continue // delta pruning
			}
			if p.seeBelow(m, 0) {
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
		e.pcStack[ply] = pieceIdx(p.b[m.To()])*64 + m.To()
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
			e.ttStore(p.hash, bestM, best, 0, ttBeta, ply, se)
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
	e.ttStore(p.hash, bestM, best, 0, flag, ply, se)
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
		if p.half >= 100 || e.isRepetition(ply, depth) || insufficientMaterial(p) {
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
			p.rev++
			e.moveStack[ply] = NoMove
			e.pcStack[ply] = -1
			score := -e.search(depth-1-R, ply+1, -beta, -beta+1, false)
			p.rev--
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
			caps := p.gen(e.mbuf[ply][:0], true)
			e.orderMoves(caps, NoMove, ply)
			tried := 0
			for _, m := range caps {
				if tried >= 3 {
					break
				}
				if m.Flags()&fCapture != 0 && p.seeBelow(m, 0) {
					continue
				}
				u := p.make(m)
				if p.attacked(p.kingSq[p.side^1], p.side) {
					p.unmake(m, u)
					continue
				}
				tried++
				e.moveStack[ply] = m
				e.pcStack[ply] = pieceIdx(p.b[m.To()])*64 + m.To()
				v := -e.quiesce(-pcBeta, -pcBeta+1, ply+1)
				if v >= pcBeta {
					v = -e.search(depth-4, ply+1, -pcBeta, -pcBeta+1, true)
				}
				p.unmake(m, u)
				if e.stop {
					return 0
				}
				if v >= pcBeta {
					e.ttStore(p.hash, m, v, depth-3, ttBeta, ply, rawEval)
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
	// The moves are put in order as the search gets to them: the hash move
	// first, then the captures and, only if some will be tried, the quiet
	// moves moves[qs:qe]. Mostly an early move cuts the search off.
	moves := p.gen(e.mbuf[ply][:0], false)
	first := b2i(hashFirst(moves, ttMove))
	qs, qe := len(moves), len(moves)
	origAlpha := alpha
	legal, quiets := 0, 0
	bestScore := -Inf
	bestM := NoMove
	var triedQuiets [32]Move
	var triedCaps [32]Move
	nTQ, nTC := 0, 0
	onlyChecks := false // the quiet moves left are pruned, but for the checks
	for i := 0; i < len(moves); i++ {
		if i == first {
			qs, qe = e.orderCaptures(moves[first:])
			qs, qe = qs+first, qe+first
		}
		if i == qs && qs < qe {
			if !pvNode && !inChk && legal > 0 && bestScore > -(MateScore-2000) && depth <= 8 &&
				staticEval+90+100*depth+futImp*b2i(improving) <= alpha {
				moves, qe = p.checksOnly(moves, i, qe) // futility pruning
				if onlyChecks = true; i == len(moves) {
					break
				}
			} else {
				e.orderQuiets(moves[qs:qe], ply)
			}
		}
		m := moves[i]
		if m == excl {
			continue
		}
		isQuiet := m.Flags()&fCapture == 0 && m.Promo() == 0
		// prune late/hopeless moves (only after one legal move is confirmed)
		if !pvNode && !inChk && legal > 0 && bestScore > -(MateScore-2000) {
			if isQuiet {
				lmpLimit := lmpBase + depth*depth
				if lmpHalve && !improving {
					lmpLimit /= 2
				}
				// late move pruning and futility pruning, of all the quiet moves left
				if !onlyChecks && (depth <= lmpDepth && quiets > lmpLimit ||
					depth <= 8 && staticEval+90+100*depth+futImp*b2i(improving) <= alpha) {
					moves, qe = p.checksOnly(moves, i, qe)
					onlyChecks = true
					i--
					continue
				}
				if depth <= 8 && p.see(m) < -25*depth*depth {
					continue // SEE pruning: quiet loses material badly
				}
			} else if depth <= 6 && m.Promo() == 0 && p.seeBelow(m, -100*depth) {
				continue // SEE pruning: clearly losing capture
			}
		}
		mHist := 0
		if isQuiet && depth >= 3 && !inChk {
			mHist = e.quietHist(p.side, m, ply) // for the reduction; must read before make()
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
		} else if m.Flags()&fCapture != 0 && nTC < 32 {
			triedCaps[nTC] = m
			nTC++
		}
		givesCheck := p.inCheck(p.side)
		e.moveStack[ply] = m
		e.pcStack[ply] = pieceIdx(p.b[m.To()])*64 + m.To()
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
					e.counter[p.side][prev.From()][prev.To()] = m
				}
			} else if m.Flags()&fCapture != 0 {
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
		e.ttStore(p.hash, bestM, bestScore, depth, flag, ply, rawEval)
	}
	return bestScore
}

// checksOnly drops the quiet moves moves[i:qe] but those whose man gives
// check from its new square: the search prunes late and hopeless quiet
// moves, never such a check. It returns the shorter list and the new qe.
func (p *Pos) checksOnly(moves []Move, i, qe int) ([]Move, int) {
	k := p.kingSq[p.side^1]
	occ := p.occ[White] | p.occ[Black]
	var from [7]uint64 // where each kind of man checks from
	from[WP], from[WN] = pawnAttBB[p.side^1][k], knightBB[k]
	from[WB], from[WR] = bishopAtt(k, occ), rookAtt(k, occ)
	from[WQ] = from[WB] | from[WR]
	n := i
	for _, m := range moves[i:qe] {
		if from[abs(p.b[m.From()])]>>uint(m.To())&1 != 0 {
			moves[n] = m
			n++
		}
	}
	return moves[:n+copy(moves[n:], moves[qe:])], n
}

// clone makes an independent copy for a parallel search worker.
func (p *Pos) clone() *Pos {
	q := *p
	q.hist = append([]uint64(nil), p.hist...)
	q.nn = nil
	return &q
}

// rootSearch scans the root move list with the given window (PVS).
func (e *Engine) rootSearch(moves []Move, depth, alpha, beta int) (int, Move) {
	bestScore := -Inf
	bestM := NoMove
	for i, m := range moves {
		u := e.p.make(m)
		e.moveStack[0] = m
		e.pcStack[0] = pieceIdx(e.p.b[m.To()])*64 + m.To()
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
func bestMove(p *Pos, budget time.Duration, depthCap int) Move {
	searchStart := time.Now()
	legal := p.genLegal()
	if len(legal) == 0 {
		return NoMove
	}
	workers := searchThreads()
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
		c.deadline = searchStart.Add(budget)
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
			mode = fmt.Sprintf("TIME %g s", budget.Seconds())
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
			e.p.nnAttach(&e.nn)
			e.ctl = c
			e.stop = false
			e.nodes = 0
			e.stores = 0
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
				// helpers spread out: with half of the workers on a depth
				// already, a helper goes on to the next one
				for w > 0 && depth < maxD && int(c.atDepth[depth].Load()) >= (workers+1)/2 {
					depth++
				}
				c.atDepth[depth].Add(1)
				first := myBest
				e.orderMoves(moves, first, 0)
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
				// an interrupted iteration still counts when it searched a move to
				// the end and that move beats the lower bound of the window
				partial := e.stop
				if partial && (iterBest == NoMove || iterScore <= alpha) {
					break
				}
				if iterBest != NoMove {
					myBest = iterBest
					prevScore = iterScore
					if !partial {
						e.ttStore(e.p.hash, myBest, iterScore, depth, ttExact, 0, e.evalStack[0])
					}
					mu.Lock()
					// (an interrupted search has only compared its moves with
					// the one it tried first)
					if depth > bestDepth && (!partial || first == best) {
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
							done := ""
							if partial {
								done = " (interrupted)"
							}
							vlog("depth %2d sel %2d | %s | best %s | nodes %s | nps %s | tt %.1f%% | pv %s%s",
								depth, e.seldepth, scoreStr(iterScore), moveStr(myBest),
								fmtN(nodes), fmtN(nps), ttFullPct(),
								pvString(p, myBest, imin(depth, 12)), done)
						}
					}
					mu.Unlock()
				}
				c.atDepth[depth].Add(-1)
				if partial || iterScore > MateScore-2000 || iterScore < -(MateScore-2000) {
					break // stopped, or proven mate either way: deeper search can't change it
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
	s := sqName(m.From()) + sqName(m.To())
	if m.Promo() != 0 {
		s += string("  nbrq"[m.Promo()])
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
		if m.From() == from && m.To() == to {
			if m.Promo() == promo {
				return m, ""
			}
			if m.Promo() != 0 && promo == 0 {
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
	w, b := &p.bb[White], &p.bb[Black]
	if w[WP]|b[WP]|w[WR]|b[WR]|w[WQ]|b[WQ] != 0 {
		return false
	}
	bishops, knights := w[WB]|b[WB], w[WN]|b[WN]
	if knights == 0 {
		// K vs K, or nothing but bishops on squares of one color
		return bishops&lightSquares == 0 || bishops&^lightSquares == 0
	}
	return bishops == 0 && knights&(knights-1) == 0 // K+N vs K
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

const usageText = `usage: chess [gui] [demo|silentdemo] [options] [limit]  (exactly one limit per mode)
  limit > 0: TIME-limited - max seconds per move (default 600, fractions allowed)
  limit < 0: DEPTH-limited - fixed search depth (-10 = depth 10)
  limit = 0: RAM-limited - hash auto-sized from available RAM
             (CHESS_HASH_MB overrides); moves when it's used up
  demo: computer plays both sides, board redrawn after every move
  silentdemo: same but without the boards (moves + result only)
  gui: X11 board window where you play White (same limits)
  gui demo: X11 window where the engine plays both sides
  gui options (each one is also an environment variable):
    --icons    CHESS_ICONS=1    piece icons instead of single letters
    --no-cpu   CHESS_NO_CPU=1   hide the CPU MOVE button (no engine help)
    --input=M  CHESS_INPUT=M    how to move: both (default), click or drag
  gui moves: drag a piece and drop it on its target square, or click the
    piece and then the target. Clicking the selected piece again, dropping
    it back where it was or a right click cancels, so another can be picked.
  -h, --help: this text
  env: VERBOSE=1 full search/debug trace on stderr
       CHESS_HASH_MB=<mb> hash size | CHESS_FEN=<fen> start position
examples:
  chess 5                       text mode, 5 seconds per computer move
  chess gui 5                   board window: letters and the CPU MOVE button
  chess gui --icons 5           icons instead of single letters
  chess gui --no-cpu 5          hidden CPU MOVE button
  chess gui --icons --no-cpu 5  icons and no CPU MOVE button
  CHESS_ICONS=1 CHESS_NO_CPU=1 chess gui 5   the same with the environment
`

// config is what the command line and the environment ask for.
type config struct {
	budget                  time.Duration // > 0: time limit per move
	depthCap                int           // > 0: fixed search depth
	demo, silent, gui, help bool
	opt                     guiOpts
}

// parseArgs reads the mode keywords, options and the limit, in any order.
func parseArgs(args []string, env func(string) string) (c config, err error) {
	c.budget = 600 * time.Second
	on := func(k string) bool { return env(k) != "" && env(k) != "0" }
	c.opt.icons, c.opt.noCPU = on("CHESS_ICONS"), on("CHESS_NO_CPU")
	input, limits := env("CHESS_INPUT"), 0
	bad := func(format string, a ...any) {
		if err == nil {
			err = fmt.Errorf(format, a...)
		}
	}
	for _, a := range args {
		switch la := strings.ToLower(a); {
		case la == "demo":
			c.demo = true
		case la == "silentdemo" || la == "sdemo":
			c.demo, c.silent = true, true
		case la == "gui":
			c.gui = true
		case la == "-h" || la == "--help":
			c.help = true
		case la == "--icons":
			c.opt.icons = true
		case la == "--no-cpu":
			c.opt.noCPU = true
		case strings.HasPrefix(la, "--input="):
			input = la[len("--input="):]
		default:
			v, e := strconv.ParseFloat(a, 64)
			limits++
			switch {
			case e != nil || math.IsNaN(v) || math.Abs(v) > 1e9:
				bad("bad argument %q", a)
			case limits > 1:
				bad("more than one limit: %q", a)
			case v >= 0.001:
				c.budget, c.depthCap = time.Duration(v*float64(time.Second)), 0
			case v == 0:
				c.budget, c.depthCap = 0, 0 // truly unlimited
			case v <= -1 && v == math.Trunc(v):
				c.budget, c.depthCap = 0, int(-v)
			default:
				bad("bad limit %q", a)
			}
		}
	}
	switch strings.ToLower(input) {
	case "", "both":
		c.opt.input = inputBoth
	case "click":
		c.opt.input = inputClick
	case "drag":
		c.opt.input = inputDrag
	default:
		bad("bad input mode %q (use both, click or drag)", input)
	}
	return
}

func main() {
	cfg, err := parseArgs(os.Args[1:], os.Getenv)
	if cfg.help {
		fmt.Print(usageText)
		return
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "chess: %v\n%s", err, usageText)
		os.Exit(1)
	}
	initTT(cfg.budget)
	initZobrist()
	initMasks()
	initLMR()
	kpkOnce.Do(kpkInit)
	nnOnce.Do(nnInit)
	if os.Getenv("CHESS_SELFTEST") == "1" {
		selfTest()
		return
	}
	budget, depthCap := cfg.budget, cfg.depthCap
	demo, silent := cfg.demo, cfg.silent
	p := startPos()
	if fen := os.Getenv("CHESS_FEN"); fen != "" {
		p = fromFEN(fen) // testing aid: start from an arbitrary position
	}
	if cfg.gui {
		guiLoop(p, budget, depthCap, demo, cfg.opt)
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
			if pc != Empty {
				p.set(pc, sq)
			}
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
	if len(parts) > 3 && len(parts[3]) == 2 {
		// kept only when it is a real en passant chance, as make() does
		pawn, rank, to := WP, 2, 8
		if p.side == White {
			pawn, rank, to = BP, 5, -8
		}
		ep := int(parts[3][1]-'1')*8 + int(parts[3][0]-'a')
		if ep>>3 == rank && parts[3][0] >= 'a' && parts[3][0] <= 'h' && p.b[ep] == Empty &&
			p.b[ep+to] == pawn && p.epLegal(ep, p.side) {
			p.ep = ep
		}
	}
	if len(parts) > 4 {
		if v, err := strconv.Atoi(parts[4]); err == nil && v > 0 {
			p.half = v
		}
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
			if !p.consistent() {
				panic("piece sets do not match the board")
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
// Zero dependencies kept: this speaks the core X11 protocol directly and
// paints the whole window itself (board, pieces, text) into a grayscale
// frame sent with PutImage, so it needs no server fonts, images or files
// (works with any X server: real, XQuartz, XWayland, Xvfb, ...).

type x11Conn struct {
	c         net.Conn
	buf       *bufio.Writer
	seq       uint16
	waitSeq   atomic.Uint32 // sequence number of the request awaiting its reply
	ridBase   uint32
	ridNext   uint32
	root      uint32
	rootDepth byte
	visual    uint32
	cmap      uint32
	white     uint32
	black     uint32
	maxReq    int // largest request the server takes, in bytes
	bpp, pad  int // bits per pixel and scanline pad of rootDepth images
	msbFirst  bool
	trueColor bool
	rgbMask   [3]uint32
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

// xAuthCookie finds an MIT-MAGIC-COOKIE-1 for this display number in
// ~/.Xauthority.
func xAuthCookie(dispNum string) (name string, data []byte) {
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

// x11Dial opens $DISPLAY. A display can be reached in more than one way and
// the first that answers is used: a socket path as XQuartz sets it
// (/private/tmp/com.apple.launchd.X/org.xquartz:0), the unix socket of
// display N, TCP port 6000+N.
func x11Dial() (*x11Conn, error) {
	disp := os.Getenv("DISPLAY")
	if disp == "" {
		disp = ":0"
	}
	colon := strings.LastIndexByte(disp, ':')
	num := disp[colon+1:]
	if i := strings.IndexByte(num, '.'); i >= 0 {
		num = num[:i]
	}
	port := "6000"
	if v, err := strconv.Atoi(num); err == nil {
		port = strconv.Itoa(6000 + v)
	}
	local := [][2]string{{"unix", "/tmp/.X11-unix/X" + num}, {"tcp", "localhost:" + port}}
	var ways [][2]string
	switch {
	case strings.HasPrefix(disp, "/"):
		ways = [][2]string{{"unix", disp}}
		if i := strings.LastIndexByte(disp, '.'); i > colon && colon >= 0 {
			ways = append(ways, [2]string{"unix", disp[:i]}) // without the screen number
		}
		if colon >= 0 {
			ways = append(ways, local...)
		}
	case strings.HasPrefix(disp, ":"):
		ways = local
	case strings.HasPrefix(disp, "unix:"):
		ways = local[:1]
	default: // host:N
		ways = [][2]string{{"tcp", net.JoinHostPort(strings.Trim(disp[:max(colon, 0)], "[]"), port)}}
	}
	var first error
	for _, w := range ways {
		c, err := net.DialTimeout(w[0], w[1], 5*time.Second)
		if err == nil {
			var x *x11Conn
			if x, err = x11Setup(c, num); err == nil {
				return x, nil
			}
			c.Close()
		}
		if first == nil {
			first = err
		}
	}
	return nil, first
}

// x11Setup does the connection setup on c for display number num.
func x11Setup(c net.Conn, num string) (*x11Conn, error) {
	authName, authData := xAuthCookie(num)
	req := make([]byte, 12+len(authName)+pad4(len(authName))+len(authData)+pad4(len(authData)))
	req[0] = 'l'
	le16(req[2:], 11)
	le16(req[6:], uint16(len(authName)))
	le16(req[8:], uint16(len(authData)))
	copy(req[12:], authName)
	copy(req[12+len(authName)+pad4(len(authName)):], authData)
	c.SetDeadline(time.Now().Add(15 * time.Second)) // a server may accept and never answer
	if _, err := c.Write(req); err != nil {
		return nil, err
	}
	head := make([]byte, 8)
	if _, err := io.ReadFull(c, head); err != nil {
		return nil, err
	}
	body := make([]byte, int(rd16(head[6:]))*4)
	if _, err := io.ReadFull(c, body); err != nil {
		return nil, err
	}
	c.SetDeadline(time.Time{})
	if head[0] != 1 {
		n := int(head[1])
		msg := ""
		if n <= len(body) {
			msg = string(body[:n])
		}
		return nil, fmt.Errorf("X11 connection refused: %s", msg)
	}
	x := &x11Conn{c: c, buf: bufio.NewWriter(c)}
	x.ridBase = rd32(body[4:])
	vlen := int(rd16(body[16:]))
	x.maxReq = int(rd16(body[18:])) * 4
	nform := int(body[21])
	x.msbFirst = body[22] != 0
	forms := 32 + vlen + pad4(vlen)
	off := forms + 8*nform
	// first screen
	x.root = rd32(body[off:])
	x.cmap = rd32(body[off+4:])
	x.white = rd32(body[off+8:])
	x.black = rd32(body[off+12:])
	x.visual = rd32(body[off+32:])
	x.rootDepth = body[off+38]
	for i := 0; i < nform; i++ {
		if f := body[forms+8*i:]; f[0] == x.rootDepth {
			x.bpp, x.pad = int(f[1]), int(f[2])
		}
	}
	if x.bpp < 8 || x.bpp%8 != 0 || x.pad < 8 {
		return nil, fmt.Errorf("unsupported X11 image format (depth %d, %d bits per pixel)", x.rootDepth, x.bpp)
	}
	d := off + 40
	for nd := int(body[off+39]); nd > 0 && d+8 <= len(body); nd-- {
		nv := int(rd16(body[d+2:]))
		for d += 8; nv > 0 && d+24 <= len(body); nv, d = nv-1, d+24 {
			if rd32(body[d:]) == x.visual {
				x.trueColor = body[d+4] == 4
				x.rgbMask = [3]uint32{rd32(body[d+8:]), rd32(body[d+12:]), rd32(body[d+16:])}
			}
		}
	}
	x.waitSeq.Store(1 << 16)
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
	fail := func(err error) {
		x.readErr = err
		close(x.events)
		close(x.replies)
	}
	for {
		m := make([]byte, 32)
		if _, err := io.ReadFull(r, m); err != nil {
			fail(err)
			return
		}
		switch m[0] {
		case 1: // reply, possibly with extra data
			if extra := rd32(m[4:]); extra > 0 {
				rest := make([]byte, extra*4)
				if _, err := io.ReadFull(r, rest); err != nil {
					fail(err)
					return
				}
				m = append(m, rest...)
			}
			x.replies <- m
		case 0: // protocol error: answers a pending round trip, else reported
			if uint32(rd16(m[2:])) == x.waitSeq.Load() {
				x.replies <- m
			} else {
				fmt.Fprintf(os.Stderr, "X11 error: code=%d major=%d\n", m[1], m[10])
			}
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

// roundTrip sends a request and waits for its reply; ok is false when the
// server answered with an error or the connection is gone.
func (x *x11Conn) roundTrip(b []byte) (r []byte, ok bool) {
	x.send(b)
	x.waitSeq.Store(uint32(x.seq))
	x.flush()
	r = <-x.replies
	x.waitSeq.Store(1 << 16)
	return r, len(r) >= 32 && r[0] == 1
}

func (x *x11Conn) internAtom(name string) uint32 {
	n := len(name)
	b := make([]byte, 8+n+pad4(n))
	b[0] = 16
	le16(b[2:], uint16(len(b)/4))
	le16(b[4:], uint16(n))
	copy(b[8:], name)
	r, ok := x.roundTrip(b)
	if !ok {
		return 0
	}
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
	r, ok := x.roundTrip(b)
	if !ok {
		return 0, false
	}
	return rd32(r[16:]), true
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

func (x *x11Conn) createGC(drawable, fg, bg uint32) uint32 {
	gc := x.newID()
	b := make([]byte, 24)
	b[0] = 55
	le16(b[2:], 6)
	le32(b[4:], gc)
	le32(b[8:], drawable)
	le32(b[12:], 0x4|0x8) // foreground | background
	le32(b[16:], fg)
	le32(b[20:], bg)
	x.send(b)
	return gc
}

// grayLUT maps the 256 gray levels of a frame to pixel values of the root
// visual; without TrueColor it falls back to black, one gray and white.
func (x *x11Conn) grayLUT() (lut [256]uint32) {
	if x.trueColor {
		for g := range lut {
			for _, m := range x.rgbMask {
				if m != 0 {
					top := uint32(1)<<uint(bits.OnesCount32(m)) - 1
					lut[g] |= (uint32(g)*top + 127) / 255 << uint(bits.TrailingZeros32(m))
				}
			}
		}
		return
	}
	gray := x.white
	if px, ok := x.allocColor(0xb8b8, 0xb8b8, 0xb8b8); ok {
		gray = px
	}
	for g := range lut {
		switch {
		case g < 96:
			lut[g] = x.black
		case g < 224:
			lut[g] = gray
		default:
			lut[g] = x.white
		}
	}
	return
}

// putGray draws a w x h block of a grayscale frame (rows `stride` apart) at
// (dx,dy) with ZPixmap PutImage requests no larger than the server allows.
func (x *x11Conn) putGray(wid, gc uint32, dx, dy, w, h int, src []uint8, stride int, lut *[256]uint32) {
	bytesPP := x.bpp / 8
	row := (w*x.bpp + x.pad - 1) / x.pad * x.pad / 8
	maxRows := max((x.maxReq-24)/row, 1)
	for y0 := 0; y0 < h; y0 += maxRows {
		n := min(maxRows, h-y0)
		b := make([]byte, 24+n*row+pad4(n*row))
		b[0] = 72
		b[1] = 2 // ZPixmap
		le16(b[2:], uint16(len(b)/4))
		le32(b[4:], wid)
		le32(b[8:], gc)
		le16(b[12:], uint16(w))
		le16(b[14:], uint16(n))
		le16(b[16:], uint16(dx))
		le16(b[18:], uint16(dy+y0))
		b[21] = x.rootDepth
		for y := 0; y < n; y++ {
			o := 24 + y*row
			for _, g := range src[(y0+y)*stride : (y0+y)*stride+w] {
				v := lut[g]
				for k := 0; k < bytesPP; k++ {
					if x.msbFirst {
						b[o+k] = byte(v >> uint(8*(bytesPP-1-k)))
					} else {
						b[o+k] = byte(v >> uint(8*k))
					}
				}
				o += bytesPP
			}
		}
		x.send(b)
	}
}

// present sends the 64x64 tiles of frame fb that differ from what the
// window shows (all of them when shown is nil) and returns the new state.
func (x *x11Conn) present(wid, gc uint32, fb, shown []uint8, w, h int, lut *[256]uint32) []uint8 {
	all := shown == nil
	if all {
		shown = make([]uint8, len(fb))
	}
	for ty := 0; ty < h; ty += 64 {
		for tx := 0; tx < w; tx += 64 {
			x0, y0, x1, y1 := w, h, -1, -1
			for y := ty; y < min(ty+64, h); y++ {
				for xx := tx; xx < min(tx+64, w); xx++ {
					if all || fb[y*w+xx] != shown[y*w+xx] {
						x0, y0, x1, y1 = min(x0, xx), min(y0, y), max(x1, xx), max(y1, y)
					}
				}
			}
			if x1 >= 0 {
				x.putGray(wid, gc, x0, y0, x1-x0+1, y1-y0+1, fb[y0*w+x0:], w, lut)
			}
		}
	}
	copy(shown, fb)
	x.flush()
	return shown
}

// ---------- GUI drawing: grayscale canvas, 8x8 font, piece sprites ----------

type canvas struct {
	w, h int
	pix  []uint8
}

func (c *canvas) fill(x, y, w, h int, g uint8) {
	x1, y1 := min(x+w, c.w), min(y+h, c.h)
	x, y = max(x, 0), max(y, 0)
	for ; y < y1; y++ {
		for i := x; i < x1; i++ {
			c.pix[y*c.w+i] = g
		}
	}
}

// frame draws a border of thickness t just inside the given rectangle.
func (c *canvas) frame(x, y, w, h, t int, g uint8) {
	c.fill(x, y, w, t, g)
	c.fill(x, y+h-t, w, t, g)
	c.fill(x, y+t, t, h-2*t, g)
	c.fill(x+w-t, y+t, t, h-2*t, g)
}

// text draws s with its top left corner at (x,y); every font pixel becomes
// an sx by sy block.
func (c *canvas) text(x, y int, s string, sx, sy int, g uint8) {
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch < 32 || ch > 126 {
			ch = '?'
		}
		for r, rowBits := range font8[ch-32] {
			for b := 0; b < 8; b++ {
				if rowBits>>uint(b)&1 != 0 {
					c.fill(x+(i*8+b)*sx, y+r*sy, sx, sy, g)
				}
			}
		}
	}
}

// sprite is a small grayscale picture with an alpha (coverage) channel.
type sprite struct {
	w, h int
	g, a []uint8
}

func (c *canvas) blit(s *sprite, x, y int) {
	for sy := 0; sy < s.h; sy++ {
		if y+sy < 0 || y+sy >= c.h {
			continue
		}
		for sx := 0; sx < s.w; sx++ {
			a := int(s.a[sy*s.w+sx])
			if a == 0 || x+sx < 0 || x+sx >= c.w {
				continue
			}
			d := &c.pix[(y+sy)*c.w+x+sx]
			*d = uint8((int(*d)*(255-a) + int(s.g[sy*s.w+sx])*a + 127) / 255)
		}
	}
}

// font8 is the public-domain 8x8 VGA font (font8x8_basic by Daniel Hepper),
// ASCII 32..126, one byte per row, bit 0 = leftmost pixel.
var font8 = [95][8]byte{
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ' '
	{0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00}, // '!'
	{0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // '"'
	{0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00}, // '#'
	{0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00}, // '$'
	{0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00}, // '%'
	{0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00}, // '&'
	{0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}, // '''
	{0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00}, // '('
	{0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00}, // ')'
	{0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00}, // '*'
	{0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00}, // '+'
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // ','
	{0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00}, // '-'
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // '.'
	{0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00}, // '/'
	{0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00}, // '0'
	{0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00}, // '1'
	{0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00}, // '2'
	{0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00}, // '3'
	{0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00}, // '4'
	{0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00}, // '5'
	{0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00}, // '6'
	{0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00}, // '7'
	{0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00}, // '8'
	{0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00}, // '9'
	{0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // ':'
	{0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06}, // ';'
	{0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00}, // '<'
	{0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00}, // '='
	{0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00}, // '>'
	{0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00}, // '?'
	{0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00}, // '@'
	{0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00}, // 'A'
	{0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00}, // 'B'
	{0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00}, // 'C'
	{0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00}, // 'D'
	{0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00}, // 'E'
	{0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00}, // 'F'
	{0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00}, // 'G'
	{0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00}, // 'H'
	{0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'I'
	{0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00}, // 'J'
	{0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00}, // 'K'
	{0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00}, // 'L'
	{0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00}, // 'M'
	{0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00}, // 'N'
	{0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00}, // 'O'
	{0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00}, // 'P'
	{0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00}, // 'Q'
	{0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00}, // 'R'
	{0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00}, // 'S'
	{0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'T'
	{0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00}, // 'U'
	{0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // 'V'
	{0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00}, // 'W'
	{0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00}, // 'X'
	{0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00}, // 'Y'
	{0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00}, // 'Z'
	{0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00}, // '['
	{0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00}, // '\'
	{0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00}, // ']'
	{0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00}, // '^'
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF}, // '_'
	{0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00}, // '`'
	{0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00}, // 'a'
	{0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00}, // 'b'
	{0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00}, // 'c'
	{0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00}, // 'd'
	{0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00}, // 'e'
	{0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00}, // 'f'
	{0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // 'g'
	{0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00}, // 'h'
	{0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'i'
	{0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E}, // 'j'
	{0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00}, // 'k'
	{0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // 'l'
	{0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00}, // 'm'
	{0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00}, // 'n'
	{0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00}, // 'o'
	{0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F}, // 'p'
	{0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78}, // 'q'
	{0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00}, // 'r'
	{0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00}, // 's'
	{0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00}, // 't'
	{0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00}, // 'u'
	{0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // 'v'
	{0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00}, // 'w'
	{0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00}, // 'x'
	{0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // 'y'
	{0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00}, // 'z'
	{0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00}, // '{'
	{0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00}, // '|'
	{0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00}, // '}'
	{0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // '~'
}

// letterSprite is the classic look: the FEN letter on a badge - white badge
// with black ink for White, black badge with white ink for Black.
func letterSprite(pc, sz int) *sprite {
	s := &sprite{w: sz, h: sz, g: make([]uint8, sz*sz), a: make([]uint8, sz*sz)}
	paper, ink := uint8(0), uint8(255)
	t := -pc
	if pc > 0 {
		paper, ink, t = 255, 0, pc
	}
	bx, by := sz/2-14, sz/2-16
	(&canvas{sz, sz, s.a}).fill(bx, by, 28, 32, 255)
	c := &canvas{sz, sz, s.g}
	c.fill(bx, by, 28, 32, paper)
	c.frame(bx, by, 28, 32, 2, ink)
	c.text(bx+7, by+6, "PNBRQK"[t-1:t], 2, 3, ink)
	return s
}

// sdf is a signed distance function: negative inside a shape, positive
// outside, in units of the 100x100 grid the pieces are designed on.
type sdf func(x, y float64) float64

func sdCircle(cx, cy, r float64) sdf {
	return func(x, y float64) float64 { return math.Hypot(x-cx, y-cy) - r }
}

// sdBox is a rectangle with corners rounded by radius r.
func sdBox(x0, y0, x1, y1, r float64) sdf {
	cx, cy, hx, hy := (x0+x1)/2, (y0+y1)/2, (x1-x0)/2-r, (y1-y0)/2-r
	return func(x, y float64) float64 {
		qx, qy := math.Abs(x-cx)-hx, math.Abs(y-cy)-hy
		return math.Hypot(math.Max(qx, 0), math.Max(qy, 0)) + math.Min(math.Max(qx, qy), 0) - r
	}
}

func segDist2(x, y, ax, ay, bx, by float64) float64 {
	ex, ey, wx, wy := bx-ax, by-ay, x-ax, y-ay
	t := 0.0
	if l := ex*ex + ey*ey; l > 0 {
		t = math.Max(0, math.Min(1, (wx*ex+wy*ey)/l))
	}
	dx, dy := wx-ex*t, wy-ey*t
	return dx*dx + dy*dy
}

// sdPoly is a closed polygon given as x0,y0,x1,y1,...
func sdPoly(p ...float64) sdf {
	return func(x, y float64) float64 {
		d, in := math.Inf(1), false
		for i, j := 0, len(p)-2; i < len(p); j, i = i, i+2 {
			ax, ay, bx, by := p[i], p[i+1], p[j], p[j+1]
			d = math.Min(d, segDist2(x, y, ax, ay, bx, by))
			if (ay > y) != (by > y) && x < (bx-ax)*(y-ay)/(by-ay)+ax {
				in = !in
			}
		}
		if in {
			return -math.Sqrt(d)
		}
		return math.Sqrt(d)
	}
}

// sdLine is an open polyline drawn with a round pen of half-width hw.
func sdLine(hw float64, p ...float64) sdf {
	return func(x, y float64) float64 {
		d := math.Inf(1)
		for i := 0; i+3 < len(p); i += 2 {
			d = math.Min(d, segDist2(x, y, p[i], p[i+1], p[i+2], p[i+3]))
		}
		return math.Sqrt(d) - hw
	}
}

// chaikin rounds the corners of a closed polygon with n corner-cutting passes.
func chaikin(n int, p ...float64) []float64 {
	for ; n > 0; n-- {
		q := make([]float64, 0, 2*len(p))
		for i := 0; i < len(p); i += 2 {
			j := (i + 2) % len(p)
			q = append(q, 0.75*p[i]+0.25*p[j], 0.75*p[i+1]+0.25*p[j+1],
				0.25*p[i]+0.75*p[j], 0.25*p[i+1]+0.75*p[j+1])
		}
		p = q
	}
	return p
}

// pieceShapes describes a piece (WP..WK) on a 100x100 grid, y down: the
// union of body is its silhouette, ink are the detail lines inside it.
func pieceShapes(t int) (body, ink []sdf) {
	base := sdBox(24, 76, 76, 88, 4)
	const pen = 1.3
	switch t {
	case WP:
		body = []sdf{base, sdCircle(50, 29, 11), sdBox(37, 40, 63, 47, 3.5),
			sdPoly(chaikin(2, 44, 44, 41, 55, 34, 66, 30, 78, 70, 78, 66, 66, 59, 55, 56, 44)...)}
	case WN:
		body = []sdf{sdBox(24, 77, 80, 88, 4),
			sdPoly(chaikin(2, 28, 80, 29, 74, 34, 66, 41, 59, 45, 52, 41, 50, 34, 54, 28, 60,
				21, 62, 15, 57, 14, 49, 18, 42, 25, 33, 33, 24, 38, 17, 39, 7, 39, 7, 46, 14,
				53, 6, 53, 6, 58, 15, 67, 21, 75, 32, 80, 47, 81, 65, 79, 80)...)}
		ink = []sdf{sdCircle(32, 34, 2.8), sdLine(pen, 17, 54, 22, 53),
			sdLine(pen, 60, 22, 68, 30, 73, 42, 75, 56, 75, 70)}
	case WB:
		body = []sdf{base, sdCircle(50, 15, 5), sdBox(36, 54, 64, 61, 3.5),
			sdPoly(chaikin(2, 50, 17, 50, 17, 60, 27, 65, 39, 62, 50, 55, 56, 45, 56, 38, 50, 35, 39, 40, 27)...),
			sdPoly(chaikin(1, 42, 60, 58, 60, 61, 68, 69, 78, 31, 78, 39, 68)...)}
		ink = []sdf{sdLine(pen, 57, 27, 49, 41)}
	case WR:
		body = []sdf{base,
			sdPoly(27, 16, 37, 16, 37, 23, 45, 23, 45, 16, 55, 16, 55, 23, 63, 23, 63, 16, 73, 16,
				73, 33, 66, 41, 66, 68, 72, 77, 28, 77, 34, 68, 34, 41, 27, 33)}
		ink = []sdf{sdLine(pen, 27, 33, 73, 33), sdLine(pen, 34, 41, 66, 41), sdLine(pen, 34, 68, 66, 68)}
	case WQ:
		body = []sdf{base, sdCircle(16, 31, 5), sdCircle(33, 23, 5), sdCircle(50, 20, 5),
			sdCircle(67, 23, 5), sdCircle(84, 31, 5),
			sdPoly(28, 78, 17, 34, 31, 55, 33, 26, 42, 53, 50, 23, 58, 53, 67, 26, 69, 55, 83, 34, 72, 78)}
		ink = []sdf{sdLine(pen, 28, 69, 39, 66, 50, 65, 61, 66, 72, 69)}
	case WK:
		body = []sdf{base, sdBox(46.5, 7, 53.5, 32, 1), sdBox(39, 13, 61, 20, 1),
			sdPoly(chaikin(2, 50, 28, 57, 37, 57, 46, 43, 46, 43, 37)...),
			sdPoly(chaikin(2, 29, 78, 19, 60, 18, 46, 28, 36, 41, 37, 50, 47, 59, 37, 72, 36, 82, 46,
				81, 60, 71, 78)...)}
		ink = []sdf{sdLine(pen, 29, 69, 39, 66, 50, 65, 61, 66, 71, 69), sdLine(pen, 50, 47, 50, 64)}
	}
	return
}

// iconSprite renders a piece icon sz pixels wide: White is white with a
// black outline, Black is black, both with detail lines in the other color.
func iconSprite(pc, sz int) *sprite {
	t := pc
	if t < 0 {
		t = -t
	}
	body, ink := pieceShapes(t)
	s := &sprite{w: sz, h: sz, g: make([]uint8, sz*sz), a: make([]uint8, sz*sz)}
	k := float64(sz) / 100
	cover := func(d float64) float64 { return math.Max(0, math.Min(1, 0.5-d)) }
	for y := 0; y < sz; y++ {
		for x := 0; x < sz; x++ {
			u, v := (float64(x)+0.5)/k, (float64(y)+0.5)/k
			d := math.Inf(1)
			for _, f := range body {
				d = math.Min(d, f(u, v))
			}
			all := cover(d * k)
			if all == 0 {
				continue
			}
			inner := cover(d*k + 2.2)
			line := 0.0
			for _, f := range ink {
				line = math.Max(line, cover(f(u, v)*k))
			}
			white := inner * line
			if pc > 0 {
				white = inner * (1 - line)
			}
			s.a[y*sz+x] = uint8(all*255 + 0.5)
			s.g[y*sz+x] = uint8(white/all*255 + 0.5)
		}
	}
	return s
}

// ---------- GUI state: mouse handling and rendering (no X11 in here) ----------

const (
	guiSQ      = 64
	guiBW      = 8 * guiSQ
	guiStatusH = 40
	guiBtnH    = 44
)

const (
	inputBoth  = iota // click-click and drag & drop
	inputClick        // click the piece, then the target square
	inputDrag         // drag the piece and drop it on the target square
)

type guiOpts struct {
	icons bool // piece icons instead of letters
	noCPU bool // no CPU MOVE button
	input int
}

type gui struct {
	p       *Pos
	opt     guiOpts
	demo    bool
	think   func(*Pos) Move
	out     *bufio.Writer
	engCh   chan Move
	w, h    int
	fb      []uint8 // grayscale frame, w*h
	sprites [13]*sprite

	sel                int // selected square, -1 = none
	lastFrom, lastTo   int
	promoFrom, promoTo int
	dragFrom           int // square of the piece the mouse holds, -1 = none
	dragX, dragY       int
	pressX, pressY     int
	dragMoved          bool // the held piece has left its place
	wasSel             bool // the held piece was already selected when pressed
	pressBtn           bool
	thinking           bool
	pendingCPU         int // engine moves left to play in a row (1 reply, 2 for CPU btn)
	over               bool
	status             string
}

func newGUI(p *Pos, opt guiOpts, demo bool, think func(*Pos) Move, out io.Writer) *gui {
	g := &gui{p: p, opt: opt, demo: demo, think: think, out: bufio.NewWriter(out),
		engCh: make(chan Move, 1), w: guiBW, h: guiBW + guiStatusH,
		sel: -1, lastFrom: -1, lastTo: -1, promoFrom: -1, promoTo: -1, dragFrom: -1}
	if demo || !opt.noCPU {
		g.h += guiBtnH
	}
	g.fb = make([]uint8, g.w*g.h)
	for pc := BK; pc <= WK; pc++ {
		if pc == Empty {
			continue
		}
		if opt.icons {
			g.sprites[pc+6] = iconSprite(pc, guiSQ)
		} else {
			g.sprites[pc+6] = letterSprite(pc, guiSQ)
		}
	}
	switch {
	case demo:
		g.status = "Demo: engine plays both sides."
	case opt.input == inputClick:
		g.status = "You are White: click a piece, then its target."
	case opt.input == inputDrag:
		g.status = "You are White: drag a piece to its target."
	default:
		g.status = "You are White: drag or click-click a move."
	}
	if res, why := gameResult(p); res != "" {
		fmt.Fprintln(g.out, res+" "+why)
		g.out.Flush()
		g.status = res + " " + why
		g.over = true
	} else if demo || p.side == Black {
		g.startEngine(1)
	}
	return g
}

func (g *gui) applyMove(m Move) {
	g.p.make(m)
	g.lastFrom, g.lastTo = m.From(), m.To()
	fmt.Fprintln(g.out, moveStr(m))
	if res, why := gameResult(g.p); res != "" {
		fmt.Fprintln(g.out, res+" "+why)
		g.status = res + " " + why
		g.over = true
	} else if g.p.inCheck(g.p.side) {
		fmt.Fprintln(g.out, "check")
		g.status = "check!"
	} else if g.p.side == White && !g.demo {
		g.status = "Your move."
	}
	g.out.Flush()
}

// startEngine makes the engine play n moves in a row.
func (g *gui) startEngine(n int) {
	if g.over || g.thinking {
		return
	}
	g.thinking = true
	g.pendingCPU = n
	g.status = "Thinking..."
	pos := g.p.clone() // the search must not touch the position being displayed
	go func() { g.engCh <- g.think(pos) }()
}

func (g *gui) engineMoved(m Move) {
	g.thinking = false
	g.pendingCPU--
	g.applyMove(m)
	if !g.over && (g.demo || g.pendingCPU > 0) {
		g.startEngine(max(g.pendingCPU, 1))
	} else {
		g.pendingCPU = 0
	}
}

func guiSqAt(px, py int) int {
	if px < 0 || px >= guiBW || py < 0 || py >= guiBW {
		return -1
	}
	return (7-py/guiSQ)*8 + px/guiSQ
}

func (g *gui) hasBtn() bool { return !g.demo && !g.opt.noCPU }

func (g *gui) inBtn(px, py int) bool {
	y := guiBW + guiStatusH
	return g.hasBtn() && px >= 8 && px < g.w-8 && py >= y+2 && py < y+guiBtnH-6
}

func (g *gui) canMove() bool {
	return !g.demo && !g.over && !g.thinking && g.p.side == White
}

// deselect drops the selection, a drag in progress and the promotion chooser.
func (g *gui) deselect(msg string) {
	g.sel, g.dragFrom, g.promoFrom, g.promoTo = -1, -1, -1, -1
	g.status = msg
}

func (g *gui) humanMove(m Move) {
	g.sel, g.promoFrom, g.promoTo = -1, -1, -1
	g.applyMove(m)
	if !g.over {
		g.startEngine(1)
	}
}

func (g *gui) tryHuman(from, to int) {
	if !g.canMove() {
		return
	}
	var cands []Move
	for _, m := range g.p.genLegal() {
		if m.From() == from && m.To() == to {
			cands = append(cands, m)
		}
	}
	if len(cands) == 0 {
		g.deselect("Illegal move.")
		return
	}
	if cands[0].Promo() != 0 {
		g.sel, g.promoFrom, g.promoTo = from, from, to
		g.status = "Choose promotion piece (click elsewhere to cancel)."
		return
	}
	g.humanMove(cands[0])
}

func (g *gui) promoPick(px, py int) {
	x0, y0 := 2*guiSQ, 3*guiSQ+guiSQ/2
	if px < x0 || px >= x0+4*guiSQ || py < y0 || py >= y0+guiSQ {
		g.deselect("Move cancelled.")
		return
	}
	pr := [4]int{WQ, WR, WB, WN}[(px-x0)/guiSQ]
	for _, m := range g.p.genLegal() {
		if m.From() == g.promoFrom && m.To() == g.promoTo && m.Promo() == pr {
			g.humanMove(m)
			return
		}
	}
}

// press handles a mouse button going down at window position (px,py).
func (g *gui) press(px, py, button int) {
	if g.demo {
		return
	}
	if button != 1 {
		if button == 3 && (g.sel >= 0 || g.promoFrom >= 0) {
			g.deselect("Move cancelled.")
		}
		return
	}
	if g.promoFrom >= 0 {
		g.promoPick(px, py)
		return
	}
	if g.inBtn(px, py) {
		g.pressBtn = true
		return
	}
	sq := guiSqAt(px, py)
	if sq < 0 || !g.canMove() {
		return
	}
	if g.p.b[sq] <= 0 { // not an own piece: the target of the selected one
		if g.sel >= 0 && g.opt.input != inputDrag {
			g.tryHuman(g.sel, sq)
		}
		return
	}
	name := sqName(sq)
	switch g.opt.input {
	case inputClick:
		if g.sel == sq {
			g.deselect("Selection cancelled.")
			return
		}
		g.sel = sq
		g.status = name + ": click the target square, " + name + " again cancels."
		return
	case inputDrag:
		g.status = name + ": drop it on the target square, back on " + name + " cancels."
	default:
		g.status = name + ": drop it or click the target, " + name + " again cancels."
	}
	g.wasSel = g.sel == sq
	g.sel, g.dragFrom, g.dragMoved = sq, sq, false
	g.pressX, g.pressY, g.dragX, g.dragY = px, py, px, py
}

// motion handles the mouse moving while a button is down.
func (g *gui) motion(px, py int) {
	if g.dragFrom < 0 {
		return
	}
	g.dragX, g.dragY = px, py
	if dx, dy := px-g.pressX, py-g.pressY; dx*dx+dy*dy > 9 {
		g.dragMoved = true
	}
}

// release handles a mouse button going up at window position (px,py).
func (g *gui) release(px, py, button int) {
	if g.demo || button != 1 {
		return
	}
	if g.pressBtn {
		g.pressBtn = false
		if g.inBtn(px, py) && g.canMove() {
			g.deselect("")
			g.startEngine(2) // the human's move and the reply
		}
		return
	}
	if g.dragFrom < 0 {
		return
	}
	g.motion(px, py)
	from := g.dragFrom
	g.dragFrom = -1
	if !g.dragMoved { // a plain click on the piece
		if g.wasSel || g.opt.input == inputDrag {
			g.deselect("Selection cancelled.")
		}
		return
	}
	if to := guiSqAt(px, py); to >= 0 && to != from {
		g.tryHuman(from, to)
	} else {
		g.deselect("Move cancelled.")
	}
}

// render paints the whole window into g.fb.
func (g *gui) render() {
	c := &canvas{g.w, g.h, g.fb}
	at := func(sq int) (int, int) { return (sq & 7) * guiSQ, (7 - sq>>3) * guiSQ }
	for sq := 0; sq < 64; sq++ {
		shade := uint8(255)
		if (sq&7+sq>>3)&1 == 0 {
			shade = 0xb8
		}
		x, y := at(sq)
		c.fill(x, y, guiSQ, guiSQ, shade)
	}
	for i := 1; i <= 8; i++ {
		c.fill(i*guiSQ-1, 0, 1, guiBW, 0)
		c.fill(0, i*guiSQ-1, guiBW, 1, 0)
	}
	mark := func(sq, t int) {
		if sq >= 0 {
			x, y := at(sq)
			c.frame(x, y, guiSQ, guiSQ, t, 0)
		}
	}
	mark(g.lastFrom, 2)
	mark(g.lastTo, 2)
	mark(g.sel, 4)
	held := -1
	if g.dragFrom >= 0 && g.dragMoved {
		held = g.dragFrom
	}
	for sq := 0; sq < 64; sq++ {
		if pc := g.p.b[sq]; pc != Empty && sq != held {
			x, y := at(sq)
			c.blit(g.sprites[pc+6], x, y)
		}
	}
	// status bar and CPU button (no button in demo mode or when hidden)
	c.fill(0, guiBW, g.w, g.h-guiBW, 255)
	c.text(8, guiBW+12, g.status, 1, 2, 0)
	label, y := "", guiBW+guiStatusH
	if g.hasBtn() {
		label = "CPU MOVE (plays for you)"
		c.frame(8, y+2, g.w-15, guiBtnH-7, 2, 0)
	}
	if g.thinking && (g.hasBtn() || g.demo) {
		label = "THINKING..."
	}
	c.text(g.w/2-len(label)*4, y+12, label, 1, 2, 0)
	// promotion chooser overlay
	if g.promoFrom >= 0 {
		x0, y0 := 2*guiSQ, 3*guiSQ+guiSQ/2
		c.fill(x0-8, y0-8, 4*guiSQ+16, guiSQ+16, 255)
		c.frame(x0-8, y0-8, 4*guiSQ+16, guiSQ+16, 1, 0)
		for i, pr := range [4]int{WQ, WR, WB, WN} {
			c.frame(x0+i*guiSQ, y0, guiSQ, guiSQ, 1, 0)
			c.blit(g.sprites[pr+6], x0+i*guiSQ, y0)
		}
	}
	if held >= 0 { // the piece being dragged follows the pointer
		c.blit(g.sprites[g.p.b[held]+6], g.dragX-guiSQ/2, g.dragY-guiSQ/2)
	}
}

// guiLoop runs the chess GUI: an 8x8 board window where the human moves
// White's pieces by drag & drop or click-click, with an optional button that
// makes the computer move for the human (the text-mode `c` command). Moves,
// checks and the result are still printed on stdout exactly like text mode.
func guiLoop(p *Pos, budget time.Duration, depthCap int, demo bool, opt guiOpts) {
	x, err := x11Dial()
	if err != nil {
		fmt.Fprintln(os.Stderr, "gui: cannot open X display:", err)
		os.Exit(1)
	}
	g := newGUI(p, opt, demo, func(q *Pos) Move { return bestMove(q, budget, depthCap) }, os.Stdout)
	// press, release, motion with a button down, exposure, structure
	win := x.createWindow(g.w, g.h, 1<<2|1<<3|1<<13|1<<15|1<<17)
	x.changeProp8(win, 39, 31, "chess-fable") // WM_NAME
	wmProtocols := x.internAtom("WM_PROTOCOLS")
	wmDelete := x.internAtom("WM_DELETE_WINDOW")
	x.changeProp32(win, wmProtocols, 4, []uint32{wmDelete})
	// lock the window size (PMinSize|PMaxSize)
	hints := make([]uint32, 18)
	hints[0] = 16 | 32
	hints[5], hints[6] = uint32(g.w), uint32(g.h)
	hints[7], hints[8] = uint32(g.w), uint32(g.h)
	x.changeProp32(win, 40, 41, hints) // WM_NORMAL_HINTS / WM_SIZE_HINTS
	gc := x.createGC(win, x.black, x.white)
	lut := x.grayLUT()
	x.mapWindow(win)
	x.flush()

	var shown []uint8 // what the window displays; nil = repaint everything
	dirty := false
	for {
		select {
		case m := <-g.engCh:
			g.engineMoved(m)
			dirty = true
		case ev, ok := <-x.events:
			if !ok {
				return // display gone
			}
			px, py := int(int16(rd16(ev[24:]))), int(int16(rd16(ev[26:])))
			switch ev[0] & 0x7f {
			case 12: // Expose
				shown, dirty = nil, true
			case 4: // ButtonPress
				g.press(px, py, int(ev[1]))
				dirty = true
			case 5: // ButtonRelease
				g.release(px, py, int(ev[1]))
				dirty = true
			case 6: // MotionNotify
				g.motion(px, py)
				dirty = true
			case 33: // ClientMessage (window close)
				if rd32(ev[12:]) == wmDelete {
					return
				}
			}
		}
		if dirty && len(x.events) == 0 { // repaint once per burst of events
			g.render()
			shown = x.present(win, gc, g.fb, shown, g.w, g.h, &lut)
			dirty = false
		}
	}
}

// ---------- the network ----------

// nnData is the evaluation network in the format of nnLoad, deflated, in
// base64 (written by nnue/embed.go). It was trained on games the program
// played against itself.
const nnData = `
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
8ts5C2/kfQhaeg3WAwWoyITBHE/gZqoEumH3btz4P/g6oaVYt3+TxuiHE7M77kxRSalJjf2VHQ4l4FwiZA3KwTsNr9DRsStgCII2wJTIKccesR
A+Hbm04AvLxD1FymYfovk9w1CSdopabkLoNuSrIvNHthSzBZBA/lPV7i8OV/DGtMzlzDnobpGTog9SOtpGx5UNbks9oTMgxikZSW6vCT8W/g0q
wprPhAjI+1AbFWn/SxxaDoKymLlyyaf/S2jjdHJF7bICi9QOUuJESz7OS6+lOHoq5U2b7jSr22MP7fbm2we/HZsy/ewTT8P+g3qWd37zf++ldK
8i887xFwS74b7sgt9jBdA3+4fiQa+2GX3P0fKfLw2A1bwX7rmLnTL5uIlsfTPaFpJdT15AC7/BbG3nh19FdwcknymsXfFk+iDxRiy/lrT2+/0r
9Ud77yxs3c7uXNFgF9+u3XiSrf6f/K/P9aY+wOUvYprZIWX4VvQb/0la1bsSj+T0caEzF/68d1+23h/vbiup2D+Zzw5gwl3IiQaJvY/Oq9llve
2zVl4K9s0cLX3hUx5+qP4WbYcx3bg1IiXWUwzTQ6P3rl4MbZYPeGpSdvPJk4/k9TyvrP9OBWf/UufqqXu+U/zPa4lFQl5BffQ//rxCB+zK3u+l
++9WWzSdeovDUby7v0ofb0Zv6CfvrWiJgL7baVySy2YKNRdkB0jl5KiX0W/PrvupD4u9BtSo8PdX3/p3rK1Z0Ht4m3Pn7rs91X3iS7TfXO7clL
PVDaUIX+yy0Tp3ba/2Z9MAPgz736q1+upXTLTK/2D35R3vnpzEH0T39H9oWbDw1mF80ZaNy8G3/8P2UfeK80fzbTgN/pR8zsY2OwD/EVelu7Cf
Y35p0bfwmF3ubyO49qczPMjNee6jN6cg87f2vRnqmz5u2uE352huJy+7D42UbGtE8/uLTvvYnc+QK4rSUn89r/8xb1Z5e3KTSy+zY7fLenpm1P
QyBf2K6/oLx1h4zjyw+m2yOkQyy72JYYqatTvpvf7h8xGfuAPn9y8UbNPfgHIxXsi/eaRrd5In3z7M6XX/nlZ7d3ukmv//YruzvOlhe1fHPwh5
18e7WzeQ8apbTzdVPt8wyNz34ubzR+9q1Xfgb8KdIoC7zf/O2IuSLMNWwm+k0R+7WT+ZbSanU4uaagZ08egXBScEWJAZl+m9Aynf1L//jtZqof
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
knjqn9zbXx4fj2wM5p4vK92fA11/T609YO8+zf+FruudTU8tGwWNy+eLzYG1t7d7pSFQOR5zHl5h3p6HByP5jpD8VGdt2bbNSXiA2R7oX7QyMn
/cGHX823J4P2d6ZC3nVHN3S4sLv4cC3t6LuKcL9cXncd7nxfv3TQZ71DPVP7HI4lvzvhk9K7/kbf+ubCYeKGNvBh78rxkyXHQu8gVHA+qnofzS
Ujq8sDTW9m6ocoHf/K2ikcDWZe+LaltcWpxYX4sPPsYA23Hw2Nb5QPz/Y3YuOdPncwUQreTw6XNs7GUr5A/qbLM9Y/ezggSD3X1iF/YG4i7Blt
i+rGs7Hevkx8bRW58u6BT2K2Dx+1KpHF7/nKtgX/6k4+CB2uOHem+1zzz9ZWs85UqhxCRubFzZn5F2I3ORy5AdLFaF8snAhYru05b99pr7L0/2
w2F0VzpjYxvGUI3pWlveeT88PA+ATv2q82iamDP/EPyuGTjz0rzfRIKNMTfS0SEY/ynd7cxqcb9LAy8GnvxycH+Vgh3gctJwqZcH6GDSzKVyqD
86tLvZ6e38jsJqTh5kfuHLm/O7WwO+RMTC4+jnqfjPV8OnWamb+IJ6netGN8Of1LInXvzkk8GRjKpnxeb3Lj0BeLuKJbaBtZGvQN25fZrZWT+K
D7rG/37sTVn6/3pRw7jUyx4tz8jlKIvkivPpoPu1YOudWdkcDOD7c3H/j9L6vJ0vBeonA8/T6sXE+MXW0jJW48kUy5MLumh1PzHnH3jra8FyLu
5j4fW7KL/NqjfOX24ZAvEE4EFpZz6c7RrjSYTQM/iW56U8FXF8NXnIzNuTM801t/vLXt98H+51e2hRdlco8QlZ/y5Y888KO462JtCKLeXl5x8p
Evr+66eiXtG56Z8NOphW/WyTf67qkrN6ijHnpqtlBbr7rcvX3XLvuyoU1HYvGkwImhV1cCCe+4NLvSH45fTc1PorGDN/oC8SPo8fL7qTHPc3Kt
yQw93uzr781XMxdKdPYRdhxZBCdftK6Xnq/FjucGIzPfzlVfXOVHL5JLgWUffPC1XHj/lWHttYZr35mIefpH67GDpnfgwOvfelA/nPShM1tjkW
kBCD3fPCl/Nz/GDYj+vs31g9CSfsIEc9c7j0OraIhMbYGv+6LRwSy6k9xd3D5M9AnXPkocHxweB86946WN+MlmlF8YGGvMJts9T0sbQSH5GyeZ
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
HrymcgPH9DfxgaW1ZzEl/iur8SjT65jc4/C0L3rerlKTwS9ZyW1+bXr1Uir1LFrZZzz3Zws9geXyk+yh9+LP8CLznJrf2RtI7s16trdS7OyHlc
IOe8vXDe2MrkZW7vz4AN3qsdlTu84sOypFBkbUmHbo8K73w++/Z5QWelx3ZyKXlj1F19QH+txKtpfyj5GvrPxvubEmd88+cHA9fy+0kXf3XDrL
JdKCNTqLfwqfz/xyyTHR7fvOmw8az84jsj9SX46d3Qn5HnqwtUlJdHw4Nw8JyTfnzmID447vBZZTdz+Zvqhfq4fmrKega0VyL87fXzrPS7V0TE
uMTvv+kxDMvNhPDQo3l1+VUtHa9MV1Fzbk3SnBodWeW0bfs0Bm4yNxwvbIvkucXH1/vW88VCZy1vzY3F9zGd8L4mdzC+HsxvOeRwu9obOP33o6
3xdLKzd3Q6dDZ7/8sw3/kRHyfo5JbbPpgNNzQ0tVZ9ID61737Q8L50tD0XsbNwKrUTuU3zKZyeLVSDKVkJ7f2xveAT92jedvvfOQ+KkaBwc3Dr
84aFNdWSwTWJl6Ojfat2L70/9qDf/hQ3c9kOoukc9/Mig8udK7vNTZxX6eX3J4+l8+2wr1XKv+bXqW+HTxp5MLwYhz+uRxIryb7r1/p9St7nmR
r1CjkfB+/7zCNg+tSyLz24/Hk7dG/LPrLgPriz3PxW3bSZbofYDv7z0aDl7Zcj1rU6/9r9Pe11wPt5rW5qvrX/vgTPzEtfc0CzY/zPjmnt+4ef
Kf7h/sJW9NCLat4Jrv+Qf/+uCVvcutl+OH1knfrYFwJM6N1s6dVx4GhIfvZ0pHLtfHp5c9BaJFb97O5R9M3+zvT958efbby8OllbeHrhYmWpPY
C9MrxerZN2ORbu8stRbOryzPxXq70W/9k5+63j3uWZeHtp4gs08C+IOXo7sNBos/PXj2uj72G0c7wqWxg4f2VWR+7XHjyN0jPcg+Gxkv2BJ/tP
Uc2s5IdSGC8J7V2A8x58XZZtrm/c3p68noYOTeso+IC6EP1+lAJjxABvLiweZkenS8El1pk2/8q2OHN7GvbVMX0vTXiYz3ydj0vWkDePFu4MHz
vmsrH80un4YmhiH7Snp6eOVHXz0YXx5sRBPnw3M3Lr8sUgH8UvHk8pXTNLP43Vpj1W2fXPGP7g4oob07k3OrJ9ecsdHhZPZbO2PF7tfSr7cmjr
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
w8RhLjxmPDJo4WjKQLONe0NKi1Atj6b9EAKPBNputNLsszsEArdTiYD6msIoFsFT9dAxP2VoDcflFs4XP5EOFY4CuS2eUrmGdkkCZAEpBDNHSy
1rOaoWSbQ2DwLkGiYLb3QXA+5c6G/18giEzNXUUn24mCfyXKJp0OCp2ihwwApSlj29lFdD45TSZFXDulZOlrkeANf5DWMpMy9tzVuLqnQuCtG+
4I6xfp8WQL0pl0Zqg3i9voQ413EnZWsrY8D7WzXFkLrAwLbQjZcAXZXD7nPkkoUc6Y/niiE4Uw+5PGZbYrVdVvut/HBCNzqmArEAapVyA2GUti
S15oRUDmmghJ0ZMiQVsyC70Fw4YGQ+jpj6w55ZnRwELaTeoRbG9WFhkjGfSQ2klp1WIimQpRykbShixYHkXFrhDCw0mTFjFKFqgnE+ivWajGk9
MfO0gw1aAyvHbExerz0SYnMmxCznBjQyz55v3szBukftiDqmizqiD9WHFXkPYgbFnVh+MmzZQ5OfqufoLDMrsm1Ds4rNmRmbc4uZC0gWSV0jtu
y1WHwBT90ej1oaW6W/9E5wR1m2VHM42CDIUtvIFTMBV0lGZhd1Uj0D42OUC9g7FJ6IL46kejScOeiXZUK014tBkpVWBSo2srUGQy61hEb9Dtk0
67YcCD4EldTZC2d04MhSY1oqjfEgNbbYE8wJjosRtktWA6f0mASElw4qj58wt7OvDq0sfaYHkxO1UN85q9YscFuUUMuIsz7dRO7puvK+BlusK2
TX6XStkV3RKeP8GRvpAKUJ0GR3Z5PSBq8xQ1tNSUH2mWaAEnzLPGZvu4UsVlEIEXEGXTuk5eWplHHUk420xP9xD6NFOYiN5Syc35dTlBRhvVuj
bSu6CQptzJaPNXZlLRdID0cdBn0DX5CVolQm35JczNoTdMptVMhFmCHEOu5SMtyi1MwUxYDcnERXRm3E7O6NWqwu+VRzCVhvuK2NCVXqVneljj
RVhCyNM5sbQOukRYpfE7a1RBJ3rGkRV3Hn2Io26uN5q0M3EqF0t9Bn5HE9p8yZqsYwpaJO8XhipcLorw473caKRZH0Uq3VhL5DYJUd02G+QuuN
KNNU0pCd70vHQ/Jo6u9ae3YlR1JgDiGEyhBq6HkanqER8zI2TcyLsxcoWdqp4lRpwxQzK5gCYlSEqkjigVBEEXCES46TM8oVOqFOgSNaZ3QFbB
nWKkBzpN6q91BNt1RK9xiiLo2Cp/JVIckm5HWJpUFFPQgvXBuV2oAnEl4bTyRWmDNUEzUESzWBsbtCMWnI5Mk2PV1k2kppUNqUTeYSii5srhsr
K8pwr0hPdptDGy+QtGUrwhKPz+9TFARoLJgZ81SBmgP0W+ks1gltKqZJGd/YH6qbc21b3EGJcTYRnbQhFEjyXzAxmI2Qgi1T3aKX8TuaaYcrLf
Op8xIznnKhhaFQIZCX8BEoFkxaBL66z8/wuApRs5tjsJli8lAC/w9pgeUW9hNgf9ipj6lO8owvFEFopMVtUYejAEIYNpFag0aZS3jtUnwuhcxG
Rg0byCHtbMAzq1OIkOcinLqDHUOU2Z8ubSJrBeJIWyJFD/MSTUZRqcut4tTy5mB9xlYliTgin/FB/pNRpdckNHo2HaEItDBhgyOWtyYC8wIRMk
QdFYtsVpGxeL5Qw71UYsvKtQ04ymp/PlIV52hHjWKjGQTab1jP2FiZmxenBoagVU9V19xETMxl8pXshmSyz2A1RAiXZG5CeEuqwAY0qWEmKRv4
dCYtUSozxopRlVHqcdlkkkeGR6+Tcap/ORdRICFTDIH5urjGHXZjfdNNtl8mOC6IFOKola7QW2YHP1eBJinnvEU28bsgReQgHYoXgDQxHtjaCt
pq7ExtrBqkw5aIKFAoSQgxX6MXiBUzTU7jSZpMyXhC7vqnoUKbUl46hdOSc9GTjPS68YikoI96gim3uKTMpHRmaZYNsGCb1gAz1LY6bCwT5SNJ
glqqMyod8sUSlGXnL8TFJpve8WHGisNMnRHo7DFZRJNx0Bp16ZPSu9wmBxDFOliDzUWGDJWBIi6w7QKOajiAZYBuiawtegJan+bh9UK1wmCQF8
8ADKMA4lQDmYCtcQurSeXtF8lMy1mb8+gLLLHHq4ucEiJcxRE9qYjonYvqUivl7+uvVbbMsGsgWNjSUVgqGmpEOJHOlHgX9CapQViH0jX1VMXt
UOEWqzWmjU96aOym0cQcpnpBRYYl/IVgDi5NqumMOQMaAt7MqFdX8vlMoe82DxGewQIRIKQn7fITfrglz0KLPgHnReMvZAlkQeactiwlm5rHqg
PNPAVgcWakGb11Y1G0+IiQ5Sy2CZBYRMVpc31TBBmUXAmcr8PkoYRXsjCJTCad4tEzIo1MBc3RaUrCcsgid4sZ4hgizUYDobN7gbcwCWlBFxH6
MaEypdwUlmvgrCYlp+lKbiZPopmFu6wK1VnloVqCXSRoZVXmBJc/Z9pHdCFVOOqiU3K+39HyLUJmt4ZTMIuYl8NK5VYRHGiwaWHcrzc6UhqlOy
CYhkxTTZidSUkcqax5g4TKlxvG7kKMLzJpbmNoLenbVbS5llxg5GVBmyX/SjNujQbwhc9R93DZMlQxmPIFbYEgWjWCKdGkJ2nwJIzOsoVOtugj
kmw66g+1ZJEaUJJ4TBvjFJMu1fzpDHBy0pUBMVHCmbgbggjAyNGEcBOMz4qYSNCNdgsocWvanyKNfJ8dZxk0vfaEXjXCS5FHncbPuYP571Kxrr
6kbk75qMQ8HpoFBYp62WfPKGZxc1gncqzjJW7h3gbi0mmPi9Ly5CYjBMFZuLp2Uc2Tuo1ZQyTUTXG6ptdhdQRxyoBBsGvBm/4QvvCqGQT3kojk
bckhs0ajT9s+gNIHO4IUK937IzM4JsCzcNaccgHyWcXWZgyBIU9DgS1wGeiOxZB5AfSX84FstT/EpYT2Wp8pUTTtA2amrzeV+5rFiKCCnKsMNO
mjFmCUM1odNKhKVjGWn5ozk3GhpJqB2CSdYxnK6Ix9U6zPl1r+ah6BuS07AaN0RL05OB8szkUFtbmIBpkTAmOzNDRQM0J6i66bye5IGb0aa8Ye
5EEBXWLwsQaIsS6B6cgIO8eQIteI5Uumel3YUNAoy+FkMe0LIEWJT+dK6Ip8bZxBLjd12jaXzb2oWZtUVtk+btTHWx2UNmzn6Wn2n+vmdEp5cF
V11J0ss6rS6iO9OUnEIV2LhlVOi6CdGeey8rSPkdT4BFFw5w6KjCPSMfqLrosK5DoCHk0u7qtdwkbhxXIYqolH64sSUSwY5eNlzRJvYVPeCKz1
kKaYPISY2UyKRJxwxfpaG6YOtoddutFGbw44y+lj01RmyeY7TDaV6YFSGpFfRhmg9KlUph61xoO2aahF+VSiquot9ZTYWk6EosUJWIEaXAqcI+
MgAxVqlAQazzmGmZC+pFTa9DI1luw2Wo1VJX8mF5qDLV6PoocZWB9v8udtHMsUiVnen4jnM3IVK9LcebsQICGuVrXEnUi1SFqQLUi6TW6fxZfE
qGSIQZgg3ZEEZqJiF0Fp6s0XIpO6O5JO5BwlU+BRdWKV/mBcolBWpuhCqIOSyfKoDy6rN2lmlYPHF+MJTjT1Yol6H7SY5eU5qmhLB76hUzkcT9
gL+bk9oI6d84f6nkqkp1YQWo5tG4BmLv+OebrE0A6rwmjKaDI5rOv3nHMqxIGQ4KDdBesyBTuianYEmEioJdMY+GzKG20M2Cq13MrEBtkG9WqD
qe4yNfpMS4UtpH3ist/tsjodYsWf0FwpfwwQMGFsnCc4MhapweDvjgbkRZ7DaCUBKEvlNhgLbkcRHRFrEnanOwS0bSnBzJYxl5M0QjYqyiPmSE
oXMSkufikW7lTqifHMO+2zuUqoHslQcKwXhMms9KSbHwKUwVKN5OMTNYutEOJDyIzYFjWm3VbAhfol45F5qNPFeH1KBaBq2zk1rSUWTwZSOp+r
xabU6kQIWvuaiRRIk9G+dsBGKdfddBucw7B1WMsC+SazRdNxSQPDNwhN0lLWl/EAsYhBuOCUSO0VUm0oEeozQ3l7xO7I/HhnzoSh6Yjdn0bxP5
D3QXl+GFoas3a5cIHfxfp91pixIOVU+bRqAdGaYoqi28MKBlqvPSKpRhZUQ+OzBWVosCQDrbqXRrTJRHyHDMqUlgiamfVZeJXLBlrpUH5MHNFn
Pq69Hx6GMFOy36c1K2mcXFZrVONyidxgE5CuVbnIEbnc5T5vIWVE2RPJVKKwZiJGbU0gg0A1maSTjgah7ITWEkR1RKskr5kjQHsQmFUnpfiwDW
EaJRQXd1FBlydhYyNZh1wDRsIUjTuwOURd/wKcxNfs8xFL74whwilFh8pipEZXC0LDUBcSBXpfSXolcaNdGZmTViXXWpEweoi4TxXTUtzZFZnc
FEz5GCPWbjREnB5MZug6JpJKIgKYo7mPbxIXKrIR2SUaUFckLV+/0UuGyRi7xR5xlg2ohosHZLxb1mvIMSyDcDrDGb1TSg/kME8t7YFdasfBQT
omEwZypWzRAdAETblJm9rQCLVwQLJsihWNuXQu1CwpPUF1pmsTIeFAOZ8RweEMnw+pUDMeHTRakEBZLVIWiEHNIdFbBCp/0iJg9NxSOOQZTN0o
t3mBrKhnk4XOGhiI+031WBKsDhrSwnxY63No/spDTrpEqjCSCwdqlXE0A0URH1NIVD3elfZ8IqDE/UDKkjWG+bZMy2yRRYo+f6wSMxQEEG1rNM
CgKp0PxML50i1iZurjditPmlmwanHK6kExdEYDu9maGoiDtcpwhpP5fFIrsWBEZIAImNTGiFIYIxk2QxH+UiB5LMNQS5bRV+E6q7WdNYkhobSv
5BMlcn5z1OkSeM0lbOLaZFfLOTLs0irsXqfW6VTBKcBvZ9iDOH/Ek6rbbUAwKQEwNU1lEqkSHVRBWO3Iqpqx4DAJj1rCAaIgqghoB9XEYWg4zN
Hw/i/EjO4QnfhStqaEphNXFN2hQDlkL8KhzXFKtYQsZY2U2iOTBkRndzpSEptkEOBxJSrCLE7eZOxPRB2j1EglO2XRNNKFMUi49A4HIhNMasQW
vPpg0YGYFLyM1btEA7k3byGpHrNBlbGgOkYrGXs4ecMQTea5TNoxA2RYK/ISgA8Y7kEDIDcI3PGSP5TxXd02OCMKxyJUR1YIK4JlVTTXJSpteh
Ocxps9E9JKi7oBHCMxFJMBURSHKH1umkpgCRhrUXS6qCNrJhZGpA2u26oMKhjJaNiooG3jjeEwZkwvfqEh9enBUQ40lexsnRrXDHCFbMmUD2pm
MEgfzwUzNU/WGgsGBhfSLcrGMHsoEpJJguksxzZQ000FKqCp2Uo30K+HloVFQzzwdlscEzDIPgXl5UW3WRHz4DSYTbOMjWpnjOQmSFes0NSgEA
M3ztpjfFHEMmMcPyZheud8mmTDEu5WraGQdylzdKzrz9Z0405nybJi86lSxJYFGCjNgIuyyumsXC5HCgztFC+KChnFkCWSCjs9QRcyd3moiCri
yAEkO4RzBY9Z0DZz2HsjeX8hmIu7I3AsniFIr1XkNFo/VFCHlMrgEnbntCJxH2IPKhThQsBAIOWiqBC2VHU1VVokTAYUxZA8Voow1VACAHznjI
ZyNKTwQjhScJVzhd3QTBxA016dxVmb1GYKBlRMxuWt4QBE2bMROms0uaJxW4A00m1aUwcocQQUNfWL0dJhK4B9IZ1yJkuojaCwI1tQJKoWqhTB
3DLa0c4YMk65jcJ9oMWDJOSJNCZZtKVDSUncg0s4AsEj4QBNMatZXkvHFw5oZFEHxWucJE2t5lhN+c5k0qu7AD0s5+kfZQ1EomQs7886KLnbW3
Q7WWGa9zMVY0atVZMDV93KF/Irit68Q56L2LU8oxsTzDHhki6QYH0MoQsJLZQHoOprjZip5NSWBKpYEGKKEatMRxd4+N1iFWsPG1Lm2qK4ZC/A
zFWJMWnn0DrIw+edcoNAHjVlpUp8LKxotdxGOw+ghmmI6jb565B0rrjUJVnqIrctIurY4kG9yalKhB3QwBGp5Esa1hyycyRQ5+ZyioZJS0MSVm
UwksO7wxou4yyy4mpJ3hYINb2RiFhplmeW0gxwKrXkVqHOaeYMChnEUtlOF+L2VD5Sd8YTAr3VHY3SKWLG+UMVf8xMLG/mnoaepRZ1gHaTJ4uT
OK2KoJHXjgd4nElnn44mqJXZ709k2jGA64RMFj0BdohMVmLKhwGRjsj1CruZKbHSK0/QzWmTtuBB7UhBjffwYm8uwqtYfkgj00jcajM5NtZ1Qm
UbbEjz3ClG94jBCtNKEZ5wEYGiBDQ5vTlbjkCApMsACKiGq9TS8/N+ccasaNrKrhAoUCYN7lKUpKfwag+ul8ZaZj0nT4K0sZirnU95CrynRSVF
sYSlFHAJgWWx2XHJqTggMhwOacMxsJSWJ2CfmysvOnCTmMA+UAkhcXJ4Wl5NqRQCM+gc94WiEYlVBeVYmC3pc5OUrm4waoECTmcAswEQ6AFiko
id7ieJLEG9rWzJzWI+20GY5aIdGM/BNAuMsgRyrOav8MeT5awoU0yL9Qqq62FkbhtJLGk4YKRjUI3YFUz7s3VZt0xh0MTPOUOBPC9uy1E2Rdto
ifkdAYUC66kEPjBn1So4OckuGikqDaRoL6gjIMHMDxrVNlRhGUjFqAbKMoc8BWcyplHrJXFTkCEAbK1emasMCcSS1UqbEiTOvSynj6tZqlKvOq
XloTHS7lMyo084pKY8Or/uu+pqhVCCcNvDkWUiZI5zwOBGLm3wC1yqtIT3UBl/SMAsakSMiKQC/qQ0A7kycb2lx4R8JYHlAZYhaqtbycqWdrRN
40+bSGG+bI7ZLUQsHyea9YJkyKeOqrlEiRESKESzGj5GtZxhZ8NegXkCjeQlP0O3lGaoZC5XyhFESmWKRYvRgimwKnEiBk5lyAxSysT0JLUNT/
vMDrE9BVNwD1yIlWCKhqUVBhqPPxg5tHOU2+M2F8CEaRX0q6MF3F0xySZUpfFgxKl3gKWkgQ076mi1M0Qa00L/M/a8yEoNbsZOyCBQ66oyrDfh
DoUEGJV2/kf/BV+IyGJo0IH5/OF0ThFjrL5KQtAbEouEWMBGm9IOQ6GkkT08chYZ1gdpattC7DMUvTqcJlBIE2q1zuF0Y83WmE6TEtg4FKVR8j
IZj1RuEStqoCvyqnPFCZWjf8nOiO1VAKT3DTIT2FGRUAZ9suCN5QLzbNrpL2QNKgqIZIIGb1gkDsRofzZA+vVSkdAXn8fsp19/EfNsCU//UVfd
2vvWxW/FeF3Lqyo78/qjl29XLGrOo83sGR+8B9uNPW6okJOs/myZez/e0p9tHmzJiIxnN49+CXvv3+WMfod7OP91+f9L/duNP5XWK4flR/+qoN
nyXew+9G/eP7r6cfc6wDuth1pGKzUThgFnxzIhvRSuD7y3vnn5y2zr9Oj3NP96Wfy3byQ6L48uBd9ebA6Sn8wUjw7gF+cd5fAXO6XURWnld50w
CQ0930rwZoOxLrSf+D/IZ54fdjLrk7/9DcRMSee70R/bF0kmA3rJvbijYJ/pyj/C7mLaf3OfkzPUv1PN/lvNRnT7g3xzFIiehX7TXmS/9eR9tK
XdMez9P4f8wpEyegcHYI+A7CY9jvrPIwe61b/05N5DoDdAmn+3Er6G/nj7b+Dgl0eY5eLep851FvVZ4jrdW77qETN+S/eL2q/LH+nfmD3lo7nn
Zt+3wtbbMPsPzkN3X73++usmWLAe1oa0+tbyuF4VDTuScfcTqkfuvbflv/zhIBb4z64zoflmxTDY+xRS3yuK/Y1j6vZBALkMk5vtv3SXnkXa0s
89rFHmKrJSZ4E+0Cl6uP9nJP3tT9eh19uHb33I9dcI5L8M2+R1gyzJMf7caumc45m/iq4RzFmj9cycerVZ+w35ds35caI4Svt+Ufy2Cc3I/tPH
gqhRIoIfzQy1E1fmtZ+/IfOKubqXzw+v3nt7rcrdp0n3Dymv97nDLOVROFva+rNF4Wdf03YunD+zrNGIP8Mi0f3Ra7/NHr8uiv3XHe6p3ER8m5
0hVhp//hs+w+12/b1fhLSqcDi/fVMfwqnmgM98PgiPhaOMbkmLu19Z3r1nwj7+4pPhf/kT05PJIEx3d1++TEg28Vul8y8Zjqs/K7SeMM3x/7gD
/1X4oeMda3dC+EnPerngBVNyZiX4gwvj9g/Fjdu5D7aeYX62cRDKkgYzC60R7ryg2iKnc+iLmCkvuuI3atn+ih/8hxt6/6+l0XQt/86fHu94kh
1Z5t1O2KAwp6JDQTGwcmM+uSPV8WnaxtPKh5EfmfTGuqZ9/bB+f5JKhQc/q1R2JYdnmc8+z9nPXvvV+iqV9Kjx55c1xd1v03n/de2/o0zwf5MY
en/QbQuWjP/yL4KARdr86z+uePTJbOIbiyJtl53muLSERHyqY9dXpdRnkjX7KzvGC8hXLUG/p/tybfwr8Y6g9XxAM9Zo5fJzpT39LHqyy62gf7
zV+M+nEfGVCiQ+e5bzBpat8xzps72zE5pD9xPJwa3+R68/Y/4AmpH+fUSL9tbkJUT2uUuQeNap/CDjBukdC/dcGb4f+PgfP3CfruzXsucXvofJ
X3fMsobTH8/r63dF5cMJfXC2tr/m15esbNjbjF7+6OkvvRLDQGu76RC/wxImntj+PAlXnX8ZKX/2NLmzx/mZ1zsX/Org6QutyvgNgfHfvL5f21
h8KTwZP5TDxLHt/X9yJbehrd960vU0bspl76hOP6jEqVzt5ceFvpq7vF+hngtfsW7fdu8/e5iJ7/1H87kF8dH1y/ceF2QMWBep/YXW9fgg9v4h
3oR+bm1fHX+y8mhdzAgfJ3X7IlJ2Oc1+Hfn3RsvWn0LFjepH92/Y1zou43TW5++vG7o0VdOmrJ/guf+QV1RVh2bZ55yS4d3wP1hjPXP6qsVGPH
b87DvzLrKWvKHn7m/qc1cU2uxCFcmcbUtWOHwRfA59/P7DB9htWKtK3izuUu3ZYuuTesH3Lz8mln2BzGZb/Dfr3xPyvhh9lVfo7HcSohvpbsaw
fF9IPv+3nOWZ3fvxPwMHPMH+T0IZKS1V863PLuV5caGrvhX9JEDUUiELTbAn2PZaX9uGz1HwTwL/b8VRDs4YNbefJmsOeUUyyJaN5t2PU4VPB6
XCX2jhz24uv/P5JtUU+eTEE5wSQ88k1IvDvwK8m/8Gv7SnP/X2lmGWurXLUlSjO7pxgsa4ZS7CJPyPw9/LGmLmaUCQYgdfvvUdedgZWl74Y813
X7xhnIH4wc8ppHtO3nFOpZoe3WqH35eOzB18o3NQ9D09XV/5hzUTcTbEbHyXMso+b6Tx1z5sEHajuXv71md311Hap9e/DFtk3u81lA3j2Y0AOV
biyF8blv8HwXz4z/t9nvX6T/wNVTME982Ta+ZzS4ogEV8/ekJap7f1etWp+NaK5Tt3D7/4ZcAf+H+IPmT4xDOT7qqYsRKXr6krWavl7OvQ7JSW
OH/uqv3kMvLKX+ysTD+5iryOSjhfHIh4h74fPFAz/gv5haV7KK9R99ak4jSraLxao/bGrOZMVsqxC784/zu1BymNPcJ5l7Af3vw1IHz3UnzsPx
j/8tmb1FnJkf1YDRpW7yQn81Ryl1NB9m47VUa+u//foydfH9nS2rxlRJzULRPR8prxq860tPPB7ujcnxSd6OF1TR34oHNZ2Zw6NsmK69u5ltaV
Jk8q/5e6X/Ts/Pm/8k2c37r6JNekECI9yEFHKD4xkTN1DAov5yJu+jX+8ujONwDPq68EH/pbR5f/iPCYeNW6A9jPYhG9gG9UHkY14vTja9hveD
oMyEvn0ZvNL60KfiHQ9aSV9A9g7mdH1R+t29T/aYqayZ+vNld9SrX2OeGk+fE3J7hA3VqYCX1G5e3UP8yLkQ1JVvXz6+Qlb8f4cr1NPrqK9t49
/Ab/2dWbowD48sFbtwL/C8ZIXFn3bkGr3Fs7olfIjfR+td/TWTPr7/gO27d7kWlNeXoaPm++R9xvnuDE52889tjEXwV8b4eIKt7rKvIPRscvfm
tnj/2vsj9UZ/EPPNF/uqd//cODl799sPm9WvL36edTVYnqn1l2+LHMUkFvbfxq+Ix9e936uvPRf64M9mJ/6PrFevt3zYbO4V+1gNs+C5za5Yuq
Md9FWmC++ESOnR5dd8p35iSwvLvhn0+hxFz71Pc4znjtcyu6jb/33X/HK45qjvBoQzQ4oebC899dW6N/Xk2/PDTOmSm5KfbK883K+f/8b5x/az
rIpQ5KoePxbevL5Bby4/rB37NsH7xN63Q/Ntdunazf/T7ZqqI9+aQRKQo0W9f3pr8cttZjlFmUc/MxfCr6qnb8X33Y66eyHzlMSsL+9euHGJn0
9z28P8A/67+p/Fr+xr/5GqDXE47Pfyco2oaJx68G7ho/2b1aKy+Nny5xiXq7W02zs1OyJxV7Yt3RrD9Y+6M/COPlo/9s+TPxvMnfEFz9dlv2nZ
o8AudaxpMP4KeHkiePT4Dyk+t99yW9om5cPt8sKqiNMqn84dFX5DXvB5LznebjV0L6L9eq6K/SIjYTkKdStudbkvlpHfpattafZdSa0kp2u7n3
LxiOY97Xwch+/PNTypt3Ly50nz7pxHeUhswh4W5hV5t/84tNAiAVrOAf957l/+S+gnFjZH9RabyCPKtD/b+4ejj+5vvXF42/fdcUfu0v7m2RGX
995Hunx7H9MyX4e4qLP1mKjo223ZQwytH2PjeHlXfQyKd/HpBqLhqP76ZGTHjSHlje6nXj0OuzlnWpu7a4ZRtrK9n/kAymgv/XVlzBfffeDvNn
+58S16KCJ3CkSnz2w9rBIwZ+/am9/TfHQcN7Khbl0xRorDXp4ZtJ8wx5LiAbv+xWXym/f+9GGOLW+c9xvav3OreBjfwmw+B0mPkLZFbjPuSYM1
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
Wv88O92s73f28y/3j8T+79X0b/y+bf3nwWcz/LNk7e/m/P2uM3/uFJ9Bfd8etu1nrth9P3fn6T8J9uOLWPvjbr3d34J+9MvrNzcPRs/yePwrsc
zr9TbURvbjY7P/6l/8feQP8k+BeNyo+fzflJz/vJo59Of/TzN/77wQ//wzz4D74V7Nz6xPnbP58m/yYf/egv6w8O21njn/3o60//1U/Z6s+C+Y
N7Pw1/tpENDqIt59fbj34wfP2j7//qrcHT/2Q0zA7vvn60Nfm92jD5w+p/6H7qzFq9+Ch/83h4MuqeXaue/9Z53d/9mnf4z/7RUfN6fFodDpxP
d/5B3ctHDvPD30qfPTsIg8lP/ot/vfOzWu0bf3H0H9z984O3Pq39y/Zvtc4C5/Ub5z+fnDd2svPTytHGPfF0sbjtDmbjVuXHs1nta2nveMOp/x
f/ae/sptO5r708uj2ITrwoc/LGvc5m5xs3T7b/T8MknC3mf3aa3Jp8cj96dBxO3Eo/3966/Sfd27VcnUp9K39cSc9u7zyb/1lnfH26+Npfdc7j
W+/8kevF09BbDJof3fu/xrPT7XCk/1fzXjffd/vBofO74c8GlbG/fzC58Ub7077fPGp/o/vrujvefDr+bu3HtdF77VF+bTfcrO6O08H5L6Pznd
fi07/qTZ3jjYUbVjea4Vb34Effcf2xn+btjXD/33vv+D+qz58NP53+r278svHD7eb25p8uJu9+ci0cd9OnnV/8qnrc8M5+7Ps/v5786vjDYTvu
Hz+Ng39Ri87vjpOHb8zj/+7BOH99vnCrXito1fTblej4h+e08yw9rLj76Z/8+OHpWTb2s/Nb/o033u9/N3JSZ9p549HZtDK+u13bPDnypovhP+
h9c7y9Od9L/Z36M8/ff3h2M3t/f7YT3xn9Orkf+PV6r3k4OT/44NnGL3dv+0n+reHwzlnr5OvtHvn15LaGi9e3G9PB9iI/Pm5cr90mOn747Hzv
t5uz6MPIbQUbg5NgK7zWbQ3GtePEe+1xdfrGd3uj9NHgfzedRufNeS05bvlpY+PN8/r78jZbjz+eD7aHD7YGrXvPFreS5o3wz9Oe7/u9jV9O+d
f96eJufHa00zxL4n9+8I7nfOQcvHOns7P4TtZvPA7Jc5xPO6Ob7f/8B+eHk+vB9OE74cHr4w8O66PK+4Te9oPuX8c3/2Nl7Y+cStLa3PjVG7Pm
W0ffPnr/5ne9YPawsdGJ/cXTt8LbOz+OvvkwaDn1xXFv+/0ng9d7r+3sTG/8Z8mpP2rPK+3qL93R/G485FpnmHtbM+/uKN3rXHfS2eyHv67ffc
9PX98+v/PHO4Pu3eAsfBTGaT+rxelp/Oe335id333w9v++/8T5m8B5WP+D7Uny0R0vc598GCyycBEOXx9lg378RyfuTrVRc6sjxte8o9piM/jV
prvYCo6n1RvRa/H/4b1e48HmjWb4jWZw8p1g9G/ntWAcDg/eahJVnx1WZ387m/TnT/Nkf3oyb+0MDoep3nqw2GwPksrNjcXu3vbeR4tOxfFOJ7
8bPGl9I3LCWavL2eI4bcdqLnrhgVu/v7WArQ96T3dOvnv9ZGdy7bTaPs9+1WlU6Tzyq/M4m/1W6LfD0fjwOGq095N4c3T7unv3PO+f+qc7v9oK
x/9Q0emJkqzhVKqNp4vga3ecxp1/dfz14NPgT/L5bnr7+m5+Y8utBnjd5vUsmj3dfPSBf1j90a+fhNr+UW/j/Ecfdt3hR69v8njx5LUmEVvbk3
3/x/PHz84yv1Z5s7kRf+8ng+zDuRfN4+DGzmDq3/uDej75/rh3OhvVWw/n1d5G5YEbJ/Xjmfd0m43dOzV3mv7uZN7YdIIHH/82x+/tfxpV38kq
/Lg3PMySO60ptV9P3NOmf/de9Xw+bW4737qefrQXhv7jysNGrT7p/cmz6uztKB4ek288+f4wjVphkAT48+Nwu3ni/63fbzxNRseV/E7/0Bkl7u
NquvHr81/3r3mv1x/vtZ79USWM+HCj374dPE2TLGnqr5Mawehe8+l3k/OtXz0ajjqLvKHT/6ZVCwabb2wHNx1vtNOeT7YW/pvBX0x6H977M298
b6tVSX7r14vmX/qLSPmz/ce9evVx6/HsycGt4U/+641E++9nj53/6H4WnwVPSJ+devlr1/u/qt24E0bMxj/InuycncSdH4XhH5640/f+7PDhYd
aMH0h/Mhz0t4JkXD0Ynl/bb/2Hd3q9/8X+9mJSezf7NGvf/+dHmoTO2O3nr+3V80pUbW+/GcRt3pl3Xn/6w9HYGWmzc1LtjG7l71U/nQXn3dQ9
e+3kvO7HD37YuV/7ixOv6vDLUTsYxN73ul7+9AdenncW3W/+4I9nP2teV3TWPDs5+pBgF8/769fJhm0qO+Gdc985uLaY9X/16JPz08MncX1j9m
blJ7dOnk0WT+Pxz8f33n5ysH+UfDpftP/t4U500ukG8ex72XFtWH/04JtPqv3TN//P93YObv6Ptiv3Omd/lm0Fepwe/EN/9q//5Pa37u+fdKf3
08r/Z/Yv/tsD75Pqu3lv59H/pBf9x+O3vvn2v/mW+2ZFjadh3HQ++ePj/fhH36zX7jqta08ffDP0rzVb6SCcP+tfn373F3feuzbe/FrzcbD53/
Xif/fTT4cb+fQ7b433/qv7N2fnp4+8P4/+7MknaT77l/d352Elcqb3ZsMZzvmTf3e/+c++vvdacmP0NHkSBkmz8t4nX/MrnXZW+fDcS0adrz3w
jzbPvln5fxyd7Gy+vmicNDf9b+/o5nnLz+W/88uoWjmr9M7cn54lHzenndm0stVNa1XH3ckHw8fT3r/sfvyH3e6z2uY/qXyr1sr712/+3sF11b
6eOOEDr/Lj/6wzVv1g8Y+Td2Yj7/Dj+t3m79zy+q29nbt+kHz3Trf1qPHhzl54cziovN/rRd7s95L4Trrtb1abuxtO9HNvltTohDfH+nS4Ezbi
1l9W66d34/bd1N/45XD6aTBJdg/fPH/zH4xnk+T8/qa3f7/m3nB+ee+jftxMg0+9J8nHk8nD9Fff+vCd8Phh+t612sattHe87ST/prMYzp+MD7
eeffiNdPsw2s6ftBrNoF3f+l70qPp/CxN/2gie/duvfaPbfYfzXqOT3exthZVvP5rcdkdb77Y3Gp1vT4P7rXllcbZzkgZV9930uJ06W8EfOPEs
O/xW9LD37dHwNMU5/S+yT2bzBOeNu/J+eW/Yqby+29259lb3aS2tfT/+dHqfxfDh/ea10fhrrb+tMZ0njf3o9ZPm6NSfbX7je43zd7g36rr16d
OoNZlUTxW02scV9bfD1w7av//T8esHNe+nw2fDvJr90/rOTpwND8+7Inzo9vZ6fvPuoHf8lxO/P6odb8x67ez8F63c682OFjsnR6eNc4L8U/4m
uNf/cav/Vi2sf8M73X8W9I7GW9X2J63w+3/Wu9er+9SbH77h35x+kDUzfbxIspvpDx6EJzsbLY4fP9rcnN+63pxO5t7tJxv+N92Eln/WeCN789
mb/mbl/x6d90bRa0fev5u5Wz9d3Fz8zrWjefL1wcPGKJuMnnQf+ov6xtHZoBkrfnRb4/xJfTuqVrLx4N3peDutVic062dJP0ir2/lwa5JW36pp
u99M/mcbD1v++8OvbfjOKFA1+mbjO5Unr9dPw/eP++3a7ECLuydvT3v1fzL80zcrWW3n3cYiOw27hx9Mo6OnbuX1Wfe0GS++1j+ZfFJ1fnK8Rb
vVrHy7FX97jNN1k/bdxTh89Nezbwx32xuzCpuVu/lHzWa08/NDP9yufvg0bb7VDHT+0flO6+v506/PKtfnv3uabe6NwuRWpbG/mMaNt7Lf3/x/
57cWeNfcbH522j2Nb2/2r+23an+Q+mc/qYweZ6dMvz9IO14QdW7GG5Hme59Ud2/Os9QZzn/u396qxqPK9s6vK+5mI82yBx9N0+H1vRuvhVuNrW
vd1u8H+uYfvFs9yRaNe9luIwna4REHh4O4FoTc8c9H2/X+8cnitY10/EbM2wq84yOv0/r51Pc2ve40C+da/OzsNJ9cm7rRzXYWf3fa7z9s+1Uv
Sq6dk210Nt71uxXnZCOq1vacMG3o2l8dZUnkn25u7fz+frKRhvNqNo/CmjebHnx9/mntTiV22w0ny0+j+l5t+3rgzn7mBSeTZKubTT/q7H50Yy
fyH94e1ZvvVA/rXtq/9f3HUWW8P9Dd9MYt94Pv9+L29kFadUZnXW9zEQRu3Nl/0Bo165kXTSJ/+OwtxcGs1ruVxtH0VzttDTbGeR7q/qaXsOl8
PXz69iSqu+7BpNYNOsPbfzbs7GzU/6rdPb7/TuD3f7TZaiW9I3+z3xvjhR/473/azLcWm636bzeT1p33z/v3O2HKaHg7yjdH5+k192Q2PH+0cD
fr89NGY+BNPa9Zi3tENaf1ve29435W2Zq3a36gKHt4UIsOotpf1aL46e5gUTn5n3+z/vbgB3E8uDZsHI+ip37WqrTr0w/unGfjw8pR66y/7/7j
J4MqG+47jyb5fv/Xp6fh5vDTav0xN7Ozbxx7jz5J8u/+JAxrQdh+crgYB2ctp+pG4wd+7ej1yt9E0dOn4fWqcxB+UKmrVfNvdI5HfxHh16b395
7+68pZ89EnTNPBvTxFk6E3OejV4u3Bo/Calw91392e3zr48fCxf2exqNRqlVuD4emfOv7GfP7sev1QabDpDyaDmwP+Wdx5rTZK3W7Nr+E5W/c0
nQTXbv5+Y3S0+budftsdV2Nnfu+dSvCzhL+I+t6wcnbW/etvV5IfOf9HPx9EUcc5b8Xjxq7z2t541GnWovfrWeaHx4/+nx/+2hm0Zm8za9T9B6
NusF39eVqbeZNq/Rvj0ay+teh8tMNrtd7Nv/noF4+SUdu7maZbT5J08KH7JEV/Gaf++M4/Hje3Fu2GepW2918PfC+89+h3JofOYOP7z/qzjcU7
o3zwX032/ek7h1ncTgf+16tBV7O0krd4cnJy//A8/mbdC64no1lYoxqMH2XV+aBNp5LM9ofubLDw0uzWdHi6P/wls9ngk/h9neaL6M9G93b+k7
NrTs15uh9Ns7Nq4xfh4uPBu4OHnYeanETJQ4d/eeR6rfnAe/Pe9Yeq7u5vHf3e1K/NzysVp61v+H/47ta1jYf5YPDE+/bg4Y1Bfbfxq3Qx3vGc
uLb/yc4oqvifRNPrlU+a90/HT5obtTTNu7c+Gt733cZBdB785THO4Pe7H+x+Mhn6jEZO/2g7O3L/zXF6P989mbTCxz8Y7nW8avp097t+7SOvWf
mbGVPNo2f+8Wb1H8S/9HqP374RxINWrfat6egGdd/1EjfoutXj8GHHCQ6U12r71wIFW4F/ks9aqbv/aFHxe04cPn2U3Hhwsruo/HKy87dzvRGO
W/7kQT08rYWV0aF/Gv5R+PbGNHL7m9c6N37nPLhd+frr42u6Xemd9OP7b/b8p/1G6xf3ssHU3U4Oqk4rP8m2Gm5eudeZz73/1PNff3J98GyzFn
XP/s3Wgw89FqOWjuu1OH3fmWfdk8NvnCscu5P3vnN94Vez33JaR/3+tN+tf/Tx45N7pxuP6wf1reBs4rSCm97jneTpr2873ocbB+PK01FXJ903
T1+73qg2z25m7uutbPMO9/fGzi+7QaN+87j9tScTfzzcefqp83TyRnURH+7uzp9NG63ON3b7j9Jw848DzWaL3W/fbO7FyVZ1+lG247aH6Xijet
hO4r2Pq7+49hcjv3oydBdj7389mR9EwV6orUcbeXL/6cN2XMmPg2ezs/jRs6+NDs+/lvV/e+65/7m/m/5OWvcGPwuHtzejjYOzgwfk7Wra3Z6M
Z73R5LVk9r3ajMUb73rfOQx36vXRH1emMb81PmiGe9FieHo2Oj/aH4dDr8vmp9Hpe9KgNhnXN+tvb6SHbnPR+fjJ0SI+8ft+36923N23CD6a1a
Zv1LY3v+EfvKYn7z+JmsnemdP8w4eT+uC1+cyZnm43KjV/txHW/5Zqc7+fftxyK1vd9MmDyT7T69Uqi+Pe74+/Xv2uKvPr8TwJTu486Uzu/n7u
nRw9je4ezX9Xr29+6t6Ya/OktZMeTLpxXM1/EjyO3PjsZPLr/O5u9ix68svBp+n29nZWy/vP0mdxHP7220719GlrY2/QbE0fVCcd7u7ne/18/9
p3Ptw++7B2sxOfHA7D6mQ73mo+vhM4zeHsOJ7vtV237zj77c7Bk2tvtJtBK6nOFr8I3XdGTmXr8DTZ9+jt+P6O/K1f8MbHo9zbbmyl38yi741O
ZoRJ41p0vvnT02wy8dNef6abzn44I3CST715UA0/zKvO/6bdC+MnT9584x8m3Uy+Rv/+ZLOxn4VepdKtzfJblf3G//Tpmbt7UIlqGo/OOv7H4c
6s1+TsrdtuI58m5+N3Qx79lZPFyeTGaZ4Nt4N3bvyour3RTjP/7Bvnw2Sr7t6KazeeVN3qdmvj/TQadwa/s/lGLWm4r4cPt752yt3tB7tpWI/G
r22E1aDuTbNnk0Gju9hf1Mgb22n8qH3nxlYrbKfTyS+H3ti7lW3Gg5CW05101BoPHs82/OjgjGmlEWV3Exa9yN2LovRWuLjruWE3mKWjQTup1j
fPn5HQ+OmJF8Vxt75Rv7VVaW3Ri5Ltarfq+EFyI2rUNjbjRZ7GNXeoSquVvz6Pzr3UHw1aQZC99aNmu3e3Vqu8u3uUur20kjz0m4snvUovD73d
RqVz2/352bOglfay+mwy9qet881569PO9Y8rUaXR9qMkVHb/m2I+1fW0dTrr3ttot39ZHeVRErhtJ95rMa2MPz0JasoPmqenWaffedzf87bC4X
6Ne39b8eY/r7e27pyeuLvnB2eJO7vN4Qcj7/WzMKl9s5HWr30w+qP3xmGI299th7eTqNUIHzMfH4zc682JWo1e+9CbdfyFE2zvbEevJ6+d96NK
3ZmTzbvB7em7lcajVH+REDzZOZuk7/zDSB8NPqjG/Z1xeOw6H9N00mZ99N/vnlaG40oYv3/mhH/5dFQ9+Xr06PyZuz18eHwYb+TzVn2wqGaPdj
6s3b8zaFZ+FsenrbRZmY3PBr0tMs/Nn2XD81H7T/zqVj/ZinIcxbt5p+PP20ePPva3ZsGjYToZpX+VvXvcnGxwsJHMBo1p8OSDjWFeV9e/0Vuc
PnxK15l8dLyYDjc206/Xa9WN/nn3rOJPPtbj5Pphz3/QJDiJu27l4Wg3yxtxdtQa/qlS3HlaqyvefbOS991r+0Mv2cyCOv15xnmSf3QS6FGjp8
Wv/73/NPhz58T37mfD3qxSj2ZxrVfbHvyj1tztfj878564cfJuf9HI72r47o9GN3zPCTpbJ+9Vqlmr/bPBW09rqJ9fP+w44bm7ezD78JP8wW41
bQw5ruUP73XfyRP/UVb3Dm+Px9VgurPlHFafdP5bJwic+YNbeFun/vmocb3Fk0HiPmv57uiH5/NB230891rV3htHqoy6D36cR6N+2/9aZXvU7h
xNH83d1/o6v+7fu5m/Hw3zg+39eHq0aH+95880UDR23KSxmL9zPgr96WS48H/32bD2Kz+uZefxcbpQ6H3y4Fnc3Rs4taN4p/tx6wdeOuidP/U2
Rm2/Im9j869/vjGoeKfpwXw2HQ7f/iCJk0bv4WRU/e3FpPHx1kbspZOjLHXHyU47246i7Q82nfGuPn37sZLe27vjrN7fPGvVusfdD7qTxafxeN
7bfNDdHudBRjYIrveyNNLfzEfaaH+N1q1o9r7zwywIr3u3nehhrfHRj8bbqmRRMO01G/2b7e3t6setdujWq9G1c2entufWwuSk2x9U3o7TT5Kd
0cJfRLvMZ61GNDn2Hf+tzuZ+4L62CDaUT93qTNeTSK+HR+3Ftcb82/N4UfXVH1Zarch1j73Krw6jlrv9tVrl+knt7kH7NW2Oa1PqsfJrxyGTdC
edD4LBYDt69lpE9s9ib/OOk8ZukHa0OPK8s2z27SQJT/+6HXWbjen4O5GT5PXgve1xtf7U6+zHNz+ouw9+e/eH06C6RxSMp8NuOFv8fLqTV7Zu
b83m2/3d8CSvx7PrbS9ttlrjZ09fa6aD0XQwzNygdr2yl39UOQ2S9qx67TS+Xr3mjMOG03q02LgfO8ejzVuP/NO0E06caxs3k/N8kHbeaLw2b2
SzhwdqzCbJv6+2kyxuJ8828utJ+JqbN6K8ftqojyfxybO53z8b+43zvd+5uTfrZ5sne83jBnqWNDLyyidB/bzZ5CB/NpvfiftN8qf/NmwQBbXB
uZL5znTSfBqFD3fjTvPooD09vhM20k7F7Rx71zfbj7Y++H6rdb2dshV009o/uh7eval5Hk1Oxo2q8yxt1tvVnWQn2P7OrB0fV6KoF3vxk9fixS
fn7Yqv2WzmVheee6Nf1x+OW/Przai2T+3rXHebUd1vzufbj6s3a/n35n0ibyeeT5PtSvL0ZzGblc6tWfP67rOB22iNO/9B5fVppGCq7LRW2a57
lVagmjuruZ1479nCPzoN406ydTOu3XgUvDYIU7mLCmk4l9P4OGx06zdHs297c3cn7sd82PpV1HD3w1r1xE3kxO78ZBI/HTbb1wfjzB9em4ezrX
a7MYh0fdobbD7wmzcjP9pNK55/uBc/brr57nH1NAmjXSXzqru72zz/ZPdmq3UaVutONvS8j3Ydv3qeTZpxPtsIq42URlqrPXq3k9+oe7Ubyeud
he/sRHFUCSf13th74FeycxIvi0fupFpP9LNFPx5He0Hr93eSs6R+eJ68Pl+Mfdqj8JrfqNSaroMfVreuOY280mnP3Xi679Z3o1q2/+YknA/9RS
0YXYsy13O73jvRzixKR3G1FwZkB0337eCDoJ0Oosyf7MazQXt38PXbt45Pq9nd2ka7Wq3n155uOd/wRpW7e4N4e9JVte3nwazjVPfq7fBJ0x1u
qn+Ux3HVS5/Jb7crg3gvI5lVQ2ofb3ne0yCs1fph6HvzUdtrpUz8PKscjH2vXksr++nseq/u3AqdKJtMGh6tfnNa6S0c349qj6dNt+1+4I/zxV
Yt3fvWdu28er13cryX3Ol7Cv3NxN1N02rkDfxJQDII3FqQjt18kYXpZtTUrWujJO+6o9rgpDVtRkTT/MM0yivzO0749Zk7SLO5d/7s1K0l07Ti
5JNkntB69j8+3Jw/jeJ50pz5FdfN7+9MNybOYufOcBzN/xvXy9zwblyNVa/WQs3co2iwGEZhPBu3aov6xnF8r3v9rDa7346c+qjtTd2GttuLTW
9zwsngfHecPRuMmjtTTcPrtbC27XS7Qy9wvYlf3+6H7VljkvQ1cL+Z0xz2fDdKe2fptL2912bzrdn09f39s/Hh4fG+247cMEjCauRVoun5wegj
5/Zg4p99q8lidu4nIenMeWPn/SjOUqeTfLA3cE5Pk2h8PKl7zXGeebcT9bre1xYfz3O/Wes2K8lgozrufNL0b5/NvcOwcnSzMtvN0xPvtJZfT/
Jprb55Npj+YO6+1sr28zzzw0riBrXoz+rjydzdaNLNavNfjh/OnvVafmPwUbvmeGkf9wfdw9yZH7nbff988STbdQfvLubV5KkXViud2VbW7d3z
gnk+mgW1ZySJ+2Bw0nDntd58ywtavjc492qTmrtXjbLv9JP69e3z8XDzjdnuZtNT1o/6SdSP+Pi9808/6Dwk7SfBMHfvhdFRezHo7V376IGiH8
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
+mJkysxMoBMmFHRrKZ8qhQOl1IpaLBeLo2prBY/bO1xqfjSwaz8SlTp02J5qOZfFe8nI9nPorMM2VKopKsdBRibaNXKJej8fknFdsr0dzkruml
WCxSyJXbyvl4e27KjM4ptC+fSUZjqVK81JaYPqWYzsyTqoYi8Vg8V0mmovOMyDEpUg0n25O56IxiITMyXI0Vv8kkv8rF543n8l++V4pMuX5EIV
WK5wq1ZMJKW0clEil/Mb2aTydDsRjRXCSY7yoFq9FMvVRpK4Uz6WQqvlxmarw0LdLZkdw2FI6WislYrjy6kswmw9FgW8p8R2epsNBC0VKiFu2o
BDvK7V3xqeZyyVy2Ui11hCYlPoqMff2H2dL/qqMq884zap5SsphNpQqRZLFz1er0UfMUU7n4iFSpY+J70Ui5PHmBVEeolJ4vM0+0UiinUuHOSL
IwIzVmxKj5otM/z+ZLheVDuURHvn10Mjdjaj1cq1aLxUQ4ny9NnZ5i/mKgmptWyJRGxlKdk0qRUnyBjuLiqdI3xXRigZWi5TG7dCQL8WIyGcoX
AsGO9hDl2DLzhFPZSj1XCBfS5WCyLVoKlTpisXBnslLsKhRLn3UuUIyHC6lRxTfrlWoq2pVtS2RDnZlssVjq7BqZyoxIpOkqJCqVwohYPUpXqt
wezqSyI9PhcKKtPCKXMb/MFyPT80WLyXJXriNRiCazXclqNj7i/Ujb1GyimC7mc6n0PKMqHZH4vB0dI0upzm+mZwvJVGx6rtReyMdHZCudM4L1
zCJt6XiiUklVUumOQj4dj8xXCcfDqVS+Gk/FYulsdUq8EksHM4V4MTsjHY9W410TR7bTlUkmuwod2Uqi+uzIUqaS6MrlMql6ccSMYiRR+Koj39
UWjY5K1mLZRK1rWle4mG4rxIpt+WgqkkqnFs4UEsVKIjcttn44HioU25Jd+WS5rSudiEfbMrlcV0c+FxmRLETDo6ZVYmVHZGPTK9lcZkZnpVKY
EZ6enhaLfbxsR25ycUQ5mU6355OlQjqdD6cK01av5EaliiOzmVQ63zlleqS9WJo8ctS0aK599DyZeCFbHJUOdSSThenpSNu8kWjHN51dhfwG4c
5INptJprP5ciFSK1ZyuUS4mCsWOtOhebJWu6rZZCGZGdWVLYWLmbFTs+FMVy6fiafbIsXwj6cnOuP5TCrVlQtHpo+KUUgsPjLV3kFtQjZazOSi
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
14+7n7nXjOVysfOWmL0ib/PSU//r1bays8vGXxu9Lv+rvS22201r3fPPrUMrcsctJN27T/4ISz5rvzth2rv3rg+Sve25wbT77hzd8c9PVPR/5w
gUXK9/zizHdHvXPom6tMvGfTJZbZ88yXDz1r/SVOvzV10n5/33nlW0+7+sZHExePmHZ1/a/jr//NxzuHH56y/PlrP/LrD16+MX958PT0qgs+dd
beKzx3xhanv3zk7bvss95ip1/6RNcG7+22Tfrz8O8u+HTpbQ9YddPXJ1yyy1Z3tbUfmB3zxEMXnHTyWSdddt8B+xx/6kt/nbzEx0te/PGia71/
6wo7/9AJHUcGt/ht165XzPP5CukvI2s+tPDlP+bq9w47fPKxB09ca6kp197+1W+fnm/R3Jqj/nLHqUdfeMNzPzz5t7+6fmp6y4V222SlsSsd/c
ty9Jid7tpq2ss/+GCLS6667/6pB25yxvj2cfnwOYv8/Yv5lkt9MzFxy7TFflDIHLDePRPX22K35KG3eOPGx/Lucqf++sI9p2y8zmK/PXe/5f/7
u/+G/zpy15Oue2eFzne3Peeu2B92W/pf5x76Tmb3z+6auMHU3d5fv7bcAsdcOuKcxzsWe+aZo66qHPDbc+p3THzkTydfe/Kn7ZPW/eyxdPuYX6
y8yLkLfPHNHyZ8eOV7mT3WmmftY6af+KutFj/kD3+bGlno31d/sdX/jtvjt0ut93A189Zdv3rq0u2Xuil2W+i8D5/eYI17x82b+zh1zTp7HPK3
XO6zm14YdVLg0fkefOuI9keX/urSXVZfduTVqeuXefDKo9b/+6QlttzhqosOv2ifJa5eNVpbcN55F7t+nYtfOemrJ3ZZ+qjbTpi81Vu/32DCgb
/o+PvNp0R2PXfJysuL3Ro7coORD35452fr3LXHfa/88KjnFzqsMvrvnfss+cszR//8xrGPnXvAhtvutfczW0aOefaVF0YdNf3t9M+umfzf226b
1r7OWcdtffTUV89a5IOn7z/tlg+vOOS01cMLX3vlMflx977UfsXvjz3upXzurOOe2HGnfy18UGqj9b/eb8oSvz/3xUx4vesfueHi55d9b4/Tn/
rigdsnjbqxXL33qa9O+O22C314/QFbXb7dYx/8/oDz8ycts/85G210xfLvjCkd0b7gUf85ecHSvif87YU1Xr3rqYufeuPqLfd5Y7EnJt3/ZnjP
JYP5H/9ikyWX++lHC91w7zXTTlt3wv1vtBcL49qOevSXe171zBZn/eo3n2z5wUpXPfjAxbtt97vX1k59deIT34xa6Pobln99gZ12zP1owrZ/X+
OAcV3z/m1aOD523pX3OHT3c68fv/bpTy/35IGlX1wXPPTYBa/b6vyvN7j38R/t+PSVt5yZuaT9gZPvWvrSHe+9/NrEEYd+/PZCt51+7PNbvR89
Zc1jX1x+sQPOPH/3HQ764OKnXzv9uFO3OCSxyo9v3fmnqZ/s8btz73n6pxud9d8lb311ZNsqd3pdeMLJG8T/c9QWT62zxvG/2e7CZcf9+4fLd5
6U/tcn1+366wuvHXvThs+lX3ll/31W2GSFvTf+eLNrP1z7h7/b/Mu1V7xhvi8y4YM2+jiw7qJblJc+/bkHpi304SXtf1/+2NsX+cMN+2x6UeXO
Laavsfivzk9cuUjnzuOWXeLDJer7njXfEque82j1le1PeXVK8N6vPrhvjSU/yE37ZqPLg5euuV7lgq86NhqV2HqnW0jv8EXX0WcseP7UsU88vc
/P33priW2K0YPHfbDblTemPz/+xhXeP+mtKY8d/2D59us2XCt7wVnz3vji7/befN3DHotn1r7myL9O33DKC1/uc/haf85eefoTJ/5n+5s+aYuW
Lrl99KlLX3poZs0lbr7pL+sefdhjiaPP/f2uB/9k8/jUPReYMG3tj8/80xOTL9s5VZ4xfpczfOHrVTe+99Tp5c6f3nHHY8uu8Ztxkw/9y4Q7d3
p68zs+WvakyNVHP/rm9T9q2/Rv29+1Ol+vf0nXFR+v+Y/HVtpu8opvLzTh0w1/f8PS7762zkEvLPXVj5M3rX7ugp8ttVx0yuX/vHbJMaf/4Mmd
3j32ucsPP/eE1RdeeewuZx6/1+3PrfKbCc+/dMa1T+7UdtnHV63TudbZt/y988t3//7pT157qJzO3v/e8nvtumPyur2eWT656antd5366lYfjR
l33DvVr+sXnrvvpMpOW9w8/9N//vMT61+z00fn3D3h2of/Fztz74feOGX93xu+P//zjd+afM+yiz9RdsLCD03dZf6tly2tdOQPfvXENq+tfOxy
3xx+yHv3fzF5mY9fvuqLUZ91fTpq6n1TcmPWfvrl+28vvnrn/V/dNvKlpf9x3BX3/u/ZW/+7zW+e+vFC/7tn9MZ7f7DVF9uO+e2E+9PTPv/ziI
9HPVY86cxnf5k4Z9eLHubanTqWvnH1cZe/cns8t9RJz7LhC+Pv3OH+6g9OPGb8i4kdb+KoY9782zflXZ+q7frjw97Y7cBJR/9mkS2XPvPY2EQX
m3fdzY/cbo/3Fzn5nNJTH9530G/+92zHofvd/+5R2RdCLz5x8PE7h+/unLDlZnfe+ek/Vlr4wzsvPnz/P78y8flf3v5k25vvLbv6c3ftvc6f3p
5/vT9Vl1/5968u+OW7By359Q6b/+Po/7ywznvbbPPNqLFXP/9pcu2JP3lk0nFLvJjKb/zbxQ8/fIkRX65w1K/ufmeJUe2vrXTu0xMi1z3+l+tf
7vj6tkNO3WfTcZXnNlnlwtuOHvHff66+xKpLP3LKToufcvIXz1x58MIf3j/toLvWS93+xI2XnH5sdcH1frbhk0evcebB2dU+2W371w+85YkNSl
e8PGPPLaobv7D/Kj/Y5MojXtzv0P888u6UrXObb7TJ7X/a/6qxq+xYWu3LDQ6fmK4vs9IrX9wzdtxPfvjhp2dNOHqbWO3I8x/c88Cppxxz0asP
/uK4Be9ZJDHvy39d/LHdfnzX/lMueWqv1S745x73jr/u4Dvja119TXzVLde96KT0W+duvfr7a57++DkLnnXm+l+uNvrnqfsuXXnllzY6/v4F37
l/uakX/+eDQn6p+6Y9t8Y/77p/ubsvvfD36z9z2k3rrP6PS0Z9NHLpP/js9aN/sd/ev1/z/HUfOebnOz7x7yuOfn3/jVff5dZLj4oevOlVJ/3r
zUWCi523yoLbXFp+fnL8oVptw2M2uHiHV65+fOSf1hi53fFvl+/bep5fbXXjU59eddsyYzdcr/jkuAW73nzuq9OfnHryiGWeevac8x78wxLPVN
fcf+0F1316ni2feX70DutMPvf+Q8P5Dw/ZdtT19zz9xVd3P7jVK59//Om1p2114KfTSyy43uRlbspu+DgwS/OP0C1PTDot0DF59CtHn/HjaHbT
fz47Kf/sEXuevNiFP1qxUFxhdvkXGPv2NpctPTa+xLnTOw7/+px9Dltn5N43LFsYucdrXTM2ff7yrtnl/9F9HaHjTv7Ll49OGPHA5LZDj9tnr4
2Xiozb8oAZj2745kZXfH3Y7PKfw04rLL3zmHmXTW+Xfnuvg3IjtvzrQjvtu+iLn576dHqNX79zz+zy7/Las4d+Pt8LvL/aB4uet8hDbLNq50NL
bwJ/++OIzcPLMVvz/q8fTFxfgjW3+XjLj3Jw945vXnHz8dvSfuyBW5+53+zzLxCovvnBqrDt0TPO/hyYvO1lk46cCKvudOYLu/N/b95e8ue7nb
dyvH7CTScdv9Qia1XuOyH61bX73v7G8x89d3zu2JEnnbDn/BMmX73jDfXshReuHzns8a8fvuekEx5+6v1lt/zX/S8/sdl7+5xw6S4nRp9a7F8X
PHfVl4ftssdzq27/7qtv37zBAw/GfrTZHxf7/OOjNl9lnQ8vOe/dyb997oufXnrnMhue3bnlSjv+eEzyiQ9ffeSRf0888qIvdzvnsQlL7LrpWa
9cuWj12C+v2/LEVVbZfYMrvn7wjBN/8OmX47Z+e/K77+1/yCaP7dD1ydQ3P//z5HETc8VrtqhfHlts4wP2GDvqmveX3myT342Yck1sg00WXXuD
c968/OUL/3bOTsdMuuOSXx/4O06evPrbfz3m0jEXrZPe/IQdjrxkuQ+OXPjd8dHDt1327rsWvTX+8RFPjF9pmSXW3mz/v9xw/SG//POzLxxx6a
aHbvrZRsd+GCjcffL+Nxz3wS5nceFC6xxw9rNHH3v01Z/ddOCVR9387q3hI9cqhuedMGmz6AnXdCz888v+sdIxe+27w+mv3z523bU+mXj3vmf8
Yckt3x+1262VayevcP9iR48/aJPtd9zirHcWn/Dxhbf8/crDtj94wac73iiudvl3pd85Lz272XKrHvnofZtc+XH0xBMWWO78fPj480+49IDsff
utOn67Lz6N/b5wVvqSVS757e537/GHeUbvM+mfW7z/+uTCCxMXvL/ztKVT35zwzrHRdHHTS44/7Z/t096duOZqJ+594OpHxpcYefeBb00Zc/yX
9x236oPT9rtwdHAdJ30RmnzNddufc9rUo7Ze5O37asGFnwtscmzpmxXXuzl+9fJLjapdc2p0k8uOzHz+z1GptX1wkzeSJx1eKl+3xVNnHfWHB9
baet17LiyN+nzDNzc+/q4/vjGj7cMXx2wzfs2nLnpw21V222WdlzL/GJUa+ZPNPj7/uqt2u+Pof8dvbrvwhBe/Kl6wwPU3LvX8a4nV7rj/uY0e
Pf+UxYrrHHHwq+ednPry6NfmHXPip9sd8uIdS8euf/mGK/97+eE/Ouamyzfb7snXwrsfOc/mV6yX3vDnqXE/HXfKvm3p6s8nvPDp1qVnbxp75G
UrfvzGbcdfdPEfp651x7gvf/jv3afuvebkv05Z8ctnPvvqhvfzF3f9aM+D99x+5zf2X+2XRx+24C8vG/uX+6++5q6Tpxz251djz6102nxb/HPc
v39230WjDj5vleyDP9/o320b//UADvnlLm90/GyV8Zkjjt+3cNKvCguunPtwvs03Ke+25SXHBNZetHL2/PW/Fn++wVOhPS+ZuvyV+6Z3Tt4y6r
2/X9n1kyW3m7jAZ0c/vP3+9z45/v77ljr1kcdufmGvfcdUT/zb2uv876oRf6qfc9bKI8e+FD+48tjG+2+w4HXrH7zNqF/f8qPTnnjh8x02v+y4
6qIfT57oX9r+tm96nTVXeWXyin94P/XrT2KHjh69whW3rDBmk0TytxtPq/1ku93XKN+/+uo3LHnmBY8fG1pmh7FL/W37n9VOzv/rvMt3eHzC2h
sv9PhCFy1w3YoXXF7NXrzGRvuv+YtIV/6gJ1845NxDqrd/NfXoHzwZ+OAYsluc9eZCh56Y3OiBykX3H/7r++rBvZ+5f6UTzzv95b9/Efn7a+Gf
HPrzI//xbPHGP+698imj2tf73fo/32fhf40bf9nRb+z75rhpY46ctNPz8a32uuyOD3/z11eXW6P2+M0H33rg7/dY5o8rr779ig899u6dv9lh2d
RjP/3jAx9tdtlRjx2zzcJP3NL5+VnBI37o829++qOzvynscUaxvNkx5x36/GHzPrH0Bp1bjKnsP/X2azcMfnLJ0hO3z7zx6GELrVhZ+5kfjf3Z
bj8L/mNU6cLS6oevf8eS6x6w3H+v/8fH2332u8y06je3j1t8/P7VO79a7tHVVnj7pOUfXOvsQz9bO3PrBr/913tPvfTIj24cO2mNcafc9MAuL/
38rJ8fccoxa7261HXXP7raMeeu8NXrBy0XjwXfWO+80VueetuZL/xl00evPn79pSauNupfTx791n8p733TtPjXl74W/O1bW203fvFRGx+y/BF3
3HZ+8PTUxbc8fuEif/vB4bufdO3f9vnnWSv96z8fPbjDFflXvzh2yTumT4ofcNNfLjzi0/3/EMitsPR/3zv62LeXm7THT57u/FvbpDGrv3zQn5
d79ZSx41Y//ZTjTz77+UTg5s0eOPnSPXZZ4qZNzwptedlaf/r8llM33mHrh0/bYuunn1zjqZX2LJy42cZjxl39+XZscsyKO7x920KH7H/Re3fu
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
`
