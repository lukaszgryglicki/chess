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
// Searches in parallel on all CPU cores (Lazy SMP, shared lock-free TT).
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
	mg, eg  int // material + piece-square score, White minus Black
	phase   int // sum of phaseVal over all men
	hash    uint64
	pawnKey uint64
	kingSq  [2]int
	hist    []uint64 // hashes after every move (incl. start position)
}

// set puts piece pc on the empty square sq (board, piece sets and the
// incremental evaluation terms); clr takes it off again.
func (p *Pos) set(pc, sq int) {
	c, t := White, pc
	if pc < 0 {
		c, t = Black, -pc
	}
	bit := uint64(1) << uint(sq)
	p.b[sq] = pc
	p.bb[c][t] |= bit
	p.occ[c] |= bit
	p.mg += pstMG[pc+6][sq]
	p.eg += pstEG[pc+6][sq]
	p.phase += phaseVal[t]
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
	p.mg -= pstMG[pc+6][sq]
	p.eg -= pstEG[pc+6][sq]
	p.phase -= phaseVal[t]
}

// consistent reports whether the incremental state matches the board.
func (p *Pos) consistent() bool {
	var q Pos
	for sq, pc := range p.b {
		if pc != Empty {
			q.set(pc, sq)
		}
	}
	return q.bb == p.bb && q.occ == p.occ && q.mg == p.mg && q.eg == p.eg && q.phase == p.phase
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
	return u
}

func (p *Pos) unmake(m Move, u Undo) {
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

// material + piece-square value of every piece (index pc+6) on every
// square, White positive and Black negative: Pos keeps their sums.
var pstMG, pstEG [13][64]int

func init() {
	for t := WP; t <= WK; t++ {
		for sq := 0; sq < 64; sq++ {
			pstMG[6+t][sq] = mgVal[t] + mgPST[t][sq^56]
			pstEG[6+t][sq] = egVal[t] + egPST[t][sq^56]
			pstMG[6-t][sq] = -mgVal[t] - mgPST[t][sq]
			pstEG[6-t][sq] = -egVal[t] - egPST[t][sq]
		}
	}
}

// passed pawn bonus by relative rank (0 = own back rank)
var passedMG = [8]int{0, 4, 9, 15, 30, 60, 100, 0}
var passedEG = [8]int{0, 14, 24, 40, 70, 125, 195, 0}

// eval returns the score from the side-to-move's perspective (centipawns).
// PeSTO tapered material+PST plus: bishop pair, doubled/isolated/passed
// pawns, rooks on (semi-)open files, king pawn shield, tempo.
// ---------- evaluation: bitboard-assisted (mobility, king safety, threats) ----------

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

// mobility bonuses indexed by number of reachable safe squares
var mobN = [9]int{-50, -22, -8, 0, 7, 13, 18, 22, 26}
var mobB = [14]int{-40, -18, 0, 10, 18, 25, 30, 34, 38, 41, 44, 46, 48, 50}
var mobR = [15]int{-40, -18, -4, 2, 6, 10, 14, 18, 22, 25, 28, 31, 33, 35, 37}
var mobQ = [28]int{-20, -12, -6, -2, 2, 5, 8, 11, 13, 15, 17, 19, 21, 23, 25, 26,
	27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38}

// connected pawns bonus by rank (white perspective)
var connPawn = [8]int{0, 3, 5, 8, 14, 24, 40, 0}

// passedMask[c][sq]: where an enemy pawn stops the pawn of color c on sq
// from being passed (its own and the adjacent files, ahead of it).
var passedMask [2][64]uint64

func init() {
	for sq := 0; sq < 64; sq++ {
		f, r := sq&7, sq>>3
		files := fileABB << uint(f)
		if f > 0 {
			files |= fileABB << uint(f-1)
		}
		if f < 7 {
			files |= fileABB << uint(f+1)
		}
		below := uint64(1)<<uint(8*r) - 1
		passedMask[White][sq] = files &^ (below | 0xFF<<uint(8*r))
		passedMask[Black][sq] = files & below
	}
}

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

func (p *Pos) eval() int {
	mg, eg, phase := p.mg, p.eg, p.phase // material + piece-square tables
	w, b := &p.bb[White], &p.bb[Black]
	wp, bp := w[WP], b[WP]
	wOcc, bOcc := p.occ[White], p.occ[Black]
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
	for x := w[WN]; x != 0; x &= x - 1 {
		a := knightBB[bits.TrailingZeros64(x)]
		wAtt |= a
		c := bits.OnesCount64(a & wMobArea)
		mg += mobN[c]
		eg += mobN[c] * 2 / 3
		if t := a & bRing; t != 0 {
			wUnits += 2 * bits.OnesCount64(t)
			wAttackers++
		}
	}
	for x := w[WB]; x != 0; x &= x - 1 {
		a := bishopAtt(bits.TrailingZeros64(x), occ)
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
	for x := w[WR]; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		a := rookAtt(sq, occ)
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
		if file := fileABB << uint(sq&7); wp&file == 0 { // (semi-)open file
			if bp&file == 0 {
				mg += 25
				eg += 12
			} else {
				mg += 12
				eg += 6
			}
		}
	}
	for x := w[WQ]; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		a := rookAtt(sq, occ) | bishopAtt(sq, occ)
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
	for x := b[WN]; x != 0; x &= x - 1 {
		a := knightBB[bits.TrailingZeros64(x)]
		bAtt |= a
		c := bits.OnesCount64(a & bMobArea)
		mg -= mobN[c]
		eg -= mobN[c] * 2 / 3
		if t := a & wRing; t != 0 {
			bUnits += 2 * bits.OnesCount64(t)
			bAttackers++
		}
	}
	for x := b[WB]; x != 0; x &= x - 1 {
		a := bishopAtt(bits.TrailingZeros64(x), occ)
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
	for x := b[WR]; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		a := rookAtt(sq, occ)
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
		if file := fileABB << uint(sq&7); bp&file == 0 {
			if wp&file == 0 {
				mg -= 25
				eg -= 12
			} else {
				mg -= 12
				eg -= 6
			}
		}
	}
	for x := b[WQ]; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		a := rookAtt(sq, occ) | bishopAtt(sq, occ)
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
	if wAttackers >= 2 && w[WQ] != 0 {
		u := wUnits * wUnits / 6
		if u > 450 {
			u = 450
		}
		mg += u
		eg += u / 4
	}
	if bAttackers >= 2 && b[WQ] != 0 {
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
	for x := wp; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		if bp&passedMask[White][sq] == 0 {
			r, stop := sq>>3, sq+8
			if occ&(1<<uint(stop)) != 0 {
				mg -= 8
				eg -= passedEG[r] / 3
			}
			eg += 6*chebDist(bk, stop)*r/4 - 3*chebDist(wk, stop)*r/4
		}
	}
	for x := bp; x != 0; x &= x - 1 {
		sq := bits.TrailingZeros64(x)
		if wp&passedMask[Black][sq] == 0 {
			rr, stop := 7-sq>>3, sq-8
			if occ&(1<<uint(stop)) != 0 {
				mg += 8
				eg += passedEG[rr] / 3
			}
			eg -= 6*chebDist(wk, stop)*rr/4 - 3*chebDist(bk, stop)*rr/4
		}
	}
	// bishop pair
	if w[WB]&(w[WB]-1) != 0 {
		mg += 24
		eg += 45
	}
	if b[WB]&(b[WB]-1) != 0 {
		mg -= 24
		eg -= 45
	}
	// king pawn shield (middlegame): own pawns 1-2 ranks ahead of the king
	if kf, kr := wk&7, wk>>3; kr <= 1 {
		near := uint64(0xFFFF) << uint(8*(kr+1))
		for f := max(kf-1, 0); f <= min(kf+1, 7); f++ {
			file := fileABB << uint(f)
			if wp&file&near != 0 {
				mg += 8
			} else {
				mg -= 14
			}
			if (wp|bp)&file == 0 {
				mg -= 10
			}
		}
	}
	if kf, kr := bk&7, bk>>3; kr >= 6 {
		near := uint64(0xFFFF) << uint(8*(kr-2))
		for f := max(kf-1, 0); f <= min(kf+1, 7); f++ {
			file := fileABB << uint(f)
			if bp&file&near != 0 {
				mg -= 8
			} else {
				mg += 14
			}
			if (wp|bp)&file == 0 {
				mg += 10
			}
		}
	}
	if phase > 24 {
		phase = 24
	}
	score := (mg*phase + eg*(24-phase)) / 24
	if phase == 2 && wp|bp == 0 { // bishop and knight against the bare king
		if w[WB] != 0 && w[WN] != 0 && bOcc == b[WK] {
			score += kbnk(w[WB], bk, wk)
		} else if b[WB] != 0 && b[WN] != 0 && wOcc == w[WK] {
			score -= kbnk(b[WB], wk, bk)
		}
	}
	// scale down drawish endgames
	if phase <= 8 {
		strongerPawns := wp
		if score < 0 {
			strongerPawns = bp
		}
		if bits.OnesCount64(w[WB]) == 1 && bits.OnesCount64(b[WB]) == 1 &&
			w[WN]|w[WR]|w[WQ]|b[WN]|b[WR]|b[WQ] == 0 {
			// opposite-colored bishops: strong draw tendency
			if (w[WB]&lightSquares != 0) != (b[WB]&lightSquares != 0) {
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

// repetition: has the current position occurred before (game + search path)?
func (e *Engine) isRepetition() bool {
	p := e.p
	n := len(p.hist)
	// walk back through the positions with the same side to move since the
	// last irreversible move; a match with an earlier hash is scored as a
	// draw inside the search (2-fold).
	for i, first := n-3, max(n-1-p.rev, 0); i >= first; i -= 2 {
		if p.hist[i] == p.hash {
			return true
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
		if p.half >= 100 || e.isRepetition() || insufficientMaterial(p) {
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
	for i := 0; i < len(moves); i++ {
		if i == first {
			qs, qe = e.orderCaptures(moves[first:])
			qs, qe = qs+first, qe+first
		}
		if i == qs && qs < qe {
			if !pvNode && !inChk && legal > 0 && bestScore > -(MateScore-2000) && depth <= 8 &&
				staticEval+90+100*depth+futImp*b2i(improving) <= alpha {
				if i = qe; i == len(moves) { // futility prunes every quiet move
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
				if depth <= lmpDepth && quiets > lmpLimit {
					i = qe - 1 // late move pruning, of all the quiet moves left
					continue
				}
				if depth <= 8 && staticEval+90+100*depth+futImp*b2i(improving) <= alpha {
					i = qe - 1 // futility pruning, of all the quiet moves left
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
				panic("piece sets or scores do not match the board")
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
