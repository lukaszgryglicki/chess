// datagen: games of the program against itself with a node limit per move, written as 32-byte
// training records. It uses the search of chess.go, so it is built together with a copy of it
// whose main() has another name (see README.md). CHESS_NET=file plays with another network.
//
// record: occ uint64 | 16 bytes: 4 bits per man in the order of the bits of occ
// (0..5 white P..K, 6..11 black P..K) | score int16 (side to move) | result (0 black won,
// 1 draw, 2 white won) | side to move | ply uint16 | halfmove clock | flags (bit 0: played to the end)
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

func dgSearch(e *Engine, root *Pos, legal []Move, soft uint64, maxD int) (Move, int, int) {
	c := &searchCtl{}
	ttGen.Add(1)
	e.p = root.clone()
	e.p.nnAttach(&e.nn)
	e.ctl = c
	e.stop = false
	e.nodes, e.stores, e.seldepth = 0, 0, 0
	e.killers = [MaxPly][2]Move{}
	e.moveStack = [MaxPly]Move{}
	e.evalStack = [MaxPly]int{}
	e.pcStack = [MaxPly]int{}
	moves := append([]Move(nil), legal...)
	best, score, d := moves[0], 0, 0
	if e.p.inCheck(e.p.side) {
		e.evalStack[0] = evalNone
	} else {
		e.evalStack[0] = e.p.eval()
	}
	for depth := 1; depth <= maxD; depth++ {
		e.orderMoves(moves, best, 0)
		window := 30
		alpha, beta := -Inf, Inf
		if depth >= 5 {
			alpha, beta = score-window, score+window
		}
		var s int
		var m Move
		for {
			s, m = e.rootSearch(moves, depth, alpha, beta)
			if s > alpha && s < beta {
				break
			}
			window *= 3
			if s <= alpha {
				alpha = s - window
			} else {
				beta = s + window
			}
			if alpha < -MateScore {
				alpha = -Inf
			}
			if beta > MateScore {
				beta = Inf
			}
		}
		best, score, d = m, s, depth
		e.ttStore(e.p.hash, best, score, depth, ttExact, 0, e.evalStack[0])
		if score > MateScore-2000 || score < -(MateScore-2000) || e.nodes >= soft {
			break
		}
	}
	return best, score, d
}

func encode(p *Pos, score, ply int, rec *[32]byte) {
	occ := p.occ[White] | p.occ[Black]
	binary.LittleEndian.PutUint64(rec[0:], occ)
	for i := 8; i < 24; i++ {
		rec[i] = 0
	}
	n := 0
	for b := occ; b != 0; b &= b - 1 {
		rec[8+n/2] |= byte(pieceIdx(p.b[bits.TrailingZeros64(b)])) << uint(4*(n&1))
		n++
	}
	binary.LittleEndian.PutUint16(rec[24:], uint16(int16(score)))
	rec[26] = 1
	rec[27] = byte(p.side)
	binary.LittleEndian.PutUint16(rec[28:], uint16(ply))
	rec[30] = byte(min(p.half, 255))
	rec[31] = 0
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func main() {
	out := flag.String("out", "", "output file (appended)")
	seed := flag.Int64("seed", 1, "random seed")
	nodes := flag.Uint64("nodes", 5000, "soft node limit per move")
	maxD := flag.Int("depth", 14, "depth cap per move")
	games := flag.Int("games", 0, "games to play (0: until <ctl>/STOP exists)")
	ctl := flag.String("ctl", "", "control directory: STOP ends the run, PAUSE suspends it")
	full := flag.Int("full", 15, "percent of the games played to the end without adjudication")
	hash := flag.Int("hash", 8, "hash table size in MB")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "datagen: -out is required")
		os.Exit(2)
	}
	os.Setenv("CHESS_HASH_MB", fmt.Sprint(*hash))
	os.Setenv("CHESS_THREADS", "1")
	initTT(0)
	initZobrist()
	initMasks()
	initLMR()
	nnOnce.Do(nnInit)
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "datagen:", err)
		os.Exit(1)
	}
	w := bufio.NewWriterSize(f, 1<<16)
	rng := rand.New(rand.NewSource(*seed))
	e := &Engine{}
	var recs [][32]byte
	var nGames, nPos, nMoves, sumNodes, sumDepth uint64
	var res [3]uint64
	start := time.Now()
	for g := 0; *games == 0 || g < *games; g++ {
		if *ctl != "" {
			if exists(filepath.Join(*ctl, "STOP")) {
				break
			}
			for exists(filepath.Join(*ctl, "PAUSE")) && !exists(filepath.Join(*ctl, "STOP")) {
				time.Sleep(2 * time.Second)
			}
		}
		p := startPos()
		ok := true
		for i, n := 0, 4+rng.Intn(7); i < n && ok; i++ {
			legal := p.genLegal()
			if len(legal) == 0 {
				ok = false
				break
			}
			p.make(legal[rng.Intn(len(legal))])
			if r, _ := gameResult(p); r != "" {
				ok = false
			}
		}
		if !ok {
			continue
		}
		toEnd := rng.Intn(100) < *full
		recs = recs[:0]
		result, winRun, winSign, drawRun := -1, 0, 0, 0
		for ply := 0; ply < 500 && result < 0; ply++ {
			if r, _ := gameResult(p); r != "" {
				switch r {
				case "1-0":
					result = 2
				case "0-1":
					result = 0
				default:
					result = 1
				}
				break
			}
			legal := p.genLegal()
			if len(legal) == 1 {
				p.make(legal[0])
				continue
			}
			m, score, d := dgSearch(e, p, legal, *nodes, *maxD)
			nMoves++
			sumNodes += e.nodes
			sumDepth += uint64(d)
			if ply == 0 && (score > 1000 || score < -1000) {
				ok = false
				break
			}
			mate := score > MateScore-2000 || score < -(MateScore-2000)
			if !mate && m.Flags()&fCapture == 0 && m.Promo() == 0 && !p.inCheck(p.side) {
				var rec [32]byte
				encode(p, score, len(p.hist)-1, &rec)
				if toEnd {
					rec[31] = 1
				}
				recs = append(recs, rec)
			}
			ws := score
			if p.side == Black {
				ws = -ws
			}
			if !toEnd {
				sign := 0
				if ws >= 2000 {
					sign = 1
				} else if ws <= -2000 {
					sign = -1
				}
				if sign != 0 && sign == winSign {
					winRun++
				} else if sign != 0 {
					winSign, winRun = sign, 1
				} else {
					winSign, winRun = 0, 0
				}
				if winRun >= 5 {
					result = 1 + winSign
				}
				if ply >= 60 && ws >= -10 && ws <= 10 {
					drawRun++
				} else {
					drawRun = 0
				}
				if drawRun >= 12 {
					result = 1
				}
			}
			p.make(m)
		}
		if !ok {
			continue
		}
		if result < 0 {
			result = 1
		}
		for i := range recs {
			recs[i][26] = byte(result)
			w.Write(recs[i][:])
		}
		if err := w.Flush(); err != nil {
			fmt.Fprintln(os.Stderr, "datagen:", err)
			os.Exit(1)
		}
		nGames++
		nPos += uint64(len(recs))
		res[result]++
		if nGames%200 == 0 {
			el := time.Since(start).Seconds()
			fmt.Fprintf(os.Stderr, "games %d pos %d (%.0f/s) moves %d nodes/move %d depth %.1f | W %d D %d L %d\n",
				nGames, nPos, float64(nPos)/el, nMoves, sumNodes/max(nMoves, 1), float64(sumDepth)/float64(max(nMoves, 1)),
				res[2], res[1], res[0])
		}
	}
	el := time.Since(start).Seconds()
	fmt.Fprintf(os.Stderr, "done: games %d pos %d (%.0f/s) moves %d nodes/move %d depth %.1f | W %d D %d L %d | %.0f s\n",
		nGames, nPos, float64(nPos)/el, nMoves, sumNodes/max(nMoves, 1), float64(sumDepth)/float64(max(nMoves, 1)),
		res[2], res[1], res[0], el)
	f.Close()
}
