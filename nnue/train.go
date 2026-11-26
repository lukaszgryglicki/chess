// train: trains the evaluation network (768 -> H) x 2 perspectives -> 1 (x NB output buckets by the
// number of men) on shards of 32-byte records (see datagen.go, shuffle.go).
//
// Float model: acc = B1 + sum of W1 rows of the men (own/enemy, type, square as seen by one side,
// mirrored left to right when that side's king is on the files e-h);
// act = clamp(acc, 0, 63/64); out = B2[b] + act(side to move).W2[b][:H] + act(other).W2[b][H:]
// + sum of PQ[man][b] as the side to move sees the men - the same sum as the other side sees them;
// evaluation in centipawns = K * out; trained so that sigmoid(out) is the expected game score.
//
// Integer model written to -out (what the program computes): W1, B1 scaled by 256 (int16), W2 by 64
// (int8), B2 and PQ by 4096 (int32, int16); file: "CNN3", H, NB, K, 0 (uint16 each), W1 low bytes, W1 high bytes,
// B1, W2, B2, PQ low bytes, PQ high bytes; q = min((acc + 2) >> 2, 63), 0 for a negative acc;
// evaluation = (B2[b] + sum q * W2) * K / 4096.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const (
	nf    = 768
	amax  = 63.0 / 64.0
	piece = 128
)

type model struct {
	h, nb              int
	w1, b1, w2, b2, pq []float32
}

func newModel(h, nb int, rng *rand.Rand) *model {
	m := &model{h: h, nb: nb, w1: make([]float32, nf*h), b1: make([]float32, h),
		w2: make([]float32, nb*2*h), b2: make([]float32, nb), pq: make([]float32, nf*nb)}
	s1 := float32(1 / math.Sqrt(nf))
	for i := range m.w1 {
		m.w1[i] = (rng.Float32()*2 - 1) * s1
	}
	s2 := float32(1 / math.Sqrt(float64(2*h)))
	for i := range m.w2 {
		m.w2[i] = (rng.Float32()*2 - 1) * s2
	}
	return m
}

func (m *model) params() [][]float32 { return [][]float32{m.w1, m.b1, m.w2, m.b2, m.pq} }

// decode fills the feature rows of the side to move and of the other side; returns the number of men.
// Each side sees the board mirrored left to right when its own king is on the files e-h.
func decode(rec []byte, fs, fn *[32]int32) int {
	occ := binary.LittleEndian.Uint64(rec)
	stm := rec[27]
	var sqs, codes [32]int32
	var mw, mb int32
	n := 0
	for b := occ; b != 0; b &= b - 1 {
		sq := int32(bits.TrailingZeros64(b))
		code := int32(rec[8+n/2] >> uint(4*(n&1)) & 15)
		if code == 5 && sq&7 >= 4 {
			mw = 7
		} else if code == 11 && sq&7 >= 4 {
			mb = 7
		}
		sqs[n], codes[n] = sq, code
		n++
	}
	for i := 0; i < n; i++ {
		c, t := codes[i]/6, codes[i]%6
		iw := c*384 + t*64 + (sqs[i] ^ mw)
		ib := (1-c)*384 + t*64 + (sqs[i] ^ 56 ^ mb)
		if stm == 0 {
			fs[i], fn[i] = iw, ib
		} else {
			fs[i], fn[i] = ib, iw
		}
	}
	return n
}

func target(rec []byte, k, wdl float64) float64 {
	s := float64(int16(binary.LittleEndian.Uint16(rec[24:])))
	r := float64(rec[26]) / 2
	if rec[27] == 1 {
		r = 1 - r
	}
	return (1-wdl)/(1+math.Exp(-s/k)) + wdl*r
}

type worker struct {
	m              *model
	gw1, gb1       []float32
	gw2, gb2, gpq  []float32
	as, an, ds, dn []float32
	loss           float64
	seen           [nf]bool
}

func newWorker(m *model) *worker {
	h := m.h
	return &worker{m: m, gw1: make([]float32, nf*h), gb1: make([]float32, h), gw2: make([]float32, m.nb*2*h),
		gb2: make([]float32, m.nb), gpq: make([]float32, nf*m.nb), as: make([]float32, h), an: make([]float32, h), ds: make([]float32, h), dn: make([]float32, h)}
}

func addRow(dst, row []float32) {
	row = row[:len(dst)]
	for i := range dst {
		dst[i] += row[i]
	}
}

func (w *worker) forward(rec []byte, fs, fn *[32]int32) (out float32, n, bkt int) {
	m, h := w.m, w.m.h
	n = decode(rec, fs, fn)
	copy(w.as, m.b1)
	copy(w.an, m.b1)
	for i := 0; i < n; i++ {
		addRow(w.as, m.w1[int(fs[i])*h:])
		addRow(w.an, m.w1[int(fn[i])*h:])
	}
	bkt = min((n-2)/4, m.nb-1)
	w2 := m.w2[bkt*2*h : (bkt+1)*2*h]
	out = m.b2[bkt]
	for i := 0; i < n; i++ {
		out += m.pq[int(fs[i])*m.nb+bkt] - m.pq[int(fn[i])*m.nb+bkt]
	}
	as, an := w.as, w.an[:h]
	ws, wn := w2[:h], w2[h:]
	for j := range as {
		a := as[j]
		if a < 0 {
			a = 0
		} else if a > amax {
			a = amax
		}
		out += a * ws[j]
		b := an[j]
		if b < 0 {
			b = 0
		} else if b > amax {
			b = amax
		}
		out += b * wn[j]
	}
	return
}

func (w *worker) train(rec []byte, k, wdl float64) {
	m, h := w.m, w.m.h
	var fs, fn [32]int32
	out, n, bkt := w.forward(rec, &fs, &fn)
	p := 1 / (1 + math.Exp(-float64(out)))
	t := target(rec, k, wdl)
	w.loss += (p - t) * (p - t)
	g := float32(2 * (p - t) * p * (1 - p))
	w.gb2[bkt] += g
	w2 := m.w2[bkt*2*h : (bkt+1)*2*h]
	gw2 := w.gw2[bkt*2*h : (bkt+1)*2*h]
	as, an, ds, dn := w.as, w.an[:h], w.ds[:h], w.dn[:h]
	ws, wn, gs, gn, gb1 := w2[:h], w2[h:], gw2[:h], gw2[h:], w.gb1[:h]
	for j := range as {
		a, d := as[j], float32(0)
		if a > 0 {
			if a < amax {
				d = g * ws[j]
			} else {
				a = amax
			}
			gs[j] += g * a
		}
		ds[j] = d
		b, e := an[j], float32(0)
		if b > 0 {
			if b < amax {
				e = g * wn[j]
			} else {
				b = amax
			}
			gn[j] += g * b
		}
		dn[j] = e
		gb1[j] += d + e
	}
	for i := 0; i < n; i++ {
		addRow(w.gw1[int(fs[i])*h:int(fs[i])*h+h], ds)
		addRow(w.gw1[int(fn[i])*h:int(fn[i])*h+h], dn)
		w.gpq[int(fs[i])*m.nb+bkt] += g
		w.gpq[int(fn[i])*m.nb+bkt] -= g
		w.seen[fs[i]], w.seen[fn[i]] = true, true
	}
}

func (w *worker) grads() [][]float32 { return [][]float32{w.gw1, w.gb1, w.gw2, w.gb2, w.gpq} }

// quant is the integer network.
type quant struct {
	h, nb, k   int
	w1, b1, pq []int16
	w2         []int8
	b2         []int32
}

func clampRound(v float64, lo, hi int) int {
	r := int(math.Round(v))
	return max(lo, min(hi, r))
}

// quantize makes the integer network; the rows of inputs that never occurred (seen, when given) are zero.
func (m *model) quantize(k int, seen *[nf]bool) *quant {
	q := &quant{h: m.h, nb: m.nb, k: k, w1: make([]int16, len(m.w1)), b1: make([]int16, len(m.b1)),
		w2: make([]int8, len(m.w2)), b2: make([]int32, len(m.b2)), pq: make([]int16, len(m.pq))}
	for i, v := range m.w1 {
		if seen == nil || seen[i/m.h] {
			q.w1[i] = int16(clampRound(float64(v)*256, -480, 480))
		}
	}
	for i, v := range m.pq {
		q.pq[i] = int16(clampRound(float64(v)*4096, -32767, 32767))
	}
	for i, v := range m.b1 {
		q.b1[i] = int16(clampRound(float64(v)*256, -1000, 1000))
	}
	for i, v := range m.w2 {
		q.w2[i] = int8(clampRound(float64(v)*64, -128, 127))
	}
	for i, v := range m.b2 {
		q.b2[i] = int32(clampRound(float64(v)*4096, -1<<24, 1<<24))
	}
	return q
}

func (q *quant) eval(rec []byte) int {
	var fs, fn [32]int32
	n := decode(rec, &fs, &fn)
	h := q.h
	bkt := min((n-2)/4, q.nb-1)
	sum := int(q.b2[bkt])
	for i := 0; i < n; i++ {
		sum += int(q.pq[int(fs[i])*q.nb+bkt]) - int(q.pq[int(fn[i])*q.nb+bkt])
	}
	w2 := q.w2[bkt*2*h:]
	for j := 0; j < h; j++ {
		s, t := int(q.b1[j]), int(q.b1[j])
		for i := 0; i < n; i++ {
			s += int(q.w1[int(fs[i])*h+j])
			t += int(q.w1[int(fn[i])*h+j])
		}
		sum += max(0, min((s+2)>>2, 63))*int(w2[j]) + max(0, min((t+2)>>2, 63))*int(w2[h+j])
	}
	return sum * q.k / 4096
}

func (q *quant) write(name string) error {
	buf := make([]byte, 0, 12+2*len(q.w1)+2*len(q.b1)+len(q.w2)+4*len(q.b2))
	buf = append(buf, "CNN3"...)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(q.h))
	buf = binary.LittleEndian.AppendUint16(buf, uint16(q.nb))
	buf = binary.LittleEndian.AppendUint16(buf, uint16(q.k))
	buf = binary.LittleEndian.AppendUint16(buf, 0)
	for _, v := range q.w1 { // low bytes, then high bytes: compresses better
		buf = append(buf, byte(v))
	}
	for _, v := range q.w1 {
		buf = append(buf, byte(uint16(v)>>8))
	}
	for _, v := range q.b1 {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(v))
	}
	for _, v := range q.w2 {
		buf = append(buf, byte(v))
	}
	for _, v := range q.b2 {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(v))
	}
	for _, v := range q.pq {
		buf = append(buf, byte(v))
	}
	for _, v := range q.pq {
		buf = append(buf, byte(uint16(v)>>8))
	}
	return os.WriteFile(name, buf, 0o644)
}

func (m *model) save(name string) error {
	var n int
	for _, p := range m.params() {
		n += len(p)
	}
	buf := make([]byte, 0, 4*n)
	for _, p := range m.params() {
		for _, v := range p {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}
	return os.WriteFile(name, buf, 0o644)
}

func (m *model) load(name string) error {
	buf, err := os.ReadFile(name)
	if err != nil {
		return err
	}
	for _, p := range m.params() {
		if len(buf) < 4*len(p) {
			return fmt.Errorf("%s: too short", name)
		}
		for i := range p {
			p[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		buf = buf[4*len(p):]
	}
	return nil
}

// fitK finds the K for which sigmoid(score/K) is closest to the game results.
func fitK(data []byte) float64 {
	loss := func(k float64) float64 {
		var s float64
		for o := 0; o+32 <= len(data); o += 32 {
			d := target(data[o:], k, 0) - target(data[o:], k, 1)
			s += d * d
		}
		return s
	}
	lo, hi := 50.0, 2000.0
	for hi-lo > 1 {
		a, b := lo+(hi-lo)/3, hi-(hi-lo)/3
		if loss(a) < loss(b) {
			hi = b
		} else {
			lo = a
		}
	}
	return (lo + hi) / 2
}

func main() {
	dir := flag.String("shards", "", "directory with the shards")
	nval := flag.Int("val", 1, "shards kept for validation (the last ones)")
	h := flag.Int("h", 128, "hidden units (a multiple of 4)")
	nb := flag.Int("nb", 8, "output buckets")
	epochs := flag.Int("epochs", 10, "passes over the data")
	batch := flag.Int("batch", 16384, "batch size")
	lr0 := flag.Float64("lr", 0.001, "learning rate at the start")
	lr1 := flag.Float64("lrend", 0.00002, "learning rate at the end")
	wdl := flag.Float64("wdl", 0.3, "weight of the game result in the target")
	kflag := flag.Float64("k", 0, "centipawns per unit of the output (0: fitted to the results)")
	threads := flag.Int("threads", 8, "worker threads")
	out := flag.String("out", "net", "output name: NAME.bin (integer network), NAME.f32 (float weights), NAME.probe")
	initFrom := flag.String("init", "", "float weights to start from")
	seed := flag.Int64("seed", 1, "random seed")
	maxShards := flag.Int("max", 0, "use only the first N training shards (0: all)")
	flag.Parse()
	names, _ := filepath.Glob(filepath.Join(*dir, "s-*.bin"))
	sort.Strings(names)
	if len(names) <= *nval || *h%4 != 0 {
		fmt.Fprintln(os.Stderr, "train: need more shards than validation shards and h a multiple of 4")
		os.Exit(2)
	}
	valNames, names := names[len(names)-*nval:], names[:len(names)-*nval]
	if *maxShards > 0 && *maxShards < len(names) {
		names = names[:*maxShards]
	}
	var val []byte
	for _, n := range valNames {
		d, err := os.ReadFile(n)
		if err != nil {
			panic(err)
		}
		val = append(val, d[:len(d)/32*32]...)
	}
	var total int64
	for _, n := range names {
		st, _ := os.Stat(n)
		total += st.Size() / 32
	}
	k := *kflag
	fitted := fitK(val)
	if k == 0 {
		k = math.Round(fitted)
	}
	rng := rand.New(rand.NewSource(*seed))
	m := newModel(*h, *nb, rng)
	if *initFrom != "" {
		if err := m.load(*initFrom); err != nil {
			panic(err)
		}
	}
	steps := int(total/int64(*batch)) * *epochs
	fmt.Printf("train: %d records in %d shards, validation %d, h %d, buckets %d, K %.0f (fitted %.0f), wdl %.2f, %d steps, %d threads\n",
		total, len(names), len(val)/32, *h, *nb, k, fitted, *wdl, steps, *threads)
	ws := make([]*worker, *threads)
	for i := range ws {
		ws[i] = newWorker(m)
	}
	ps := m.params()
	am := make([][]float32, len(ps))
	av := make([][]float32, len(ps))
	gt := make([][]float32, len(ps))
	for i, p := range ps {
		am[i], av[i], gt[i] = make([]float32, len(p)), make([]float32, len(p)), make([]float32, len(p))
	}
	lim := [5][2]float32{{-1.875, 1.875}, {-3.9, 3.9}, {-2, 127.0 / 64}, {-4000, 4000}, {-7.99, 7.99}}
	const b1c, b2c, eps = 0.9, 0.999, 1e-8
	valLoss := func() (fl, ql, diff float64) {
		q := m.quantize(int(k), nil)
		n := len(val) / 32
		type res struct{ fl, ql, diff float64 }
		rs := make([]res, *threads)
		var wg sync.WaitGroup
		for t := 0; t < *threads; t++ {
			wg.Add(1)
			go func(t int) {
				defer wg.Done()
				w := newWorker(m)
				var fs, fn [32]int32
				for i := t; i < n; i += *threads {
					rec := val[32*i : 32*i+32]
					o, _, _ := w.forward(rec, &fs, &fn)
					tg := target(rec, k, *wdl)
					p := 1 / (1 + math.Exp(-float64(o)))
					rs[t].fl += (p - tg) * (p - tg)
					e := float64(q.eval(rec))
					pq := 1 / (1 + math.Exp(-e/k))
					rs[t].ql += (pq - tg) * (pq - tg)
					rs[t].diff += math.Abs(e - float64(o)*k)
				}
			}(t)
		}
		wg.Wait()
		for _, r := range rs {
			fl += r.fl
			ql += r.ql
			diff += r.diff
		}
		return fl / float64(n), ql / float64(n), diff / float64(n)
	}
	start := time.Now()
	step := 0
	idx := make([]int32, 0, 1<<21)
	for ep := 1; ep <= *epochs; ep++ {
		var epLoss float64
		var epN int
		for _, si := range rng.Perm(len(names)) {
			data, err := os.ReadFile(names[si])
			if err != nil {
				panic(err)
			}
			n := len(data) / 32
			idx = idx[:0]
			for i := 0; i < n; i++ {
				idx = append(idx, int32(i))
			}
			rng.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
			for off := 0; off+*batch <= n; off += *batch {
				// the workers take the batch in small pieces, so that a slow or busy core does not hold it up
				var wg sync.WaitGroup
				var next atomic.Int64
				ids := idx[off : off+*batch]
				for _, w := range ws {
					wg.Add(1)
					go func(w *worker) {
						defer wg.Done()
						for _, g := range w.grads() {
							clear(g)
						}
						w.loss = 0
						for {
							hi := int(next.Add(piece))
							lo := hi - piece
							if lo >= len(ids) {
								return
							}
							for _, i := range ids[lo:min(hi, len(ids))] {
								w.train(data[32*int(i):32*int(i)+32], k, *wdl)
							}
						}
					}(w)
				}
				wg.Wait()
				step++
				lr := float32(*lr0 * math.Pow(*lr1 / *lr0, float64(step)/float64(steps)))
				c1 := float32(1 - math.Pow(b1c, float64(step)))
				c2 := float32(1 - math.Pow(b2c, float64(step)))
				inv := 1 / float32(*batch)
				var pw sync.WaitGroup
				for pi := range ps {
					pw.Add(1)
					go func(pi int) {
						defer pw.Done()
						g := gt[pi]
						copy(g, ws[0].grads()[pi])
						for _, w := range ws[1:] {
							addRow(g, w.grads()[pi])
						}
						p, mm, vv := ps[pi], am[pi], av[pi]
						lo, hi := lim[pi][0], lim[pi][1]
						for i, gi := range g {
							gi *= inv
							mm[i] = b1c*mm[i] + (1-b1c)*gi
							vv[i] = b2c*vv[i] + (1-b2c)*gi*gi
							x := p[i] - lr*(mm[i]/c1)/(float32(math.Sqrt(float64(vv[i]/c2)))+eps)
							if x < lo {
								x = lo
							} else if x > hi {
								x = hi
							}
							p[i] = x
						}
					}(pi)
				}
				pw.Wait()
				for _, w := range ws {
					epLoss += w.loss
				}
				epN += *batch
			}
		}
		fl, ql, diff := valLoss()
		fmt.Printf("epoch %2d  train %.6f  val %.6f  val(int) %.6f  |int-float| %.1f cp  lr %.6f  %.0f s  %.0fK pos/s\n",
			ep, epLoss/float64(epN), fl, ql, diff, *lr0*math.Pow(*lr1 / *lr0, float64(step)/float64(steps)),
			time.Since(start).Seconds(), float64(step)*float64(*batch)/time.Since(start).Seconds()/1000)
		if err := m.save(*out + ".f32"); err != nil {
			panic(err)
		}
	}
	var seen [nf]bool
	for _, w := range ws {
		for i, s := range w.seen {
			seen[i] = seen[i] || s
		}
	}
	q := m.quantize(int(k), &seen)
	if *epochs == 0 {
		q = m.quantize(int(k), nil)
	}
	if err := q.write(*out + ".bin"); err != nil {
		panic(err)
	}
	var maxW1, maxB1 int
	for _, v := range q.w1 {
		maxW1 = max(maxW1, int(v), -int(v))
	}
	for _, v := range q.b1 {
		maxB1 = max(maxB1, int(v), -int(v))
	}
	np := min(len(val)/32, 4096)
	probe := make([]byte, 0, 36*np)
	for i := 0; i < np; i++ {
		probe = append(probe, val[32*i:32*i+32]...)
		probe = binary.LittleEndian.AppendUint32(probe, uint32(int32(q.eval(val[32*i:]))))
	}
	if err := os.WriteFile(*out+".probe", probe, 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("done: %s.bin  max |W1| %d  max |B1| %d\n", *out, maxW1, maxB1)
}
