// shuffle: distributes the 32-byte records of the input files over N shard files at random and
// shuffles every shard, so that a trainer can read them one by one with little memory.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"
)

func main() {
	out := flag.String("out", "", "output directory")
	per := flag.Int("per", 1<<20, "records per shard (about)")
	maxAbs := flag.Int("maxabs", 20000, "drop records with a larger |score|")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()
	if *out == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: shuffle -out DIR [-per N] files...")
		os.Exit(2)
	}
	var total int64
	for _, f := range flag.Args() {
		st, err := os.Stat(f)
		if err != nil {
			panic(err)
		}
		total += st.Size() / 32
	}
	n := int(total/int64(*per)) + 1
	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}
	rng := rand.New(rand.NewSource(*seed))
	files := make([]*os.File, n)
	ws := make([]*bufio.Writer, n)
	for i := range ws {
		f, err := os.Create(filepath.Join(*out, fmt.Sprintf("s-%04d.bin", i)))
		if err != nil {
			panic(err)
		}
		files[i], ws[i] = f, bufio.NewWriterSize(f, 1<<17)
	}
	var kept, dropped int64
	var res [3]int64
	var men [33]int64
	var sumAbs float64
	var rec [32]byte
	for _, name := range flag.Args() {
		f, err := os.Open(name)
		if err != nil {
			panic(err)
		}
		r := bufio.NewReaderSize(f, 1<<20)
		for {
			if _, err := io.ReadFull(r, rec[:]); err != nil {
				break
			}
			s := int(int16(binary.LittleEndian.Uint16(rec[24:])))
			if s > *maxAbs || s < -*maxAbs || rec[26] > 2 || rec[27] > 1 {
				dropped++
				continue
			}
			kept++
			res[rec[26]]++
			men[bits.OnesCount64(binary.LittleEndian.Uint64(rec[:]))]++
			sumAbs += math.Abs(float64(s))
			ws[rng.Intn(n)].Write(rec[:])
		}
		f.Close()
	}
	for i := range ws {
		if err := ws[i].Flush(); err != nil {
			panic(err)
		}
		files[i].Close()
	}
	for i := 0; i < n; i++ {
		name := filepath.Join(*out, fmt.Sprintf("s-%04d.bin", i))
		data, err := os.ReadFile(name)
		if err != nil {
			panic(err)
		}
		m := len(data) / 32
		var a, b [32]byte
		for j := m - 1; j > 0; j-- {
			k := rng.Intn(j + 1)
			copy(a[:], data[32*j:])
			copy(b[:], data[32*k:])
			copy(data[32*j:], b[:])
			copy(data[32*k:], a[:])
		}
		if err := os.WriteFile(name, data, 0o644); err != nil {
			panic(err)
		}
	}
	fmt.Printf("records %d (dropped %d) in %d shards | white wins %.1f%% draws %.1f%% black wins %.1f%% | mean |score| %.0f\n",
		kept, dropped, n, 100*float64(res[2])/float64(kept), 100*float64(res[1])/float64(kept),
		100*float64(res[0])/float64(kept), sumAbs/float64(kept))
	fmt.Print("men:")
	for i, c := range men {
		if c > 0 {
			fmt.Printf(" %d:%.1f%%", i, 100*float64(c)/float64(kept))
		}
	}
	fmt.Println()
}
