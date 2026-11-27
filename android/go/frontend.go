// The Android app's other half. This file is no program of its own: the build
// (../Makefile) compiles it together with the unchanged ../../chess.go, same
// package, chess.go listed first, into the executable that the app starts as
// a child process:
//
//	libchess.so android <limit> <cpu 0|1> <demo 0|1> [the moves played so far]
//
// It drives the gui of chess.go, the state machine behind the X11 window
// (selection, cancelling, drag & drop, promotion chooser, CPU MOVE, demo
// mode), and talks to the app in lines of text.
//
// From the app, on stdin; x and y are in that gui's units, guiSQ to a square,
// from the top left corner of the board:
//
//	size <n> <dir>   a square is n pixels on the screen: draw the piece
//	            pictures into that directory (wP.png ... bK.png)
//	d <x> <y>   finger down
//	m <x> <y>   finger moved (needed once, when the piece leaves its place)
//	u <x> <y>   finger up
//	cancel      the gesture was taken away: drop the selection
//	cpu         the CPU MOVE button
//	quit        the end of stdin does the same
//
// To the app, on stdout; squares are 0 (a1) to 63 (h8), -1 is none:
//
//	ready <pid> <guiSQ> <threads> <hash MB>
//	sprites <n> <dir>             the pictures of that size are there
//	state <board a1..h8> <w|b to move> <selected> <last from> <last to>
//	      <promotion square> <held> <thinking> <over> <button> <status text>
//	moves <every move so far>     what to start it with again, should it die
//	error <why it cannot start>
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

func init() {
	if len(os.Args) > 1 && os.Args[1] == "android" {
		droidNice()
		os.Exit(droidRun(os.Args[2:], os.Stdin, os.Stdout, nil))
	}
}

// droidNice gives the game's threads a low priority. Started by the app they
// have that of the thread which draws its screen, and eight such threads at
// work leave the app, and the rest of the phone, little time for anything.
// The search is no slower for it: nothing else wants all the cores. Threads
// started later inherit the setting.
func droidNice() {
	tasks, _ := os.ReadDir("/proc/self/task")
	for _, t := range tasks {
		if tid, err := strconv.Atoi(t.Name()); err == nil {
			syscall.Setpriority(syscall.PRIO_PROCESS, tid, 10)
		}
	}
}

// droid is one game played through the app.
type droid struct {
	g     *gui
	w     *bufio.Writer
	moves []string // every move played, as parseMove reads them
	told  int      // how many of them the app has been sent
	last  string   // the state line sent last
	part  []byte   // unfinished line of the gui's printout
}

var droidOnce sync.Once

// droidHashMB is the hash size on a phone: a quarter of the free memory,
// 64 to 512 MB. The desktop's 80% would get the app killed.
func droidHashMB(avail uint64) int {
	return max(64, min(int(avail>>22), 512))
}

func droidLetter(pc int) byte {
	if pc < 0 {
		return " pnbrqk"[-pc]
	}
	return ".PNBRQK"[pc]
}

func droidIsMove(s string) bool {
	return (len(s) == 4 || len(s) == 5) && s[0] >= 'a' && s[0] <= 'h' && s[1] >= '1' && s[1] <= '8' &&
		s[2] >= 'a' && s[2] <= 'h' && s[3] >= '1' && s[3] <= '8'
}

// Write receives what the gui prints, the moves among "check" and the
// result, and keeps the moves.
func (d *droid) Write(b []byte) (int, error) {
	d.part = append(d.part, b...)
	for {
		i := bytes.IndexByte(d.part, '\n')
		if i < 0 {
			return len(b), nil
		}
		if s := string(d.part[:i]); droidIsMove(s) {
			d.moves = append(d.moves, s)
		}
		d.part = d.part[i+1:]
	}
}

// droidRun plays one game over in and out; think replaces the engine in tests.
func droidRun(args []string, in io.Reader, out io.Writer, think func(*Pos) Move) int {
	d := &droid{w: bufio.NewWriter(out), told: -1}
	fail := func(format string, a ...any) int {
		fmt.Fprintf(d.w, "error "+format+"\n", a...)
		d.w.Flush()
		return 2
	}
	if len(args) < 3 {
		return fail("usage: android <limit> <cpu 0|1> <demo 0|1> [moves]")
	}
	if _, err := strconv.ParseFloat(args[0], 64); err != nil {
		return fail("bad limit %q", args[0])
	}
	a := []string{"gui", "--icons", args[0]}
	if args[1] != "1" {
		a = append(a, "--no-cpu")
	}
	if args[2] == "1" {
		a = append(a, "demo")
	}
	cfg, err := parseArgs(a, func(string) string { return "" })
	if err != nil {
		return fail("%v", err)
	}
	if len(knightSeq[0]) == 0 {
		return fail("built wrongly: chess.go has to come before this file")
	}
	droidOnce.Do(func() {
		if os.Getenv("CHESS_HASH_MB") == "" {
			os.Setenv("CHESS_HASH_MB", strconv.Itoa(droidHashMB(availableRAM())))
		}
		initTT(cfg.budget)
		initZobrist()
		initMasks()
		initLMR()
		kpkOnce.Do(kpkInit)
		nnOnce.Do(nnInit)
	})
	p := startPos()
	if fen := os.Getenv("CHESS_FEN"); fen != "" {
		p = fromFEN(fen) // testing aid, as in main
	}
	from, to := -1, -1
	for _, s := range args[3:] {
		if res, _ := gameResult(p); res != "" {
			break
		}
		m, bad := parseMove(p, s, p.genLegal())
		if bad != "" {
			break // a damaged record: play on from its sound part
		}
		p.make(m)
		from, to = m.From(), m.To()
		d.moves = append(d.moves, moveStr(m))
	}
	if think == nil {
		think = func(q *Pos) Move { return bestMove(q, cfg.budget, cfg.depthCap) }
	}
	d.g = newGUI(p, cfg.opt, cfg.demo, think, d)
	d.g.lastFrom, d.g.lastTo = from, to
	fmt.Fprintf(d.w, "ready %d %d %d %d\n", os.Getpid(), guiSQ, searchThreads(), ttSize*ttEntryBytes>>20)

	cmds := make(chan string, 256)
	go func() {
		sc := bufio.NewScanner(in)
		for sc.Scan() {
			cmds <- sc.Text()
		}
		close(cmds)
	}()
	for {
		d.emit()
		select {
		case m := <-d.g.engCh:
			d.g.engineMoved(m)
		case line, ok := <-cmds:
			if !ok || d.command(line) {
				return 0
			}
		}
	}
}

// command obeys one line of the app; true means quit.
func (d *droid) command(line string) bool {
	g, f := d.g, strings.Fields(line)
	if len(f) == 0 {
		return false
	}
	x, y, xy := 0, 0, false
	if len(f) == 3 {
		var e1, e2 error
		x, e1 = strconv.Atoi(f[1])
		y, e2 = strconv.Atoi(f[2])
		xy = e1 == nil && e2 == nil
	}
	switch {
	case f[0] == "d" && xy && y < guiBW: // below the board is the gui's own button
		g.press(x, y, 1)
	case f[0] == "m" && xy:
		g.motion(x, y)
	case f[0] == "u" && xy:
		g.release(x, y, 1)
	case f[0] == "cancel":
		g.press(0, 0, 3)
	case f[0] == "cpu" && g.hasBtn():
		bx, by := g.w/2, guiBW+guiStatusH+guiBtnH/2
		g.press(bx, by, 1)
		g.release(bx, by, 1)
	case f[0] == "size" && len(f) >= 3:
		if n, err := strconv.Atoi(f[1]); err == nil && n >= 8 && n <= 1024 {
			d.sprites(n, strings.SplitN(line, " ", 3)[2])
		}
	case f[0] == "quit":
		return true
	}
	return false
}

// droidPNG writes a picture of n by n pixels with sprite s in its middle.
func droidPNG(s *sprite, n int, file string) error {
	img, off := image.NewNRGBA(image.Rect(0, 0, n, n)), (n-s.w)/2
	for y := 0; y < s.h; y++ {
		for x := 0; x < s.w; x++ {
			g := s.g[y*s.w+x]
			img.SetNRGBA(off+x, off+y, color.NRGBA{g, g, g, s.a[y*s.w+x]})
		}
	}
	f, err := os.Create(file + ".tmp")
	if err != nil {
		return err
	}
	if err = (&png.Encoder{CompressionLevel: png.BestSpeed}).Encode(f, img); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(file+".tmp", file) // whole or not there
}

// sprites draws the twelve piece pictures, n pixels wide, into dir: wP.png
// to bK.png. As text on stdout they took the app seconds to take in while
// the engine was thinking; one after the other they take a phone most of a
// second to draw.
func (d *droid) sprites(n int, dir string) {
	err := os.MkdirAll(dir, 0o700)
	var errs [12]error
	var wg sync.WaitGroup
	for i, pc := range [12]int{WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			name := string([]byte{"wb"[i/6], "PNBRQK"[i%6]}) + ".png"
			errs[i] = droidPNG(iconSprite(pc, n), n, filepath.Join(dir, name))
		}()
	}
	wg.Wait()
	for _, e := range errs {
		if err == nil {
			err = e
		}
	}
	if err != nil {
		fmt.Fprintln(d.w, "error no piece pictures:", err)
		return
	}
	fmt.Fprintln(d.w, "sprites", n, dir)
}

// emit tells the app what has changed: the moves, then the picture.
func (d *droid) emit() {
	g := d.g
	if len(d.moves) != d.told {
		fmt.Fprintln(d.w, strings.TrimSpace("moves "+strings.Join(d.moves, " ")))
		d.told = len(d.moves)
	}
	var b [64]byte
	for sq, pc := range g.p.b {
		b[sq] = droidLetter(pc)
	}
	side, held := 'w', -1
	if g.p.side == Black {
		side = 'b'
	}
	if g.dragFrom >= 0 && g.dragMoved {
		held = g.dragFrom
	}
	bit := func(v bool) int {
		if v {
			return 1
		}
		return 0
	}
	s := fmt.Sprintf("state %s %c %d %d %d %d %d %d %d %d %s", b[:], side, g.sel, g.lastFrom, g.lastTo,
		g.promoTo, held, bit(g.thinking), bit(g.over), bit(g.hasBtn()), g.status)
	if s != d.last {
		fmt.Fprintln(d.w, s)
		d.last = s
	}
	d.w.Flush()
}
