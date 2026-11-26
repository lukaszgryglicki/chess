package main

import (
	"bufio"
	"bytes"
	"fmt"
	"math/bits"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestMain lets the test binary double as the chess program (for the X11
// end-to-end test): CHESS_TEST_MAIN=1 runs main() with CHESS_TEST_ARGS.
func TestMain(m *testing.M) {
	if os.Getenv("CHESS_TEST_MAIN") == "1" {
		os.Args = append([]string{"chess"}, strings.Fields(os.Getenv("CHESS_TEST_ARGS"))...)
		main()
		os.Exit(0)
	}
	os.Setenv("CHESS_HASH_MB", "16")
	os.Setenv("CHESS_THREADS", "1")
	initTT(0)
	initZobrist()
	initMasks()
	initLMR()
	nnOnce.Do(nnInit)
	os.Exit(m.Run())
}

// ---------- GUI state machine (no X server needed) ----------

func sqOf(name string) int { return int(name[1]-'1')*8 + int(name[0]-'a') }

// center of a square in window coordinates
func sqXY(name string) (int, int) {
	sq := sqOf(name)
	return (sq&7)*guiSQ + guiSQ/2, (7-sq>>3)*guiSQ + guiSQ/2
}

type testGUI struct {
	*gui
	t   *testing.T
	out *bytes.Buffer
}

// newTestGUI builds a GUI whose "engine" answers with the first legal move.
func newTestGUI(t *testing.T, fen string, opt guiOpts, demo bool) *testGUI {
	t.Helper()
	p := startPos()
	if fen != "" {
		p = fromFEN(fen)
	}
	out := &bytes.Buffer{}
	think := func(q *Pos) Move { return q.genLegal()[0] }
	return &testGUI{newGUI(p, opt, demo, think, out), t, out}
}

func (g *testGUI) click(sq string) {
	x, y := sqXY(sq)
	g.press(x, y, 1)
	g.release(x, y, 1)
}

// drag picks a piece up, carries it away and drops it on a square.
func (g *testGUI) drag(from, to string) {
	x0, y0 := sqXY(from)
	x1, y1 := sqXY(to)
	g.press(x0, y0, 1)
	g.motion(x0+45, y0-70)
	g.motion(x1, y1)
	g.release(x1, y1, 1)
}

// reply lets the stub engine answer and returns everything printed so far.
func (g *testGUI) reply() string {
	g.t.Helper()
	select {
	case m := <-g.engCh:
		g.engineMoved(m)
	case <-time.After(5 * time.Second):
		g.t.Fatal("the engine was not started")
	}
	return g.out.String()
}

// idle checks that nothing was played and nothing is selected or pending.
func (g *testGUI) idle(what string) {
	g.t.Helper()
	if g.sel != -1 || g.dragFrom != -1 || g.promoFrom != -1 || g.thinking || g.out.Len() != 0 {
		g.t.Fatalf("%s: sel=%d drag=%d promo=%d thinking=%v out=%q status=%q",
			what, g.sel, g.dragFrom, g.promoFrom, g.thinking, g.out.String(), g.status)
	}
}

func TestGUIClickClickAndDeselect(t *testing.T) {
	g := newTestGUI(t, "", guiOpts{}, false)
	start := toFEN(g.p)

	g.click("c1") // the bishop of the request: select it ...
	if g.sel != sqOf("c1") {
		t.Fatalf("c1 not selected: sel=%d", g.sel)
	}
	g.click("c1") // ... and click it again to put it down
	g.idle("second click on the selected piece")

	g.click("c1")
	g.click("g1") // another own piece takes over the selection
	if g.sel != sqOf("g1") || g.out.Len() != 0 {
		t.Fatalf("selection did not move to g1: sel=%d out=%q", g.sel, g.out.String())
	}
	g.click("g5") // not a knight move
	if g.status != "Illegal move." {
		t.Fatalf("status %q", g.status)
	}
	g.idle("illegal target")
	if toFEN(g.p) != start {
		t.Fatalf("position changed: %s", toFEN(g.p))
	}

	g.click("g1")
	g.click("f3")
	if g.out.String() != "g1f3\n" || !g.thinking || g.sel != -1 {
		t.Fatalf("g1f3 not played: out=%q thinking=%v sel=%d", g.out.String(), g.thinking, g.sel)
	}
	g.click("e2") // no moves while the engine thinks
	if g.sel != -1 {
		t.Fatal("piece selected while the engine is thinking")
	}
	if out := g.reply(); !strings.HasPrefix(out, "g1f3\n") || strings.Count(out, "\n") != 2 || g.p.side != White {
		t.Fatalf("after the reply: out=%q", out)
	}
}

func TestGUIDragAndDrop(t *testing.T) {
	g := newTestGUI(t, "", guiOpts{}, false)
	start := toFEN(g.p)

	g.drag("e2", "e2") // carried around and put back on its square
	g.idle("dropped back")
	g.drag("e2", "e5") // illegal drop
	g.idle("illegal drop")
	x, y := sqXY("e2")
	g.press(x, y, 1)
	g.motion(x, guiBW+20)
	g.release(x, guiBW+20, 1) // dropped below the board
	g.idle("dropped outside the board")
	if toFEN(g.p) != start {
		t.Fatalf("position changed: %s", toFEN(g.p))
	}

	g.press(x, y, 1) // a click with a shaky hand is still a click
	g.motion(x+2, y-2)
	g.release(x+2, y-2, 1)
	if g.sel != sqOf("e2") || g.dragFrom != -1 || g.out.Len() != 0 {
		t.Fatalf("shaky click: sel=%d drag=%d out=%q", g.sel, g.dragFrom, g.out.String())
	}
	g.drag("d2", "d4") // dragging another piece overrides the selection
	if g.out.String() != "d2d4\n" || g.sel != -1 || !g.thinking {
		t.Fatalf("d2d4 not played: out=%q sel=%d", g.out.String(), g.sel)
	}
}

func TestGUIDragRendering(t *testing.T) {
	for _, icons := range []bool{false, true} {
		g := newTestGUI(t, "", guiOpts{icons: icons}, false)
		g.render()
		still := append([]uint8(nil), g.fb...)
		x0, y0 := sqXY("e2")
		x1, y1 := sqXY("e4")
		x1, y1 = x1+5, y1-7
		g.press(x0, y0, 1)
		g.motion(x1, y1)
		g.render()
		pawn := g.sprites[WP+6]
		for y := 0; y < guiSQ; y++ {
			for x := 0; x < guiSQ; x++ {
				a, want := pawn.a[y*guiSQ+x], pawn.g[y*guiSQ+x]
				if got := g.fb[(y1-guiSQ/2+y)*g.w+x1-guiSQ/2+x]; a == 255 && got != want {
					t.Fatalf("icons=%v: pawn not under the pointer at %d,%d: %d != %d", icons, x, y, got, want)
				}
				// its own square shows no pawn while it is carried (frame of 4 = selection)
				if x >= 4 && x < guiSQ-4 && y >= 4 && y < guiSQ-4 {
					if got := g.fb[(y0-guiSQ/2+y)*g.w+x0-guiSQ/2+x]; got != 255 {
						t.Fatalf("icons=%v: e2 not empty at %d,%d: %d", icons, x, y, got)
					}
				}
			}
		}
		g.motion(x0, y0)
		g.release(x0, y0, 1)
		g.render()
		if !bytes.Equal(g.fb[:guiBW*g.w], still[:guiBW*g.w]) {
			t.Fatalf("icons=%v: board differs after the piece was put back", icons)
		}
	}
}

func TestGUIPromotion(t *testing.T) {
	const fen = "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1"
	g := newTestGUI(t, fen, guiOpts{}, false)
	g.click("b7")
	g.click("b8")
	if g.promoFrom != sqOf("b7") || g.promoTo != sqOf("b8") || g.out.Len() != 0 {
		t.Fatalf("no promotion chooser: %d %d %q", g.promoFrom, g.promoTo, g.out.String())
	}
	g.click("h1") // outside the chooser: the move is taken back
	g.idle("promotion cancelled")

	g.drag("b7", "b8")
	if g.promoFrom != sqOf("b7") {
		t.Fatal("no promotion chooser after a drop")
	}
	g.press(3*guiSQ+10, 4*guiSQ, 1) // second cell of the chooser: rook
	if g.out.String() != "b7b8r\ncheck\n" || g.p.b[sqOf("b8")] != WR || g.promoFrom != -1 {
		t.Fatalf("rook promotion: out=%q b8=%d", g.out.String(), g.p.b[sqOf("b8")])
	}
}

func TestGUIButtons(t *testing.T) {
	g := newTestGUI(t, "", guiOpts{}, false)
	x, y := sqXY("e2")
	for _, b := range []int{2, 4, 5} { // middle button and the wheel do nothing
		g.press(x, y, b)
		g.release(x, y, b)
	}
	g.idle("other buttons")
	g.click("e2")
	g.press(x, y, 3) // right click drops the selection
	g.release(x, y, 3)
	g.idle("right click")

	bx, by := g.w/2, guiBW+guiStatusH+guiBtnH/2
	if g.h != guiBW+guiStatusH+guiBtnH || !g.inBtn(bx, by) {
		t.Fatalf("no CPU button: h=%d", g.h)
	}
	g.press(bx, by, 1)
	g.release(bx, by, 1)
	if !g.thinking || g.pendingCPU != 2 {
		t.Fatalf("CPU button: thinking=%v pending=%d", g.thinking, g.pendingCPU)
	}
	g.reply()
	if out := g.reply(); strings.Count(out, "\n") != 2 || g.p.side != White || g.thinking {
		t.Fatalf("CPU button must play both moves: %q", out)
	}

	h := newTestGUI(t, "", guiOpts{noCPU: true}, false)
	if h.h != guiBW+guiStatusH || h.hasBtn() || h.inBtn(bx, by) {
		t.Fatalf("hidden CPU button: h=%d", h.h)
	}
	h.press(bx, by, 1)
	h.release(bx, by, 1)
	h.idle("hidden CPU button")
}

func TestGUIInputModes(t *testing.T) {
	x0, y0 := sqXY("e2")
	x1, y1 := sqXY("e4")

	c := newTestGUI(t, "", guiOpts{input: inputClick}, false)
	c.drag("e2", "e4") // click mode: the press selects, nothing is carried
	if c.sel != sqOf("e2") || c.dragFrom != -1 || c.out.Len() != 0 {
		t.Fatalf("click mode: sel=%d drag=%d out=%q", c.sel, c.dragFrom, c.out.String())
	}
	c.click("e2")
	c.idle("click mode deselect")
	c.click("e2")
	c.click("e4")
	if c.out.String() != "e2e4\n" {
		t.Fatalf("click mode: %q", c.out.String())
	}

	d := newTestGUI(t, "", guiOpts{input: inputDrag}, false)
	d.click("e2") // drag mode: a click leaves nothing selected
	d.idle("drag mode click")
	d.press(x1, y1, 1)
	d.release(x1, y1, 1)
	d.idle("drag mode click on an empty square")
	d.press(x0, y0, 1)
	d.motion(x1, y1)
	d.release(x1, y1, 1)
	if d.out.String() != "e2e4\n" {
		t.Fatalf("drag mode: %q", d.out.String())
	}

	demo := newTestGUI(t, "", guiOpts{}, true)
	demo.drag("a7", "a6")
	demo.click("a7")
	if demo.sel != -1 || demo.dragFrom != -1 || demo.hasBtn() {
		t.Fatal("demo mode must ignore the mouse")
	}
}

// The engine goroutine must search a copy: it makes and unmakes moves while
// the GUI goroutine draws the position.
func TestGUIEngineGetsACopy(t *testing.T) {
	g := newTestGUI(t, "", guiOpts{}, false)
	seen := make(chan *Pos, 1)
	g.think = func(q *Pos) Move {
		m := q.genLegal()[0]
		seen <- q
		return m
	}
	g.drag("e2", "e4")
	if q := <-seen; q == g.p || toFEN(q) != toFEN(g.p) {
		t.Fatal("the engine must get its own copy of the position")
	}
	g.reply()
}

func TestIconSprites(t *testing.T) {
	var masks []string
	for pt := WP; pt <= WK; pt++ {
		w, b, l := iconSprite(pt, guiSQ), iconSprite(-pt, guiSQ), letterSprite(pt, guiSQ)
		solid, white, dark := 0, 0, 0
		mask := make([]byte, guiSQ*guiSQ)
		for i, a := range w.a {
			x, y := i%guiSQ, i/guiSQ
			if a != 0 && (x < 2 || y < 2 || x >= guiSQ-2 || y >= guiSQ-2) {
				t.Fatalf("piece %d touches the edge of its square at %d,%d", pt, x, y)
			}
			if a != b.a[i] {
				t.Fatalf("piece %d: white and black outlines differ", pt)
			}
			if a == 255 {
				mask[i] = 1
				solid++
				if w.g[i] > 200 {
					white++
				}
				if b.g[i] < 50 {
					dark++
				}
			}
		}
		// White is white inside its black outline, Black is black
		if solid < 600 || white*4 < solid || dark*4 < solid*3 {
			t.Fatalf("piece %d: solid=%d white=%d dark=%d", pt, solid, white, dark)
		}
		if bytes.Equal(w.g, l.g) && bytes.Equal(w.a, l.a) {
			t.Fatalf("piece %d: icon equals the letter badge", pt)
		}
		masks = append(masks, string(mask))
	}
	for i := range masks {
		for j := 0; j < i; j++ {
			if masks[i] == masks[j] {
				t.Fatalf("pieces %d and %d look the same", j+1, i+1)
			}
		}
	}
}

// ---------- rules ----------

// The en passant square is part of a position only while the capture is
// legal: the position after a double step repeats later without it.
func TestRepetitionAndEnPassant(t *testing.T) {
	cases := []struct {
		name, fen, moves string
		drawAt           int
	}{
		{"no pawn to capture", "", "e2e4 g8f6 g1f3 f6g8 f3g1 g8f6 g1f3 f6g8 f3g1", 9},
		{"capture possible", "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1",
			"e2e4 e8d8 e1d1 d8e8 d1e1 e8d8 e1d1 d8e8 d1e1 e8d8", 10},
		{"capturing pawn pinned", "8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 1",
			"e2e4 a4a5 e1f1 a5a4 f1e1 a4a5 e1f1 a5a4 f1e1", 9},
	}
	for _, c := range cases {
		p := startPos()
		if c.fen != "" {
			p = fromFEN(c.fen)
		}
		for i, s := range strings.Fields(c.moves) {
			m, msg := parseMove(p, s, p.genLegal())
			if msg != "" {
				t.Fatalf("%s: %s: %s", c.name, s, msg)
			}
			p.make(m)
			res, why := gameResult(p)
			if want := i+1 == c.drawAt; want != (res+" "+why == "1/2-1/2 repetition") || (!want && res != "") {
				t.Fatalf("%s: after %d plies: %q %q", c.name, i+1, res, why)
			}
		}
	}
}

func TestFENClockAndEnPassant(t *testing.T) {
	p := fromFEN("4k3/8/8/8/8/8/4P3/4K3 w - - 98 60")
	for i, s := range []string{"e1d1", "e8d8"} {
		m, _ := parseMove(p, s, p.genLegal())
		p.make(m)
		if res, why := gameResult(p); (i == 1) != (res+" "+why == "1/2-1/2 fifty-move") {
			t.Fatalf("after %s: %q %q (clock %d)", s, res, why, p.half)
		}
	}
	q := startPos()
	m, _ := parseMove(q, "e2e4", q.genLegal())
	q.make(m)
	p = fromFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
	if p.ep != -1 || q.ep != -1 || p.hash != q.hash {
		t.Fatalf("e3 without a capturing pawn: ep %d %d, hash %x %x", p.ep, q.ep, p.hash, q.hash)
	}
	p = fromFEN("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
	if p.ep != sqOf("e3") || p.hash != p.computeHash() {
		t.Fatalf("e3 with a capturing pawn: ep %d", p.ep)
	}
	if _, msg := parseMove(p, "d4e3", p.genLegal()); msg != "" {
		t.Fatalf("d4e3: %s", msg)
	}
}

// ---------- search ----------

func TestSearchFindsMates(t *testing.T) {
	for fen, want := range map[string]string{
		"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1":                             "a1a8", // mate in 1
		"r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w - - 0 1": "d2h6", // queen sacrifice, mate in 2
		"7k/5Q2/6K1/8/8/8/8/8 b - - 0 1":                                   "",     // stalemate: no move
	} {
		got := ""
		if m := bestMove(fromFEN(fen), 0, 5); m != NoMove {
			got = moveStr(m)
		}
		if got != want {
			t.Errorf("%s: played %q, want %q", fen, got, want)
		}
	}
}

// Positions nobody can win are draws for the search too, whatever material
// the evaluation counts.
func TestSearchKnowsDeadDraws(t *testing.T) {
	for fen, dead := range map[string]bool{
		"8/8/4k3/8/8/2B5/8/4K3 w - - 0 1":   true,  // K+B vs K
		"8/8/4k3/8/8/2N5/8/4K3 b - - 0 1":   true,  // K+N vs K
		"8/4b3/4k3/8/8/2B5/8/4K3 w - - 0 1": true,  // bishops on squares of one color
		"8/5b2/4k3/8/8/2B5/8/4K3 w - - 0 1": false, // bishops on both colors
		"8/8/4k3/8/8/2R5/8/4K3 w - - 0 1":   false, // K+R vs K
	} {
		e := &Engine{p: fromFEN(fen), ctl: &searchCtl{}}
		full, quiet := e.search(3, 1, -Inf, Inf, true), e.quiesce(-Inf, Inf, 1)
		if dead != (full == 0 && quiet == 0) || dead != insufficientMaterial(e.p) {
			t.Errorf("%s: search %d, quiescence %d, dead %v", fen, full, quiet, dead)
		}
	}
}

// The quiescence search tries captures only: a king left without a move
// must not look lost to it, or a search of depth 1 stalemates bare kings.
func TestStalemateSeen(t *testing.T) {
	for fen, want := range map[string]bool{
		"7k/5Q2/6K1/8/8/8/8/8 b - - 0 1":    true,
		"7k/8/6K1/8/8/8/8/5Q2 b - - 0 1":    false,
		"k7/P7/K7/8/8/8/8/8 b - - 0 1":      true,  // the pawn is defended
		"k7/1R6/2K5/8/8/8/8/8 b - - 0 1":    true,  // so is the rook
		"k7/1R6/8/8/8/8/7P/7K b - - 0 1":    false, // this one is not
		"7k/5Q1p/7P/8/8/8/8/6K1 b - - 0 1":  true,  // a blocked pawn
		"7k/5Q1p/8/7P/8/8/8/6K1 b - - 0 1":  false, // a pawn that moves
		"7k/5Q1p/6BP/8/8/8/8/6K1 b - - 0 1": false, // a pawn that takes
		"8/8/8/8/8/5k2/5p2/5K2 w - - 0 1":   true,
	} {
		e := &Engine{p: fromFEN(fen), ctl: &searchCtl{}}
		if got, legal := e.p.stalemated(), len(e.p.genLegal()); got != want || want != (legal == 0) {
			t.Errorf("%s: stalemated %v, want %v, %d legal moves", fen, got, want, legal)
		}
		if q := e.quiesce(-Inf, Inf, 1); want != (q == 0) {
			t.Errorf("%s: quiescence %d, stalemate %v", fen, q, want)
		}
	}
	// a bare king against king, queen and rook anywhere: exact
	rng := rand.New(rand.NewSource(5))
	for n, stale := 0, 0; n < 20000; {
		sq := rng.Perm(64)[:4]
		p := &Pos{ep: -1, side: Black}
		p.set(BK, sq[0])
		p.set(WK, sq[1])
		p.set(WQ, sq[2])
		p.set(WR, sq[3])
		p.kingSq = [2]int{sq[1], sq[0]}
		if chebDist(sq[0], sq[1]) < 2 || p.inCheck(Black) {
			continue
		}
		p.hash = p.computeHash()
		p.hist = append(p.hist, p.hash)
		got, want := p.stalemated(), len(p.genLegal()) == 0
		if got != want {
			t.Fatalf("%s: stalemated %v, want %v", toFEN(p), got, want)
		}
		if n, stale = n+1, stale+b2i(want); n == 20000 && stale < 100 {
			t.Errorf("only %d stalemates in %d positions", stale, n)
		}
	}
	// depth 1 used to play Kg3 here
	fen := "8/8/8/8/P4K2/5Q2/8/6k1 w - - 19 71"
	p := fromFEN(fen)
	p.make(bestMove(fromFEN(fen), 0, 1))
	if len(p.genLegal()) == 0 && !p.inCheck(p.side) {
		t.Errorf("%s: stalemated the bare king at depth 1", fen)
	}
}

// For the search a position is a draw when it comes again within the lines
// it looks at, or for the third time in the game. Coming again once in the
// game is not, where the search goes on: the opponent, who missed a capture
// there, may see it now.
func TestSearchRepetition(t *testing.T) {
	play := func(p *Pos, moves string) {
		for _, s := range strings.Fields(moves) {
			m, msg := parseMove(p, s, p.genLegal())
			if msg != "" {
				t.Fatalf("%s: %s", s, msg)
			}
			p.make(m)
		}
	}
	p := startPos()
	e := &Engine{p: p}
	play(p, "g1f3 g8f6 f3g1 f6g8") // the start position again
	if a, b, c, d := e.isRepetition(1, 1), e.isRepetition(4, 1), e.isRepetition(5, 1), e.isRepetition(1, 0); a || b || !c || !d {
		t.Errorf("second occurrence, 1, 4, 5 plies into the search and 1 at its end: %v %v %v %v, want false false true true", a, b, c, d)
	}
	play(p, "g1f3 g8f6 f3g1 f6g8")
	if !e.isRepetition(1, 1) {
		t.Errorf("third occurrence: no draw")
	}
	p = fromFEN("8/8/8/6rk/5K2/8/8/6R1 w - - 0 1") // White could take the rook
	play(p, "g1c1 g5b5 c1g1")
	if got := moveStr(bestMove(p, 0, 6)); got == "b5g5" {
		t.Errorf("played %s, back into the position where the rook is lost", got)
	}
}

// Bishop and knight mate only in a corner of the bishop's color: the
// evaluation must want the bare king there, the same for both sides.
func TestBishopAndKnightCorner(t *testing.T) {
	right := fromFEN("k7/8/1K6/8/8/8/8/5BN1 w - - 0 1").eval() // light bishop, king on a8
	wrong := fromFEN("7k/8/6K1/8/8/8/8/5BN1 w - - 0 1").eval() // king on h8
	if right < wrong+100 {
		t.Errorf("white: right corner %d, wrong corner %d", right, wrong)
	}
	if mirror := fromFEN("5bn1/8/8/8/8/1k6/8/K7 b - - 0 1").eval(); mirror != right {
		t.Errorf("black: %d, white: %d", mirror, right)
	}
}

// The ordering works in place and in stages: it must keep every move and
// put the hash move, the good captures, the quiet moves and the losing
// captures in that order.
func TestMoveOrderingKeepsEveryMove(t *testing.T) {
	e := &Engine{}
	for _, fen := range []string{
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
		"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
	} {
		p := fromFEN(fen)
		e.p = p
		for capsOnly := 0; capsOnly < 2; capsOnly++ {
			gen := p.genPseudo(capsOnly == 1)
			for i, tt := range append([]Move{NoMove}, gen...) {
				e.history[p.side][tt.From()][tt.To()] = int32(i*37%200 - 100)
				moves := append([]Move(nil), gen...)
				e.orderMoves(moves, tt, 3)
				count := map[Move]int{}
				for _, m := range gen {
					count[m]++
				}
				stage := 0 // 0 hash move, 1 good captures, 2 quiet moves, 3 losing captures
				for j, m := range moves {
					count[m]--
					now := 2
					if m.Flags()&fCapture != 0 || m.Promo() != 0 {
						now = 1
						if m.Promo() == 0 && p.see(m) < 0 {
							now = 3
						}
					}
					if m == tt {
						if now = 0; j != 0 {
							t.Fatalf("%s: hash move %s at %d", fen, moveStr(tt), j)
						}
					}
					if now < stage {
						t.Fatalf("%s: %s comes after a move of a later stage", fen, moveStr(m))
					}
					stage = now
				}
				for m, n := range count {
					if n != 0 {
						t.Fatalf("%s: move %s lost or doubled", fen, moveStr(m))
					}
				}
			}
		}
	}
}

// The quiet moves left when the search prunes them: those that give check
// from the square they go to, in their order, then the moves that followed.
func TestChecksOnly(t *testing.T) {
	for _, fen := range []string{
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"6k1/5ppp/8/8/8/2B2N2/5PPP/R2Q2K1 w - - 0 1",
		"4k3/8/8/8/8/8/3PP3/RN1QKB1R w KQ - 0 1",
		"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
	} {
		p := fromFEN(fen)
		e := &Engine{p: p}
		all := p.genPseudo(false)
		qs, qe := e.orderCaptures(all)
		want := append([]Move(nil), all[:qs]...)
		for _, m := range all[qs:qe] {
			u := p.make(m) // the man now on m.To() attacks the king?
			to, k, occ := m.To(), uint(p.kingSq[p.side]), p.occ[White]|p.occ[Black]
			var att uint64
			switch abs(p.b[to]) {
			case WP:
				att = pawnAttBB[p.side^1][to]
			case WN:
				att = knightBB[to]
			case WB:
				att = bishopAtt(to, occ)
			case WR:
				att = rookAtt(to, occ)
			case WQ:
				att = bishopAtt(to, occ) | rookAtt(to, occ)
			}
			p.unmake(m, u)
			if att>>k&1 != 0 {
				want = append(want, m)
			}
		}
		nq := len(want)
		want = append(want, all[qe:]...)
		got, gqe := p.checksOnly(append([]Move(nil), all...), qs, qe)
		if gqe != nq || fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("%s: %d moves, checks end at %d; want %d, %d", fen, len(got), gqe, len(want), nq)
		}
	}
}

// ---------- evaluation network ----------

// nnPlain computes the output of the network without the tricks on words:
// every hidden unit is summed, clipped and weighted by itself.
func nnPlain(p *Pos) int {
	nw, nb := nn.nw, nn.nb
	lane := func(w []uint64, i int) int { return int(w[i/4] >> uint(16*(i&3)) & 0xFFFF) }
	bkt := min((bits.OnesCount64(p.occ[White]|p.occ[Black])-2)/4, nb-1)
	sum := int(nn.ob[bkt])
	for v := 0; v < 2; v++ { // the side to move, then the other side
		c := p.side ^ v
		feat := &nn.feat[c][p.kingSq[c]>>2&1]
		for u := 0; u < 4*nw; u++ {
			h := lane(nn.fb, u) - nnZero
			for sq, pc := range p.b {
				if pc != Empty {
					h += lane(nn.ft[int(feat[pieceIdx(pc)*64+sq])*nw:], u) - nnWZero
				}
			}
			if h >= 0 {
				sum += min(h>>2, 63) * (int(nn.ow[(bkt*2+v)*nw+u/4]>>uint(16*(3-u&3))&0xFFFF) - 128)
			}
		}
		for sq, pc := range p.b {
			if pc != Empty {
				sum += (1 - 2*v) * int(nn.pq[int(feat[pieceIdx(pc)*64+sq])*nb+bkt])
			}
		}
	}
	return sum * nn.k / 4096
}

// transformed returns p with every man on square f(sq); with swap the
// colors and the side to move change too.
func transformed(p *Pos, f func(int) int, swap bool) *Pos {
	q := &Pos{ep: -1, side: p.side}
	for sq, pc := range p.b {
		if pc == Empty {
			continue
		}
		if swap {
			pc = -pc
		}
		q.set(pc, f(sq))
		if pc == WK {
			q.kingSq[White] = f(sq)
		} else if pc == BK {
			q.kingSq[Black] = f(sq)
		}
	}
	if swap {
		q.side ^= 1
	}
	q.hash = q.computeHash()
	q.pawnKey = q.computePawnKey()
	q.hist = append(q.hist, q.hash)
	return q
}

// The hidden units kept up to date move by move, and only when an
// evaluation asks for them, are those computed from the board; the output
// computed on words is the plain sum; a position with the colors swapped
// or mirrored left to right has the same value.
func TestNetwork(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	var s, fresh nnStack
	nw, nb := nn.nw, nn.nb
	fresh.v, fresh.ps = make([]uint64, 2*nw), make([]int32, 2*nb)
	checks := 0
	var walk func(p *Pos, depth int)
	walk = func(p *Pos, depth int) {
		if rng.Intn(3) > 0 {
			j := s.cur(p)
			fresh.refresh(p, 0, White)
			fresh.refresh(p, 0, Black)
			for i, x := range fresh.v {
				if got := s.v[2*j*nw+i]; got != x || got&0x8000800080008000 != 0 {
					t.Fatalf("%s: hidden units %d: %016x, from the board %016x", toFEN(p), i, got, x)
				}
			}
			for i, x := range fresh.ps {
				if got := s.ps[2*j*nb+i]; got != x {
					t.Fatalf("%s: sum %d: %d, from the board %d", toFEN(p), i, got, x)
				}
			}
			if checks++; checks%50 == 0 {
				want := p.nnEval()
				if got := nnPlain(p); got != want {
					t.Fatalf("%s: plain sum %d, on words %d", toFEN(p), got, want)
				}
				flip := transformed(p, func(sq int) int { return sq ^ 56 }, true)
				mirror := transformed(p, func(sq int) int { return sq ^ 7 }, false)
				if a, b := flip.eval(), mirror.eval(); a != p.eval() || b != p.eval() {
					t.Fatalf("%s: %d, colors swapped %d, mirrored %d", toFEN(p), p.eval(), a, b)
				}
			}
		}
		if depth == 0 {
			return
		}
		legal := p.genLegal()
		for k := 0; k < 2 && len(legal) > 0; k++ {
			m := legal[rng.Intn(len(legal))]
			u := p.make(m)
			walk(p, depth-1)
			p.unmake(m, u)
		}
	}
	for g := 0; g < 200; g++ {
		p := startPos()
		for i := rng.Intn(150); i > 0; i-- {
			legal := p.genLegal()
			if len(legal) == 0 {
				break
			}
			p.make(legal[rng.Intn(len(legal))])
		}
		q := p.clone()
		q.nnAttach(&s)
		walk(q, 8)
		if s.at != 0 {
			t.Fatalf("stack not back at the root: %d", s.at)
		}
	}
	if checks < 50000 {
		t.Fatalf("only %d checks", checks)
	}
}

// The network counts the material.
func TestNetworkKnowsMaterial(t *testing.T) {
	for _, c := range []struct {
		fen    string
		lo, hi int
	}{
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -60, 100},
		{"rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 500, 2500},
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1", -600, -120},
		{"4k3/8/8/8/8/8/8/R3K3 b - - 0 1", -3000, -300},
	} {
		if v := fromFEN(c.fen).eval(); v < c.lo || v > c.hi {
			t.Errorf("%s: %d, want %d..%d", c.fen, v, c.lo, c.hi)
		}
	}
}

// King and pawn against king is known exactly; against a bare king more
// material is worth more, and what cannot be won is worth nothing much.
func TestKnownEndings(t *testing.T) {
	kpkOnce.Do(kpkInit)
	wins := 0
	for _, w := range kpkWins {
		wins += bits.OnesCount64(w)
	}
	if wins != 111282 {
		t.Errorf("king and pawn against king: %d won positions, want 111282", wins)
	}
	ev := func(fen string) int { return fromFEN(fen).eval() }
	for fen, won := range map[string]bool{
		"4k3/8/4K3/4P3/8/8/8/8 w - - 0 1": true, // king on the sixth rank in front of the pawn
		"4k3/8/4K3/4P3/8/8/8/8 b - - 0 1": true,
		"8/4k3/8/4K3/4P3/8/8/8 w - - 0 1": false, // Black has the opposition
		"8/4k3/8/4K3/4P3/8/8/8 b - - 0 1": true,
		"k7/8/1K6/P7/8/8/8/8 w - - 0 1":   false, // rook pawn
		"8/8/4k3/8/8/8/4P3/4K3 w - - 0 1": false, // the black king keeps the opposition
		"4k3/4P3/4K3/8/8/8/8/8 b - - 0 1": false, // stalemate
		"4k3/4P3/4K3/8/8/8/8/8 w - - 0 1": true,
		"8/8/8/8/8/2k5/2p5/2K5 b - - 0 1": true,
		"8/8/8/8/8/2k5/2p5/2K5 w - - 0 1": false, // stalemate
	} {
		if v := ev(fen); won != (v >= nnWon || v <= -nnWon) || !won && v != 0 {
			t.Errorf("%s: %d, won %v", fen, v, won)
		}
	}
	for _, c := range [][2]string{ // the second is worth more to White than the first
		{"k7/8/8/8/8/8/8/KNN5 w - - 0 1", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"},
		{"4k3/4P3/4K3/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"},
		{"4k3/8/8/8/8/8/8/1N2KB2 w - - 0 1", "4k3/8/8/8/8/8/P7/1N2KB2 w - - 0 1"},
		{"4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1"},
		{"4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"},
		{"4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "4k3/8/8/8/8/8/8/R2QK3 w - - 0 1"},
	} {
		if a, b := ev(c[0]), ev(c[1]); a >= b {
			t.Errorf("%s: %d, %s: %d", c[0], a, c[1], b)
		}
	}
	for _, fen := range []string{
		"k7/8/8/8/P7/8/P7/K7 w - - 0 1",  // two rook pawns, the king in the corner
		"7k/8/8/8/8/8/7P/5B1K w - - 0 1", // h-pawn and light bishop
		"k7/8/8/8/8/8/P7/K1B5 w - - 0 1", // a-pawn and dark bishop
	} {
		if v := ev(fen); v < 0 || v > 50 {
			t.Errorf("%s: %d, want a draw", fen, v)
		}
	}
	if v := ev("7k/8/8/8/8/8/7P/4B2K w - - 0 1"); v < nnWon { // the bishop controls h8
		t.Errorf("h-pawn and dark bishop: %d", v)
	}
	if a, b := ev("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"), ev("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"); a != -b {
		t.Errorf("rook against king: %d for White, %d for Black", a, b)
	}
}

// ---------- memory ----------

func TestMemorySizing(t *testing.T) {
	if got := freeBSDAvail(4096, 1000, 5000, 0); got != 1000*4096 {
		t.Errorf("FreeBSD without swap: %d", got)
	}
	if got := freeBSDAvail(4096, 1000, 5000, 1<<30); got != 6000*4096 {
		t.Errorf("FreeBSD with swap: %d", got)
	}
	const vmStat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                3620.
Pages active:                            228760.
Pages inactive:                          227629.
Pages speculative:                          274.
Pages throttled:                              0.
Pages wired down:                        143227.
Pages purgeable:                              4.
"Translation faults":                4395641948.
`
	if got := vmStatAvail(vmStat); got != (3620+227629+274+4)*16384 {
		t.Errorf("vm_stat: %d", got)
	}
	if got := vmStatAvail("no such command"); got != 0 {
		t.Errorf("bad vm_stat output: %d", got)
	}
	const MB, GB = uint64(1) << 20, uint64(1) << 30
	cases := []struct {
		avail   uint64
		limit   time.Duration
		threads int
		want    uint64
	}{
		{40 * GB, 0, 16, 32 * GB},                     // depth and RAM modes: 80%
		{40 * GB, 600 * time.Second, 16, 32 * GB},     // long limit: 80%
		{40 * GB, 5 * time.Second, 16, 3840 * MB},     // what 16 threads fill in 5 s
		{40 * GB, 100 * time.Millisecond, 1, 64 * MB}, // floor
		{10 * MB, 0, 16, 64 * MB},
		{1000 * GB, 0, 16, 64 * GB}, // cap
	}
	for _, c := range cases {
		if got := hashBytes(c.avail, c.limit, c.threads); got != c.want {
			t.Errorf("hashBytes(%d MB, %v, %d) = %d MB, want %d MB", c.avail/MB, c.limit, c.threads, got/MB, c.want/MB)
		}
	}
}

// ---------- command line ----------

func TestParseArgs(t *testing.T) {
	env := func(kv ...string) func(string) string {
		return func(k string) string {
			for i := 0; i < len(kv); i += 2 {
				if kv[i] == k {
					return kv[i+1]
				}
			}
			return ""
		}
	}
	type want struct {
		budget   time.Duration
		depth    int
		gui      bool
		demo     bool
		silent   bool
		icons    bool
		noCPU    bool
		input    int
		help, ok bool
	}
	cases := []struct {
		args string
		env  func(string) string
		want want
	}{
		{"", env(), want{budget: 600 * time.Second, ok: true}},
		{"5", env(), want{budget: 5 * time.Second, ok: true}},
		{"0.25", env(), want{budget: 250 * time.Millisecond, ok: true}},
		{"-10", env(), want{depth: 10, ok: true}},
		{"0", env(), want{ok: true}},
		{"demo 1", env(), want{budget: time.Second, demo: true, ok: true}},
		{"SDEMO -3", env(), want{depth: 3, demo: true, silent: true, ok: true}},
		{"demo gui 2", env(), want{budget: 2 * time.Second, demo: true, gui: true, ok: true}},
		{"gui --icons 5", env(), want{budget: 5 * time.Second, gui: true, icons: true, ok: true}},
		{"gui --no-cpu 5", env(), want{budget: 5 * time.Second, gui: true, noCPU: true, ok: true}},
		{"gui 5 --icons --no-cpu --input=drag", env(),
			want{budget: 5 * time.Second, gui: true, icons: true, noCPU: true, input: inputDrag, ok: true}},
		{"gui 5", env("CHESS_ICONS", "1", "CHESS_NO_CPU", "1", "CHESS_INPUT", "click"),
			want{budget: 5 * time.Second, gui: true, icons: true, noCPU: true, input: inputClick, ok: true}},
		{"gui --input=both", env("CHESS_ICONS", "0", "CHESS_INPUT", "drag"),
			want{budget: 600 * time.Second, gui: true, ok: true}},
		{"-h", env(), want{budget: 600 * time.Second, help: true, ok: true}},
		{"bogus --help", env(), want{budget: 600 * time.Second, help: true}},
		{"1 2", env(), want{budget: time.Second}},
		{"-1.5", env(), want{budget: 600 * time.Second}},
		{"gui --input=mouse", env(), want{budget: 600 * time.Second, gui: true}},
		{"--bogus", env(), want{budget: 600 * time.Second}},
	}
	for _, c := range cases {
		cfg, err := parseArgs(strings.Fields(c.args), c.env)
		got := want{cfg.budget, cfg.depthCap, cfg.gui, cfg.demo, cfg.silent, cfg.opt.icons,
			cfg.opt.noCPU, cfg.opt.input, cfg.help, err == nil}
		if got != c.want {
			t.Errorf("%q: got %+v (err %v), want %+v", c.args, got, err, c.want)
		}
	}
	for _, s := range []string{"chess gui --icons 5", "chess gui --no-cpu 5",
		"chess gui --icons --no-cpu 5", "CHESS_ICONS=1 CHESS_NO_CPU=1 chess gui 5"} {
		if !strings.Contains(usageText, "\n  "+s+" ") {
			t.Errorf("the help has no example %q", s)
		}
	}
}

// ---------- X11 end to end: the real program on a private Xvfb ----------

func findXvfb() string {
	if p, err := exec.LookPath("Xvfb"); err == nil {
		return p
	}
	for _, p := range []string{"/opt/X11/bin/Xvfb", "/usr/local/bin/Xvfb", "/usr/X11R6/bin/Xvfb"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// startXvfb runs a private X server with the given root depth and returns
// its display name (":N"). The number is one nobody uses - neither its lock
// nor its socket exists - and is given explicitly: a server left to pick a
// number itself may take over the socket of a running one. It is stopped
// with a signal it handles, so that it removes its socket and lock.
func startXvfb(t *testing.T, depth int) string {
	bin := findXvfb()
	if bin == "" {
		t.Skip("no Xvfb on this machine")
	}
	for n := 90; n < 200; n++ {
		display := fmt.Sprintf(":%d", n)
		sock := fmt.Sprintf("/tmp/.X11-unix/X%d", n)
		if _, err := os.Lstat(sock); err == nil {
			continue
		}
		if _, err := os.Lstat(fmt.Sprintf("/tmp/.X%d-lock", n)); err == nil {
			continue
		}
		cmd := exec.Command(bin, display, "-screen", "0", fmt.Sprintf("800x700x%d", depth),
			"-nolisten", "tcp", "-noreset")
		if err := cmd.Start(); err != nil {
			t.Skipf("cannot start %s: %v", bin, err)
		}
		gone := make(chan struct{})
		go func() {
			cmd.Wait()
			close(gone)
		}()
		t.Cleanup(func() {
			cmd.Process.Signal(os.Interrupt)
			select {
			case <-gone:
			case <-time.After(10 * time.Second):
				cmd.Process.Kill()
				<-gone
			}
		})
		for deadline := time.Now().Add(20 * time.Second); time.Now().Before(deadline); {
			select {
			case <-gone: // lost the race for this number: try the next one
				deadline = time.Time{}
			case <-time.After(50 * time.Millisecond):
				if c, err := net.Dial("unix", sock); err == nil {
					c.Close()
					return display
				}
			}
		}
	}
	t.Skipf("%s did not start", bin)
	return ""
}

// chessProc is the program under test with its stdout lines.
type chessProc struct {
	t     *testing.T
	lines chan string
}

// startChess runs the program: the test binary itself or, to test a built
// one, the binary named by CHESS_TEST_BINARY.
func startChess(t *testing.T, display, args string, env ...string) *chessProc {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe)
	if bin := os.Getenv("CHESS_TEST_BINARY"); bin != "" {
		cmd = exec.Command(bin, strings.Fields(args)...)
	}
	cmd.Env = append(os.Environ(), "CHESS_TEST_MAIN=1", "CHESS_TEST_ARGS="+args, "DISPLAY="+display,
		"CHESS_HASH_MB=16", "CHESS_THREADS=1")
	cmd.Env = append(cmd.Env, env...)
	cmd.Stderr = os.Stderr
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	c := &chessProc{t, make(chan string, 64)}
	go func() {
		sc := bufio.NewScanner(out)
		for sc.Scan() {
			c.lines <- sc.Text()
		}
		close(c.lines)
	}()
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})
	return c
}

// move returns the next move the program printed.
func (c *chessProc) move() string {
	c.t.Helper()
	for {
		if s := c.line(); s != "check" {
			return s
		}
	}
}

func (c *chessProc) line() string {
	c.t.Helper()
	select {
	case s, ok := <-c.lines:
		if !ok {
			c.t.Fatal("the program ended")
		}
		return s
	case <-time.After(20 * time.Second):
		c.t.Fatal("no output from the program")
	}
	return ""
}

// xClient is the test's own connection to the X server: it finds the chess
// window, moves the real pointer with XTEST and reads the window back.
type xClient struct {
	t     *testing.T
	x     *x11Conn
	xtest byte
	win   uint32
	w, h  int
	lut   [256]uint32
}

func (c *xClient) call(b []byte) []byte {
	c.t.Helper()
	r, ok := c.x.roundTrip(b)
	if !ok {
		c.t.Fatalf("X request %d failed", b[0])
	}
	return r
}

func newXClient(t *testing.T, display string) *xClient {
	t.Setenv("DISPLAY", display)
	var x *x11Conn
	var err error
	for i := 0; i < 100; i++ {
		if x, err = x11Dial(); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("cannot connect to %s: %v", display, err)
	}
	t.Cleanup(func() { x.c.Close() })
	c := &xClient{t: t, x: x, lut: x.grayLUT()}
	name := "XTEST"
	b := make([]byte, 8+len(name)+pad4(len(name)))
	b[0] = 98 // QueryExtension
	le16(b[2:], uint16(len(b)/4))
	le16(b[4:], uint16(len(name)))
	copy(b[8:], name)
	if r := c.call(b); r[8] == 1 {
		c.xtest = r[9]
	} else {
		t.Skip("the X server has no XTEST extension")
	}
	return c
}

// findWindow waits for the chess window to appear on the root.
func (c *xClient) findWindow() {
	c.t.Helper()
	for deadline := time.Now().Add(20 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		b := make([]byte, 8)
		b[0] = 15 // QueryTree
		le16(b[2:], 2)
		le32(b[4:], c.x.root)
		r := c.call(b)
		for i := 0; i < int(rd16(r[16:])); i++ {
			w := rd32(r[32+4*i:])
			b = make([]byte, 24)
			b[0] = 20 // GetProperty WM_NAME
			le16(b[2:], 6)
			le32(b[4:], w)
			le32(b[8:], 39)
			le32(b[12:], 31)
			le32(b[20:], 64)
			p, ok := c.x.roundTrip(b)
			if !ok || string(p[32:32+int(rd32(p[16:]))]) != "chess-fable" {
				continue
			}
			b = make([]byte, 8)
			b[0] = 14 // GetGeometry
			le16(b[2:], 2)
			le32(b[4:], w)
			geo := c.call(b)
			c.win, c.w, c.h = w, int(rd16(geo[16:])), int(rd16(geo[18:]))
			return
		}
	}
	c.t.Fatal("the chess window did not appear")
}

// fake injects real pointer input: typ 4 press, 5 release, 6 motion.
func (c *xClient) fake(typ, button byte, px, py int) {
	b := make([]byte, 36)
	b[0], b[1] = c.xtest, 2 // XTestFakeInput
	le16(b[2:], 9)
	b[4], b[5] = typ, button
	le32(b[12:], c.x.root)
	le16(b[24:], uint16(px))
	le16(b[26:], uint16(py))
	c.x.send(b)
	sync := []byte{43, 0, 1, 0} // GetInputFocus
	c.call(sync)
}

func (c *xClient) move(px, py int)  { c.fake(6, 0, px, py) }
func (c *xClient) down(px, py int)  { c.move(px, py); c.fake(4, 1, 0, 0) }
func (c *xClient) up(px, py int)    { c.move(px, py); c.fake(5, 1, 0, 0) }
func (c *xClient) click(px, py int) { c.down(px, py); c.up(px, py) }

// pixels reads the window back as pixel values of the root visual.
func (c *xClient) pixels() []uint32 {
	c.t.Helper()
	b := make([]byte, 20)
	b[0], b[1] = 73, 2 // GetImage, ZPixmap
	le16(b[2:], 5)
	le32(b[4:], c.win)
	le16(b[12:], uint16(c.w))
	le16(b[14:], uint16(c.h))
	le32(b[16:], 0xffffffff)
	data := c.call(b)[32:]
	x := c.x
	n := x.bpp / 8
	row := (c.w*x.bpp + x.pad - 1) / x.pad * x.pad / 8
	px := make([]uint32, c.w*c.h)
	for y := 0; y < c.h; y++ {
		for i := 0; i < c.w; i++ {
			var v uint32
			for k := 0; k < n; k++ {
				sh := uint(8 * k)
				if x.msbFirst {
					sh = uint(8 * (n - 1 - k))
				}
				v |= uint32(data[y*row+i*n+k]) << sh
			}
			px[y*c.w+i] = v
		}
	}
	return px
}

// shows waits until the window shows exactly the frame of g.
func (c *xClient) shows(g *gui, what string) {
	c.t.Helper()
	if c.w != g.w || c.h != g.h {
		c.t.Fatalf("%s: window is %dx%d, want %dx%d", what, c.w, c.h, g.w, g.h)
	}
	g.render()
	bad, at := 0, 0
	for deadline := time.Now().Add(20 * time.Second); ; time.Sleep(30 * time.Millisecond) {
		bad = 0
		for i, v := range c.pixels() {
			if v != c.lut[g.fb[i]] {
				bad, at = bad+1, i
			}
		}
		if bad == 0 {
			return
		}
		if time.Now().After(deadline) {
			break
		}
	}
	c.t.Fatalf("%s: %d pixels differ from the expected frame (last at %d,%d)", what, bad, at%c.w, at/c.w)
}

// mirror plays the same mouse input into a GUI of the test, whose frames
// are what the window must show; the engine's replies come from stdout.
func mirror(t *testing.T, opt guiOpts) *gui {
	return newGUI(startPos(), opt, false, func(*Pos) Move { return NoMove }, &bytes.Buffer{})
}

func mirrorReply(t *testing.T, g *gui, c *chessProc, human string) {
	t.Helper()
	if s := c.move(); s != human {
		t.Fatalf("program printed %q, want %q", s, human)
	}
	<-g.engCh
	m, msg := parseMove(g.p, c.move(), g.p.genLegal())
	if msg != "" {
		t.Fatalf("bad engine reply: %s", msg)
	}
	g.engineMoved(m)
}

func TestX11EndToEnd(t *testing.T) {
	display := startXvfb(t, 24)
	auth := "XAUTHORITY=" + filepath.Join(t.TempDir(), "none")
	t.Setenv("XAUTHORITY", auth[len("XAUTHORITY="):])
	prog := startChess(t, display, "gui --icons -1", auth)
	c := newXClient(t, display)
	c.findWindow()
	g := mirror(t, guiOpts{icons: true})
	c.shows(g, "start position")

	// drag & drop with the real pointer: e2 is carried to e4
	x0, y0 := sqXY("e2")
	x1, y1 := sqXY("e4")
	c.down(x0, y0)
	c.move(x0+3, y0-40)
	c.move(x1+9, y1+6)
	g.press(x0, y0, 1)
	g.motion(x1+9, y1+6)
	c.shows(g, "pawn carried by the pointer")
	c.up(x1+9, y1+6)
	g.release(x1+9, y1+6, 1)
	mirrorReply(t, g, prog, "e2e4")
	c.shows(g, "after e2e4 and the reply")

	// a piece dropped back on its square is no move
	x0, y0 = sqXY("d2")
	c.down(x0, y0)
	c.move(x0+70, y0-90)
	c.up(x0+1, y0+1)
	g.press(x0, y0, 1)
	g.motion(x0+70, y0-90)
	g.release(x0+1, y0+1, 1)
	c.shows(g, "d2 dropped back")

	// click-click: select, deselect, select another piece, move it
	for _, sq := range []string{"f1", "f1", "f1", "g1"} {
		x, y := sqXY(sq)
		c.click(x, y)
		g.press(x, y, 1)
		g.release(x, y, 1)
		c.shows(g, "click on "+sq)
	}
	x, y := sqXY("f3")
	c.click(x, y)
	g.press(x, y, 1)
	g.release(x, y, 1)
	mirrorReply(t, g, prog, "g1f3")
	c.shows(g, "after g1f3 and the reply")

	// the CPU MOVE button plays for the human, then the engine answers
	bx, by := g.w/2, guiBW+guiStatusH+guiBtnH/2
	c.click(bx, by)
	g.press(bx, by, 1)
	g.release(bx, by, 1)
	for i := 0; i < 2; i++ {
		<-g.engCh
		m, msg := parseMove(g.p, prog.move(), g.p.genLegal())
		if msg != "" {
			t.Fatalf("CPU button move %d: %s", i, msg)
		}
		g.engineMoved(m)
	}
	c.shows(g, "after the CPU MOVE button")
}

// Options from the environment, a socket-path DISPLAY (as XQuartz sets it)
// and the 16-bit and 8-bit pixel formats.
func TestX11DisplaysAndDepths(t *testing.T) {
	for _, depth := range []int{16, 8} {
		display := startXvfb(t, depth)
		dir, err := os.MkdirTemp("/tmp", "chess-test")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })
		sock := filepath.Join(dir, "org.xquartz:0")
		if err := os.Symlink("/tmp/.X11-unix/X"+display[1:], sock); err != nil {
			t.Fatal(err)
		}
		auth := filepath.Join(dir, "none")
		t.Setenv("XAUTHORITY", auth)
		prog := startChess(t, sock, "gui -1", "CHESS_ICONS=1", "CHESS_NO_CPU=1", "XAUTHORITY="+auth)
		c := newXClient(t, sock)
		if c.x.trueColor != (depth == 16) {
			t.Fatalf("depth %d: trueColor=%v", depth, c.x.trueColor)
		}
		c.findWindow()
		g := mirror(t, guiOpts{icons: true, noCPU: true})
		c.shows(g, fmt.Sprintf("start position at depth %d", depth))
		x0, y0 := sqXY("e2")
		x1, y1 := sqXY("e4")
		c.click(x0, y0)
		c.click(x1, y1)
		g.press(x0, y0, 1)
		g.release(x0, y0, 1)
		g.press(x1, y1, 1)
		g.release(x1, y1, 1)
		mirrorReply(t, g, prog, "e2e4")
		c.shows(g, fmt.Sprintf("after e2e4 at depth %d", depth))

		// a socket path that leads nowhere: the display's own socket is used
		t.Setenv("DISPLAY", filepath.Join(dir, "gone", "org.xquartz"+display))
		x, err := x11Dial()
		if err != nil {
			t.Fatalf("no way to display %s found: %v", display, err)
		}
		x.c.Close()
	}
}
