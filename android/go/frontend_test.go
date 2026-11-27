package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// droidState is one "state" line of the protocol.
type droidState struct {
	board                           string
	side                            string
	sel, from, to, promo, held      int
	thinking, over, button, present bool
	status                          string
}

// droidGame is the app's side of a game: it writes commands and reads lines.
type droidGame struct {
	t     *testing.T
	in    *io.PipeWriter
	lines chan string
	done  chan int
	st    droidState
	moves []string
}

func droidFirstMove(q *Pos) Move { return q.genLegal()[0] }

func startDroid(t *testing.T, think func(*Pos) Move, args ...string) *droidGame {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	d := &droidGame{t: t, in: inW, lines: make(chan string, 4096), done: make(chan int, 1)}
	go func() {
		d.done <- droidRun(args, inR, outW, think)
		outW.Close()
	}()
	go func() {
		sc := bufio.NewScanner(outR)
		sc.Buffer(make([]byte, 1<<16), 1<<24)
		for sc.Scan() {
			d.lines <- sc.Text()
		}
		close(d.lines)
	}()
	t.Cleanup(func() {
		inW.Close()
		go func() { // the game's last lines must not block its exit
			for range d.lines {
			}
		}()
		select {
		case code := <-d.done:
			if code != 0 {
				t.Errorf("exit code %d at the end of stdin", code)
			}
		case <-time.After(10 * time.Second):
			t.Error("the game did not end with its stdin")
		}
	})
	return d
}

func (d *droidGame) send(format string, a ...any) {
	d.t.Helper()
	if _, err := fmt.Fprintf(d.in, format+"\n", a...); err != nil {
		d.t.Fatal(err)
	}
}

// line reads on until a line with that first word and returns the rest of it.
func (d *droidGame) line(word string) string {
	d.t.Helper()
	for {
		select {
		case s, ok := <-d.lines:
			if !ok {
				d.t.Fatalf("output ended, no %q line", word)
			}
			w, rest, _ := strings.Cut(s, " ")
			switch w {
			case "state":
				f := strings.SplitN(rest, " ", 11)
				if len(f) != 11 || len(f[0]) != 64 {
					d.t.Fatalf("bad state line %q", s)
				}
				n := func(i int) int { v, _ := strconv.Atoi(f[i]); return v }
				d.st = droidState{f[0], f[1], n(2), n(3), n(4), n(5), n(6), f[7] == "1", f[8] == "1", f[9] == "1", true, f[10]}
			case "moves":
				d.moves = strings.Fields(rest)
			}
			if w == word {
				return rest
			}
		case <-time.After(20 * time.Second):
			d.t.Fatalf("no %q line; state %+v moves %v", word, d.st, d.moves)
		}
	}
}

// wait reads on until the game is in the state that ok describes.
func (d *droidGame) wait(what string, ok func(s droidState) bool) droidState {
	d.t.Helper()
	for end := time.Now().Add(20 * time.Second); !d.st.present || !ok(d.st); d.line("state") {
		if time.Now().After(end) {
			d.t.Fatalf("%s: never; state %+v moves %v", what, d.st, d.moves)
		}
	}
	return d.st
}

func dsq(name string) int { return int(name[1]-'1')*8 + int(name[0]-'a') }

// dxy is the middle of a square in the gui's units.
func dxy(name string) (int, int) {
	sq := dsq(name)
	return (sq&7)*guiSQ + guiSQ/2, (7-sq>>3)*guiSQ + guiSQ/2
}

func (d *droidGame) tap(name string) {
	d.t.Helper()
	x, y := dxy(name)
	d.send("d %d %d", x, y)
	d.send("u %d %d", x, y)
}

func (d *droidGame) played(n int) droidState {
	d.t.Helper()
	return d.wait(fmt.Sprint(n, " moves played"), func(s droidState) bool { return len(d.moves) == n && !s.thinking })
}

func TestDroidBadStart(t *testing.T) {
	for _, c := range []struct {
		args []string
		want string
	}{
		{[]string{"x", "0", "0"}, `error bad limit "x"`},
		{[]string{"-0.5", "0", "0"}, `error bad limit "-0.5"`},
		{[]string{"demo", "0", "0"}, `error bad limit "demo"`},
		{[]string{"30"}, "error usage"},
	} {
		var out bytes.Buffer
		if code := droidRun(c.args, strings.NewReader(""), &out, nil); code != 2 || !strings.HasPrefix(out.String(), c.want) {
			t.Errorf("%v: exit %d, %q; want exit 2, %q", c.args, code, out.String(), c.want)
		}
	}
}

func TestDroidTapTapAndCancel(t *testing.T) {
	d := startDroid(t, droidFirstMove, "30", "0", "0")
	f := strings.Fields(d.line("ready"))
	if len(f) != 4 || f[1] != strconv.Itoa(guiSQ) {
		t.Fatalf("ready line %q", f)
	}
	s := d.wait("start", func(s droidState) bool { return true })
	if s.board[:16] != "RNBQKBNRPPPPPPPP" || s.side != "w" || s.button || s.thinking || s.over || s.sel != -1 {
		t.Fatalf("start: %+v", s)
	}
	d.tap("e2")
	d.wait("e2 selected", func(s droidState) bool { return s.sel == dsq("e2") })
	d.tap("e2") // the same piece again lets it go ...
	d.wait("e2 let go", func(s droidState) bool { return s.sel == -1 })
	d.tap("g1")
	d.wait("g1 selected", func(s droidState) bool { return s.sel == dsq("g1") })
	d.tap("d2") // ... and so does picking another one
	d.wait("d2 selected", func(s droidState) bool { return s.sel == dsq("d2") })
	if len(d.moves) != 0 {
		t.Fatalf("moves after selecting only: %v", d.moves)
	}
	d.tap("d4")
	s = d.played(2)
	if d.moves[0] != "d2d4" || s.board[dsq("d4")] != 'P' || s.board[dsq("d2")] != '.' || s.side != "w" || s.sel != -1 {
		t.Fatalf("after d2d4: moves %v state %+v", d.moves, s)
	}
	if want := d.moves[1]; sqName(s.from)+sqName(s.to) != want {
		t.Errorf("last move marks %s%s, want the reply %s", sqName(s.from), sqName(s.to), want)
	}
	d.tap("e5") // an empty square with nothing selected
	d.tap("a2")
	d.wait("a2 selected", func(s droidState) bool { return s.sel == dsq("a2") })
	d.tap("h8") // not a move of that pawn
	s = d.wait("illegal move refused", func(s droidState) bool { return s.sel == -1 })
	if s.status != "Illegal move." || len(d.moves) != 2 {
		t.Errorf("after an illegal move: %q, moves %v", s.status, d.moves)
	}
}

func TestDroidDragAndDrop(t *testing.T) {
	d := startDroid(t, droidFirstMove, "30", "0", "0")
	x, y := dxy("g1")
	d.send("d %d %d", x, y)
	d.send("m %d %d", x-20, y-50)
	d.wait("knight held", func(s droidState) bool { return s.held == dsq("g1") })
	x, y = dxy("f3")
	d.send("u %d %d", x, y)
	s := d.played(2)
	if d.moves[0] != "g1f3" || s.board[dsq("f3")] != 'N' || s.held != -1 {
		t.Fatalf("after the drag: moves %v state %+v", d.moves, s)
	}
	// put back where it was: nothing is played and another piece can be picked
	x, y = dxy("e2")
	d.send("d %d %d", x, y)
	d.send("m %d %d", x+40, y-90)
	d.wait("pawn held", func(s droidState) bool { return s.held == dsq("e2") })
	d.send("u %d %d", x+5, y-5)
	s = d.wait("pawn back", func(s droidState) bool { return s.held == -1 && s.sel == -1 })
	if s.status != "Move cancelled." || len(d.moves) != 2 {
		t.Errorf("after putting it back: %q, moves %v", s.status, d.moves)
	}
	// the gesture is taken away in mid-air
	x, y = dxy("d2")
	d.send("d %d %d", x, y)
	d.send("m %d %d", x, y-100)
	d.wait("pawn held", func(s droidState) bool { return s.held == dsq("d2") })
	d.send("cancel")
	d.wait("gesture cancelled", func(s droidState) bool { return s.held == -1 && s.sel == -1 })
	// dropped outside the board
	d.send("d %d %d", x, y)
	d.send("m %d %d", x, y-100)
	d.send("u %d %d", -30, guiBW+200)
	s = d.wait("dropped outside", func(s droidState) bool { return s.held == -1 && s.sel == -1 && s.status == "Move cancelled." })
	if len(d.moves) != 2 {
		t.Errorf("moves after cancelled drags: %v", d.moves)
	}
}

func TestDroidRestart(t *testing.T) {
	// the record has a damaged entry: the game goes on from the sound part
	d := startDroid(t, droidFirstMove, "-3", "1", "0", "e2e4", "e7e5", "g1f3", "zz", "b8c6")
	if got := d.line("moves"); got != "e2e4 e7e5 g1f3" {
		t.Fatalf("moves after the restart: %q", got)
	}
	s := d.wait("restored", func(s droidState) bool { return true })
	if s.board[dsq("e4")] != 'P' || s.board[dsq("e5")] != 'p' || s.board[dsq("f3")] != 'N' ||
		s.from != dsq("g1") || s.to != dsq("f3") || !s.button {
		t.Fatalf("restored: %+v", s)
	}
	s = d.played(4) // it was Black's turn: the engine goes on by itself
	if s.side != "w" {
		t.Errorf("after the engine's move: %+v", s)
	}
}

func TestDroidRestartFinishedGame(t *testing.T) {
	d := startDroid(t, droidFirstMove, "30", "0", "0", "f2f3", "e7e5", "g2g4", "d8h4", "a2a3")
	s := d.wait("mate shown", func(s droidState) bool { return s.over })
	if s.status != "0-1 checkmate" || len(d.moves) != 4 || s.thinking {
		t.Errorf("finished game: %+v moves %v", s, d.moves)
	}
}

func TestDroidCPUButton(t *testing.T) {
	d := startDroid(t, droidFirstMove, "5", "1", "0")
	d.wait("button shown", func(s droidState) bool { return s.button })
	d.send("cpu")
	d.played(2) // the engine moved for White and answered for Black

	d = startDroid(t, droidFirstMove, "5", "0", "0")
	d.send("cpu")
	d.tap("e2")
	s := d.wait("e2 selected", func(s droidState) bool { return s.sel == dsq("e2") })
	if s.button || len(d.moves) != 0 {
		t.Errorf("hidden button: %+v moves %v", s, d.moves)
	}
}

func TestDroidDemo(t *testing.T) {
	d := startDroid(t, droidFirstMove, "5", "1", "1")
	d.tap("e2")
	s := d.wait("demo plays", func(s droidState) bool { return len(d.moves) >= 6 })
	if s.button || s.sel != -1 {
		t.Errorf("demo: %+v", s)
	}
}

func TestDroidPromotion(t *testing.T) {
	t.Setenv("CHESS_FEN", "8/4P3/8/8/8/7p/k7/4K3 w - - 0 1")
	d := startDroid(t, droidFirstMove, "30", "0", "0")
	pick := func() {
		d.tap("e7")
		d.tap("e8")
		d.wait("chooser", func(s droidState) bool { return s.promo == dsq("e8") })
	}
	pick()
	d.tap("a1") // beside the chooser
	s := d.wait("chooser closed", func(s droidState) bool { return s.promo == -1 })
	if len(d.moves) != 0 || s.board[dsq("e7")] != 'P' {
		t.Fatalf("after cancelling the promotion: %+v moves %v", s, d.moves)
	}
	pick()
	d.send("d %d %d", 2*guiSQ+3*guiSQ+10, 3*guiSQ+guiSQ/2+10) // queen, rook, bishop, knight
	s = d.played(2)
	if d.moves[0] != "e7e8n" || s.board[dsq("e8")] != 'N' {
		t.Errorf("after the promotion: %+v moves %v", s, d.moves)
	}
}

func TestDroidSprites(t *testing.T) {
	d := startDroid(t, droidFirstMove, "30", "0", "0")
	dir := filepath.Join(t.TempDir(), "a b", "pieces-80")
	d.send("size 80 %s", dir)
	if got := d.line("sprites"); got != "80 "+dir {
		t.Fatalf("sprites line %q", got)
	}
	for _, name := range strings.Fields("wP wN wB wR wQ wK bP bN bB bR bQ bK") {
		f, err := os.Open(filepath.Join(dir, name+".png"))
		if err != nil {
			t.Fatal(err)
		}
		img, err := png.Decode(f)
		f.Close()
		if err != nil || img.Bounds().Dx() != 80 || img.Bounds().Dy() != 80 {
			t.Fatalf("%s.png: %v, %v", name, img.Bounds(), err)
		}
		// a white piece is white in the middle of its foot, a black one black, the corner is clear
		r, _, _, a := img.At(40, 65).RGBA()
		_, _, _, corner := img.At(0, 0).RGBA()
		if a != 0xffff || corner != 0 || (r > 0x8000) != (name[0] == 'w') {
			t.Errorf("%s.png: foot %04x/%04x, corner %04x", name, r, a, corner)
		}
	}
	if left, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(left) != 0 {
		t.Errorf("left behind: %v", left)
	}
}

func TestDroidEngine(t *testing.T) {
	d := startDroid(t, nil, "-2", "0", "0")
	d.tap("e2")
	d.tap("e4")
	d.played(2)
	p := startPos()
	for _, s := range d.moves {
		m, bad := parseMove(p, s, p.genLegal())
		if bad != "" {
			t.Fatalf("%v: %s is no legal move (%s)", d.moves, s, bad)
		}
		p.make(m)
	}
}
