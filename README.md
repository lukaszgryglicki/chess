# chess-fable — text-mode chess

A complete text-mode chess game in a single Go file (standard library only),
written by Claude Fable 5 (GitHub Copilot CLI) from the same spec
(`chess.md`) given to the local Qwen model — see `STATS.md` for the
head-to-head numbers.

## Build

    go build -o chess chess.go

(Any Go ≥ 1.21; no dependencies.) `make` builds `chess-<os>` for the machine
it runs on and `make all` cross-compiles the three binaries kept in the
repository: `chess-freebsd`, `chess-linux` (both amd64) and `chess-macos`
(arm64). `make test` runs the tests, `make install` copies the binary to
`$(INSTALL_DIR)/chess`.

## Usage

    ./chess [limit]
    ./chess demo [limit]         # self-playing demo, board after every move
    ./chess silentdemo [limit]   # same demo without boards (moves only)
    ./chess gui [limit]          # X11 board window (drag & drop), same engine
    ./chess gui demo [limit]     # X11 window, engine plays both sides
    ./chess gui --icons --no-cpu [limit]   # piece icons, no CPU MOVE button
    ./chess --help               # all options, with examples

- The single optional argument is the computer's per-move limit — there is
  always **exactly one limiting resource per mode**:
  - **positive** = **TIME-limited**: max think seconds per move (wall clock,
    enforced, checked every ~1024 nodes). The whole time is used: a deeper
    iteration cut off by the clock still counts when it has found a better
    move. Fractions are accepted (`0.5`). Depth and RAM are not limits here.
    **Default when omitted: 600.**
  - **negative** = **DEPTH-limited**: fixed search depth (`-1` = depth 1,
    `-10` = depth 10); no clock, no RAM criterion.
  - **`0`** = **RAM-limited**: no clock and no depth cap. The engine
    auto-sizes its hash table to ~80% of the RAM currently available on the
    machine (floor 64 MB, cap 64 GB; override with `CHESS_HASH_MB=<mb>`),
    and plays its move once the search has stored one full table's worth of
    positions — i.e. it thinks until the RAM you gave it is fully utilized.
    More RAM ⇒ longer, deeper thinks. Ctrl-C still aborts.
  It always answers instantly with a single legal move and stops early once
  a forced mate is proven.
- **It can never OOM**: all engine data is fixed-size after startup and a Go
  runtime memory limit (hash + 25% GC headroom) bounds transients; measured
  RSS stays flat for hours (sampled: no growth over long runs). Because the
  hash is sized from *available* (not total) RAM, even several instances
  coexist safely: `MemAvailable` on Linux, free + inactive memory on FreeBSD
  and macOS (FreeBSD without swap: free memory only, the rest cannot be
  reclaimed). In TIME mode the table is also no larger than the search can
  fill within the limit: about 48 MB per thread and second.
- Set `CHESS_VERBOSE=1` to print the chosen hash size, memory limit and
  thread count to stderr at startup.
- The search runs in parallel on **all CPU cores**, in every mode (Lazy
  SMP: every core searches the same position with a shared transposition
  table; helpers spread over the search depths). The number of threads is
  what the system reports and grants the program: all logical CPUs, fewer
  in a container with a CPU limit. On an 8-core box expect roughly 6-7x
  more nodes searched per second than single-threaded. `CHESS_THREADS=<n>`
  uses fewer cores: do that in a virtual machine whose host cannot spare
  all the CPUs the guest shows, where a search on all of them can crawl.
- You play White and move first. Type moves in coordinate notation on stdin:
  `e2e4`, castling as the king move (`e1g1`), promotion with a suffix
  (`e7e8q`, `e7e8n`, ...). Uppercase input is accepted.
- The computer answers with its move on stdout in the same notation.
- Illegal or unparsable input prints a short message on **stderr** and the
  game state is unchanged; just type again.
- `check` is announced on stdout after any move that gives check. When the
  game ends the program prints the result and a one-word reason, e.g.
  `1-0 checkmate`, `1/2-1/2 stalemate`, `0-1 checkmate`, `1/2-1/2 fifty-move`,
  `1/2-1/2 repetition`, `1/2-1/2 material` — then exits.
- End of input (Ctrl-D) quits.

### Extra stdin commands

- `d` — draw the current board: exactly 8 lines of exactly 8 characters,
  rank 8 at the top, file a leftmost, UPPERCASE = White (`KQRBNP`),
  lowercase = black (`kqrbnp`), space = empty square. No labels or borders.
- `c` — the computer chooses and plays **your** move for you (same think-time
  limit), prints it, then answers with its own move as usual (two move lines).

### DEMO mode

`./chess demo [limit]` — the computer plays both sides, as if the human
typed `c` forever, and the board is redrawn after every move (an implicit
`d`). Limit semantics are identical to normal mode (default 600 s/move,
negative = fixed depth, `0` = RAM-limited); use `./chess demo 1` for a fast
~1 s/move show. Ends with the normal result line.

`./chess silentdemo [limit]` (alias `sdemo`) — exactly the same, but the
board printouts are skipped: output is just the move list (plus `check` and
the final result line). Nothing else differs.

### Verbose / debug mode

`VERBOSE=1 ./chess ...` prints a full technical trace of the engine's
thinking to **stderr** (stdout remains the clean move protocol, so you can
pipe/tee it in any mode, including the demos):

- startup: hash-table size, Go memory limit, thread count
- per search: side to move, active limit (time/depth/RAM), legal-move count
- per completed depth: depth, selective depth (deepest quiescence ply),
  score (`cp` or `mate n`), best move, node count, nodes/sec, hash-table
  fill %, and the principal variation reconstructed from the hash table
- aspiration-window events (`fail-low`/`fail-high` + re-search window)
- 0-mode: RAM-budget progress every 10% (stores used, nodes, table fill)
- per move played: final depth/seldepth, score, total nodes (exact), nps,
  wall time, table fill, Go heap in use / reserved
- after every applied move: the position as a FEN string + zobrist key

Every line is stamped `[seconds.milliseconds]` since program start. With
`VERBOSE` unset (or `0`) none of the instrumentation runs — no extra
atomics, no timers, no allocations — the engine is bit-for-bit as fast as
before. `CHESS_VERBOSE=1` still prints just the one startup line.

### GUI mode

`./chess gui [limit]` opens a black-&-white X11 chessboard window (512x596).
The limit argument and all engine behaviour are identical to text mode.

- **No dependencies**: the program speaks the raw X11 wire protocol itself
  over `$DISPLAY` (unix socket, TCP or a socket path as XQuartz sets it;
  `.Xauthority` MIT-MAGIC-COOKIE-1 supported) and paints the whole window
  itself - board, pieces and text - so it needs no server fonts, images or
  files: still a single stdlib-only Go file. Works under a real X server,
  XQuartz or headless `Xvfb`.
- **Your move (White)**: drag a piece and drop it on its destination, or
  click the piece and then the target square. Illegal moves are refused in
  the status bar and the position is unchanged. Promotions pop up a Q/R/B/N
  chooser (a click outside it takes the move back).
- **Changing your mind**: click the selected piece again, drop the piece
  back on its own square (or outside the board), or click the right mouse
  button - the selection is gone and another piece can be picked. Clicking
  another piece of yours selects that one instead.
- **`--icons`** (or `CHESS_ICONS=1`): piece icons instead of letters. They
  are drawn by the program from geometric shapes, there are no image files.
  Without it White pieces are letters on white badges, black pieces on black
  badges (standard FEN letters).
- **CPU MOVE button**: the engine picks and plays *your* move for you, then
  answers normally - the GUI equivalent of the `c` command. **`--no-cpu`**
  (or `CHESS_NO_CPU=1`) hides it, so the human gets no engine help.
- **`--input=click`** / **`--input=drag`** (or `CHESS_INPUT=...`) leave only
  one of the two ways to move; the default `both` has them side by side.
- The status bar shows whose turn it is, "check", thinking state and the
  final result (e.g. `1-0 checkmate`); the last move's squares are outlined.
- stdout still carries the exact text-mode move protocol (`e2e4`, `check`,
  `1-0 checkmate`, ...), so you can log or pipe a GUI game just the same.
- **`./chess gui demo [limit]`** combines both modes: the engine plays both
  sides in the window (no CPU button, the board ignores the mouse), the
  board animates move by move and stdout gets the usual move list.
  `./chess gui demo 1` is a fast ~1 s/move show. Mode keywords and options
  may be given in any order (`demo gui` works too).

## Full rules implemented

Castling (all legality conditions), en passant, promotion (q/r/b/n), check,
checkmate, stalemate, 50-move rule, threefold repetition, insufficient
material (K vs K, K+minor vs K, same-colored-bishops only).

## Engine

Tournament-grade techniques, all in one file (validated feature-by-feature
in head-to-head gauntlets — an addition that neither beat the previous
binary nor fixed a flaw without costing strength was reverted):

- **Board**: a mailbox plus incrementally maintained piece bitboards; move
  generation, attack detection and SEE run on the bitboards
- **Search**: parallel Lazy SMP on **all CPU cores** sharing a lock-free
  XOR-validated, generation-aged, 2-way bucketed transposition table that
  also caches static evals (auto-sized from available RAM, see above);
  iterative deepening with **aspiration windows**; **principal variation
  search** (PVS); **late move reductions** (log-based table, history- and
  improving-adjusted); null-move pruning (adaptive R, eval-scaled, with
  verification search at high depths against zugzwang); **singular
  extensions + multicut**; **probcut**; razoring; **reverse futility**,
  **futility**, **late-move** and **SEE pruning** (quiets and losing
  captures) - none of which drops a quiet move that gives check; internal
  iterative reduction; mate-distance pruning; check extensions; in-search
  repetition, 50-move and insufficient-material detection
- **Quiescence**: transposition-table probing and storing, SEE (static
  exchange evaluation) pruning of losing captures, delta pruning, and full
  check-evasion search (no stand-pat in check — finds quiescence mates);
  a king that is left with pawns at most and has no move is the stalemate
  it is, so even a search of depth 1 does not stalemate a bare king
- **Move ordering**: TT move → good captures by SEE + capture history →
  promotions → killer moves → countermove → quiet history (butterfly +
  two-ply **continuation history**, gravity-updated with penalties for
  failed tries) → losing captures last; done in stages - the TT move is
  tried before the others are scored, quiet moves are only sorted when
  some of them will be searched
- **Correction history**: a pawn-structure-keyed table learns the static
  eval's bias per structure during the search and corrects it on the fly
- **Evaluation**: a small neural network, trained on nothing but games
  the program played against itself. Each side sees the board in its own
  way (768 inputs: own or enemy man, type, square; mirrored when its king
  is on the files e-h), 256 hidden units per side, one output with 8 sets
  of weights by the number of men. The hidden units are kept up to date
  move by move and everything is plain 64-bit integer arithmetic (four
  16-bit hidden units to a word), so it searches only about a quarter
  fewer positions per second than the hand-written evaluation it replaced
  and plays far better. The weights are part of `chess.go` (its last 270 KB);
  `nnue/README.md` tells how they were made. The network is not asked
  where the answer is known: king and pawn against king (an exact table
  built at startup), the mates against a bare king (bishop and knight
  included) and clearly won positions, which are scored by material and
  progress
- **Strength**: with one thread and 1 s a move it performs at about 3250
  on the scale of the CCRL Blitz list, measured against six engines rated
  there (17-23 against Weiss 2.0, 24-1 against Weiss 1.1, 4-28 against
  Stockfish 11). That is far beyond any human player and some 500 points
  below Stockfish 19, which won every game. More threads and more time
  add to it; `STATS.md` has the matches
- No difficulty levels — always plays at full strength for the time given.

## Testing

    make test        # or: go test chess.go chess_test.go; CHESS_SELFTEST=1 ./chess

`chess_test.go` covers the GUI (mouse handling, rendering, options), the
command line, the rules around repetition and en passant, the memory
sizing, the evaluation (the network's word arithmetic against a plain
computation, its move-by-move updates, its symmetries, the endings known
exactly), the checks kept by the pruning, the stalemates seen by the
quiescence search, the repetitions the search counts as draws, and - on a
private `Xvfb`, skipped without one - the real program end to end: the
test moves the X pointer (XTEST) and compares the window pixel by pixel,
on 24-, 16- and 8-bit displays.

    CHESS_SELFTEST=1 ./chess

runs a 13-case perft suite (start position d1–d5, Kiwipete d1–d4, and four
more standard trap positions covering en passant, promotions, castling
edge cases) with zobrist-hash **and pawn-key** consistency, and the piece
sets, verified at every visited node. All cases pass. Also validated:
exact `d` output format, illegal-move rejection, wall-clock cap accuracy,
multi-core scaling, full self-play demo games at fixed depth and timed
modes, the rules against python-chess in thousands of games, and
head-to-head matches for every feature pass, in every mode (see
`STATS.md`).

Env vars for testing (not part of the spec-required CLI): `CHESS_FEN`
starts from an arbitrary FEN, `CHESS_THREADS` caps the worker count,
`CHESS_HASH_MB` fixes the hash size, `CHESS_ONESHOT=1` plays a single move
and exits, `CHESS_NET=<file>` evaluates with another network (`nnue/`).
