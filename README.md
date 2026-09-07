# chess-fable — text-mode chess

A complete text-mode chess game in a single Go file (standard library only),
written by Claude Fable 5 (GitHub Copilot CLI) from the same spec
(`chess.md`) given to the local Qwen model — see `STATS.md` for the
head-to-head numbers.

## Build

    go build -o chess chess.go

(Any Go ≥ 1.21; no dependencies. Cross-compile: `GOOS=freebsd go build ...`)

## Usage

    ./chess [limit]
    ./chess demo [limit]         # self-playing demo, board after every move
    ./chess silentdemo [limit]   # same demo without boards (moves only)

- The single optional argument is the computer's per-move limit — there is
  always **exactly one limiting resource per mode**:
  - **positive** = **TIME-limited**: max think seconds per move (wall clock,
    enforced, checked every ~1024 nodes; also stops early when another full
    iteration clearly would not fit). Depth and RAM are not limits here.
    **Default when omitted: 600.**
  - **negative** = **DEPTH-limited**: fixed search depth (`-1` = depth 1,
    `-10` = depth 10); no clock, no RAM criterion.
  - **`0`** = **RAM-limited**: no clock and no depth cap. The engine
    auto-sizes its hash table to ~40% of the RAM currently available on the
    machine (floor 64 MB, cap 16 GB; override with `CHESS_HASH_MB=<mb>`),
    and plays its move once the search has stored one full table's worth of
    positions — i.e. it thinks until the RAM you gave it is fully utilized.
    More RAM ⇒ longer, deeper thinks. Ctrl-C still aborts.
  It always answers instantly with a single legal move and stops early once
  a forced mate is proven.
- **It can never OOM**: all engine data is fixed-size after startup and a Go
  runtime memory limit (hash + 25% GC headroom) bounds transients; measured
  RSS stays flat for hours (sampled: no growth over long runs). Because the
  hash is sized from *available* (not total) RAM, even several instances
  coexist safely.
- Set `CHESS_VERBOSE=1` to print the chosen hash size, memory limit and
  thread count to stderr at startup.
- The search runs in parallel on **all CPU cores** (Lazy SMP: every core
  searches the same position with a shared transposition table; helpers start
  one ply deeper). On an 8-core box expect roughly 3-6x more nodes searched
  per second than single-threaded.
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

## Full rules implemented

Castling (all legality conditions), en passant, promotion (q/r/b/n), check,
checkmate, stalemate, 50-move rule, threefold repetition, insufficient
material (K vs K, K+minor vs K, same-colored-bishops only).

## Engine

Tournament-grade techniques, all in one file:

- **Search**: parallel Lazy SMP on **all CPU cores** sharing a lock-free
  XOR-validated transposition table (auto-sized from available RAM, see
  above); iterative deepening with **aspiration windows**; **principal
  variation search** (PVS); **late move reductions** (log-based table);
  null-move pruning (adaptive R = 3 + depth/6); **reverse futility**,
  **futility** and **late-move pruning**; internal iterative reduction;
  mate-distance pruning; check extensions; in-search repetition and 50-move
  detection
- **Quiescence**: SEE (static exchange evaluation) pruning of losing
  captures, delta pruning, and full check-evasion search (no stand-pat in
  check — finds quiescence mates)
- **Move ordering**: TT move → MVV-LVA captures → killer moves →
  countermove heuristic → history heuristic
- **Evaluation** (tapered mg/eg by game phase): PeSTO piece-square tables
  and values, bishop pair, passed / isolated / doubled pawns, rooks on open
  and semi-open files, king pawn-shield, tempo
- No difficulty levels — always plays at full strength for the time given.

## Testing

    CHESS_SELFTEST=1 ./chess

runs a 13-case perft suite (start position d1–d5, Kiwipete d1–d4, and four
more standard trap positions covering en passant, promotions, castling
edge cases) with zobrist-hash consistency verified at every visited node.
All cases pass. Also validated: exact `d` output format, illegal-move
rejection, wall-clock cap accuracy, multi-core scaling, full self-play
demo games at fixed depth and timed modes, and tactical spot checks
(fool's mate, mate-in-1 finishes, WAC.001 `Qg6!!` found in seconds).

The env var `CHESS_FEN` starts the game from an arbitrary FEN (testing aid,
not part of the spec-required CLI).
