# chess

A text-mode chess game in a single C11 file (`chess.c`), standard library only.
You play White, the computer plays Black.

## Build

```sh
cc -O3 -std=c11 -o chess chess.c
```

Tested with gcc 15.2 on x86-64 Linux; the code is plain C11 plus libc (no `-lm`
needed) and builds clean with `-Wall -Wextra`, at `-O0` as well as `-O3`.

## Usage

```sh
./chess [seconds]
```

`seconds` is the computer's maximum thinking time **per move** (integer or
fractional, e.g. `./chess 3`, `./chess 0.5`).

* default when omitted: **600**
* `0` means *unlimited*: the search runs to a fixed maximum depth (20) instead of
  being cut off by the clock. From the starting position that takes a couple of
  minutes per move on a modern desktop CPU; later moves are faster.
* the limit is enforced against the real (wall) clock; a partially finished search
  iteration is discarded and the best completed move is played
* any other value, or more than one argument, is rejected with a non-zero exit code

The program prints nothing at startup — just type. Moves are coordinates:

```
e2e4          ordinary move
e1g1          castling (king move)
e7e8q         promotion (q, r, b or n; also e7e8n etc.)
E2E4          uppercase is accepted, surrounding spaces are ignored
```

Illegal or unparsable input gets a short message and a re-prompt; the game state is
never lost. The computer's move is printed on its own line after each of your moves.

### Extra commands

| command | effect |
| ------- | ------ |
| `d`     | draw the board: exactly 8 lines of exactly 8 characters, rank 8 first, file `a` leftmost, FEN letters, uppercase = White, spaces = empty squares (no borders or labels) |
| `c`     | let the computer choose and play **your** move too (same time limit); it prints the move it played for you, then its own reply |

### Check and the end of the game

`check` is printed on its own line whenever the side to move is in check. When the
game ends, the program prints the standard result followed by a one-word reason and
then exits:

```
1-0 checkmate
0-1 checkmate
1/2-1/2 stalemate
1/2-1/2 repetition
1/2-1/2 fifty
1/2-1/2 insufficient
```

Full rules are implemented: castling (with all rights/path/not-through-check
conditions), en passant, promotion, check, checkmate, stalemate, the fifty-move
rule, threefold repetition and insufficient material.

## Engine

Iterative-deepening alpha-beta with principal-variation search, aspiration windows,
quiescence search, a 4 M-entry transposition table, null-move pruning, late-move
reductions, futility pruning, static-exchange-aware move ordering, killer moves,
history heuristic, singular extensions and a tapered positional evaluation. There
are no difficulty levels — it always uses the whole time budget at full strength.

## Optional self test

The source contains a build-time self test (move-generation counts, rule checks,
hash-key consistency) that is compiled in with `-DSELFTEST`:

```sh
cc -O3 -std=c11 -DSELFTEST -o chess_st chess.c
./chess_st perft 5          # ~470M nodes, verifies move generation against known counts
./chess_st search "<FEN>" 2 # prints the move the search chooses for that position
```
