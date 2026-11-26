# The evaluation network

`chess.go` evaluates positions with a small neural network. Its weights are
the `nnData` constant at the end of that file. They were trained on nothing
but games the program played against itself, with the four tools here (Go,
standard library only, no GPU).

## What it computes

- Each side sees the board in its own way: 768 inputs = own or enemy man x
  type x square, the board mirrored left to right when that side's king is
  on the files e-h.
- 256 hidden units per side (the same weights for both), clipped to 0..1.
- Output: the hidden units of the side to move and of the other side times
  the weights of one of 8 buckets (by the number of men), plus a value of
  every man in that bucket. Times 211 it is the score in centipawns.
- In the program everything is integer: hidden units are 16-bit lanes, four
  to a 64-bit word, kept up to date move by move with plain word additions;
  the output takes 6 bits of each unit and 8-bit weights, four units per
  multiplication (`nnAct`, `nnEval`). `train` writes exactly these numbers
  and checks them against its floating point model.

The network is not asked where the answer is known: king and pawn against
king (an exact table), pieces against a bare king, and positions it scores
above 1000, which are scored by material and progress instead (`eval`).

## How to train one

    # 1. games of the program against itself: 32 bytes per position
    mkdir /tmp/dg && sed 's/^func main() {$/func chessMain() {/' chess.go > /tmp/dg/engine.go
    cp nnue/datagen.go /tmp/dg && (cd /tmp/dg && go build -o datagen engine.go datagen.go)
    for s in 1 2 3 4; do /tmp/dg/datagen -out data/p$s.bin -seed $s -nodes 2000 -ctl data & done
    touch data/STOP        # ends them after their current game (data/PAUSE suspends them)

    # 2. shuffle into shards of a million positions, train, look at the result
    go run nnue/shuffle.go -out shards data/p*.bin
    go run nnue/train.go -shards shards -h 256 -wdl 0.1 -k 211 -out net
    CHESS_NET=net.bin ./chess 5

    # 3. put it into chess.go
    go run nnue/embed.go net.bin

`datagen` plays with the network in `chess.go`, or with `CHESS_NET=file`.
`CHESS_NET` works for the program too, so two networks can play a match.

## The network in chess.go

Four generations of games, each played by the program of its moment (about
3,800 nodes a move) and labelled by its own searches:

| generation | played by | games | positions |
|---|---|---|---|
| 1 | v4, the hand-written evaluation | 106,355 | 9,788,724 |
| 2 | 128 hidden units, trained on 7.9M positions of generation 1 | 121,792 | 10,949,097 |
| 3 | 256 hidden units, trained on generations 1 and 2 | 164,246 | 14,896,242 |
| 4 | 256 hidden units, trained on generations 1 to 3 | 257,432 | 22,049,073 |

649,825 games, 57.7 million positions, 80 hours of processor time. The
network in `chess.go` was trained on all of them (the last shard, a million
positions, kept aside for validation):

    go run nnue/train.go -shards shards -h 256 -wdl 0.1 -k 211 -epochs 14 -threads 8

31 minutes on eight cores, mean squared error on the validation shard
0.00446. In matches at fixed depth 7 each network beat the one before it:
the first v4 by 80 ± 43 Elo, then +169 ± 23, +70 ± 18 and +57 ± 16.

What was tried and measured on the way is in `STATS.md` (v5): the weight
of the game result in the target (0.1 is best, the search score teaches
more than the result), the number of hidden units (128 is faster and
weaker; 512 is stronger at equal depth, not at equal time), the left-right
mirroring and the value of every man in the output (both help).
