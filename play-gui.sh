#!/bin/bash
if [ -z "$1" ]
then
  ./chess-macos gui --icons --no-cpu 30
else
  if [ "$1" = "max" ]
  then
    ./chess-macos gui --icons --no-cpu
  else
    ./chess-macos gui --icons --no-cpu "$1"
  fi
fi
