// embed: puts a network file (written by train) into chess.go as its nnData constant: deflated, in base64.
//
// usage: go run nnue/embed.go NET.bin [chess.go]
package main

import (
	"bytes"
	"compress/flate"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: embed NET.bin [chess.go]")
		os.Exit(2)
	}
	src := "chess.go"
	if len(os.Args) > 2 {
		src = os.Args[2]
	}
	net, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	code, err := os.ReadFile(src)
	if err != nil {
		panic(err)
	}
	const head = "const nnData = `\n"
	i := strings.Index(string(code), head)
	j := strings.LastIndex(string(code), "`")
	if i < 0 || j < i+len(head) {
		fmt.Fprintln(os.Stderr, "embed: no nnData constant in", src)
		os.Exit(1)
	}
	var z bytes.Buffer
	w, _ := flate.NewWriter(&z, flate.BestCompression)
	w.Write(net)
	w.Close()
	b := base64.StdEncoding.EncodeToString(z.Bytes())
	var out strings.Builder
	out.Write(code[:i+len(head)])
	for len(b) > 0 {
		n := min(len(b), 110)
		out.WriteString(b[:n] + "\n")
		b = b[n:]
	}
	out.Write(code[j:])
	if err := os.WriteFile(src, []byte(out.String()), 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("%s: network of %d bytes, %d deflated\n", src, len(net), z.Len())
}
