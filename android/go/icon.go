// The launcher icon. Tool of the build only, not part of the app:
//
//	go run chess.go frontend.go icon.go android-icon <pixels> <file.png>
//
// writes the foreground of the app's adaptive icon: the white knight of the
// game's own piece pictures on a transparent square.
package main

import (
	"fmt"
	"os"
	"strconv"
)

func init() {
	if len(os.Args) == 4 && os.Args[1] == "android-icon" {
		if err := droidIcon(os.Args[2], os.Args[3]); err != nil {
			fmt.Fprintln(os.Stderr, "android-icon:", err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func droidIcon(pixels, file string) error {
	n, err := strconv.Atoi(pixels)
	if err != nil || n < 16 || n > 4096 {
		return fmt.Errorf("bad size %q", pixels)
	}
	// launchers show the middle two thirds of the layer
	return droidPNG(iconSprite(WN, n*56/100), n, file)
}
