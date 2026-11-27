# Chess for Android

The same game as on the desktop, in an Android app: `../chess.go` is used as
it is, rules, engine, network and the way pieces are picked, dropped and let
go. Nothing of the desktop program depends on this directory.

The start screen asks what the command line does: **N**, the computer's limit
for a move (`30` seconds, `0` no time limit, `-7` depth 7), a checkbox for the
**CPU MOVE** button (the computer moves for you; off by default) and one for
**Demo** (the computer plays both sides; off by default). Then the board:
tap a piece and tap its target, or drag it there. Tapping the same piece
again, or putting it back, lets it go, and another can be picked.

## How it is put together

- `go/frontend.go` is compiled together with `../chess.go` (same package,
  `chess.go` first) into one program for arm64 Android. The app starts it as
  a child process and they talk in lines of text; the top of that file
  describes them. It drives the `gui` of `chess.go`, the state machine behind
  the X11 window, so both front ends behave alike. `go/frontend_test.go`
  tests it on any system.
- `java/` is the app, three classes on plain Android, no libraries:
  `MainActivity` (the two screens, the record of the game), `BoardView`
  (draws the board, passes the finger on), `Engine` (the process).
- `go/icon.go` draws the launcher icon from the game's own knight during the
  build. The piece pictures on the board come from the game too: it draws
  them in the size of the screen's squares into the app's cache, once.

On a phone the game takes a quarter of the free memory for its hash table,
64 to 512 MB (the desktop takes 80%), and all cores. With N = 0 it moves when
that table is used up, on a Galaxy S20 after about 11 seconds. While the app
is not on the screen the game process is stopped. The moves played are
recorded, so a game goes on where it was after the app or its process has
been closed by the system or the phone restarted.

## Build

    make            # ../chess-android.apk
    make test       # the tests of go/frontend.go
    make install    # build, then onto the phone on the USB cable (adb)

It needs Go, a JDK (17 or newer), `zip`, and two packages of Google's
Android SDK under `~/android-sdk` (or `$ANDROID_SDK`):

    mkdir -p ~/android-sdk/platforms ~/android-sdk/build-tools && cd ~/android-sdk
    curl -O https://dl.google.com/android/repository/platform-36_r02.zip
    curl -O https://dl.google.com/android/repository/build-tools_r37_linux.zip
    unzip -q platform-36_r02.zip && mv android-36 platforms/
    unzip -q build-tools_r37_linux.zip && mv android-37.0 build-tools/37.0.0

(`_linux` is also right for FreeBSD, `_macosx` for a Mac. They are Google's,
under the Android SDK licence.) No Gradle, no NDK. `aapt2` and `zipalign` of
the build-tools are Linux programs: FreeBSD runs them through its Linux
compatibility (`linux64.ko` and a Linux base such as `linux_base-rl9`).

The first build makes a signing key, `~/.android/chess.keystore` with its
password beside it in `chess.keystore.pass`. Keep both: Android installs a
new version over the old one only when the same key signed them. Without the
key the app has to be removed from the phone before the new one goes on, and
the game in progress is lost. The key is not in the repository.

## Install

Any 64-bit phone with Android 8 or newer.

- Over USB: switch on the phone's **USB debugging** (Settings → About phone →
  Software information → tap *Build number* 7 times, then Settings →
  Developer options), plug it in, accept "Allow USB debugging?", `make install`.
  On FreeBSD `adb` comes with `pkg install android-tools`, and because USB
  devices belong to root there, its server has to be started as root once:
  `sudo env HOME=$HOME adb start-server`.
- Without a cable: send `chess-android.apk` to the phone, tap it and allow
  "Install unknown apps" for the app that opened it.

Google Play is not involved.

## Testing on a phone

`DEBUG=1 make` builds a debuggable app that takes a game from the command
line, which the normal build ignores:

    adb shell am start -S -n io.github.lukaszgryglicki.chess/.MainActivity \
        --es limit 2 --ez cpu true --ez demo false \
        --es moves "'e2e4 e7e5'" --es fen "'8/4P3/8/8/8/7p/k7/4K3 w - - 0 1'" \
        --ez wide true --ei die 3000

`wide` turns the screen, `die` takes the game process away every so many
milliseconds (the app has to bring the game back). `adb shell input tap`,
`input swipe` and `adb exec-out screencap -p` do the rest.
