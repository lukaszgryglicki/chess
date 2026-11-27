#!/bin/sh
# Builds ../chess-android.apk: the game (../chess.go with go/frontend.go, for
# arm64 Android) and the app around it (java/), packed and signed.
#
# Needs: go, a JDK (17 or newer), zip, and of Google's Android SDK the two
# packages "platforms;android-NN" and "build-tools;NN" under $ANDROID_SDK
# (README.md says how to get them). aapt2 and zipalign of the build-tools are
# Linux programs: on FreeBSD they run through the Linux compatibility layer.
#
#   sh build.sh          build and sign
#   sh build.sh test     the tests of go/frontend.go
#   DEBUG=1 sh build.sh  a debuggable build (takes test settings from "am start")
set -eu
cd "$(dirname "$0")"

SDK=${ANDROID_SDK:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/android-sdk}}}
GO=${GO:-go}
JOBS=${JOBS:-4}
MIN_SDK=26
VERSION_CODE=${VERSION_CODE:-1}
VERSION_NAME=${VERSION_NAME:-1.0}
KEYSTORE=${KEYSTORE:-$HOME/.android/chess.keystore}
OUT=${OUT:-../chess-android.apk}

die() { echo "build.sh: $*" >&2; exit 1; }
newest() { ls -d "$1"/* 2>/dev/null | sort -V | tail -1; }

B=build
rm -rf $B/go $B/classes $B/apk $B/res $B/*.apk $B/*.zip
mkdir -p $B/go $B/classes $B/apk/lib/arm64-v8a $B/res/mipmap-xxxhdpi
cp ../chess.go go/*.go $B/go/

if [ "${1:-}" = test ]; then
	cd $B/go
	CHESS_HASH_MB=16 CHESS_THREADS=2 $GO test -p "$JOBS" -count=1 chess.go frontend.go frontend_test.go
	exit
fi

BT=${BUILD_TOOLS:-$(newest "$SDK/build-tools")}
PLATFORM=${PLATFORM:-$(newest "$SDK/platforms")}
[ -x "$BT/aapt2" ] || die "no build-tools under $SDK (set ANDROID_SDK; see README.md)"
[ -f "$PLATFORM/android.jar" ] || die "no platform under $SDK (set ANDROID_SDK; see README.md)"
TARGET_SDK=${PLATFORM##*android-}
TARGET_SDK=${TARGET_SDK%%[!0-9]*}
if [ -z "${JAVA_HOME:-}" ] && ! command -v javac >/dev/null 2>&1; then
	JAVA_HOME=$(ls -d /usr/local/openjdk2[1-9] /usr/local/openjdk17 2>/dev/null | tail -1)
fi
J=${JAVA_HOME:+$JAVA_HOME/bin/}
"${J}javac" -version >/dev/null 2>&1 || die "no JDK (set JAVA_HOME)"

echo "== the game, for arm64 Android"
(cd $B/go && CGO_ENABLED=0 GOOS=android GOARCH=arm64 $GO build -p "$JOBS" -trimpath -ldflags='-s -w' \
	-o ../apk/lib/arm64-v8a/libchess.so chess.go frontend.go)

echo "== the icon, from the game's own knight"
(cd $B/go && CHESS_HASH_MB=16 $GO run -p "$JOBS" chess.go frontend.go icon.go android-icon 432 ../res/mipmap-xxxhdpi/ic_fg.png)

echo "== resources and manifest (target API $TARGET_SDK)"
"$BT/aapt2" compile --dir res -o $B/res1.zip
"$BT/aapt2" compile --dir $B/res -o $B/res2.zip
"$BT/aapt2" link -o $B/unsigned.apk -I "$PLATFORM/android.jar" --manifest AndroidManifest.xml \
	--min-sdk-version $MIN_SDK --target-sdk-version "$TARGET_SDK" \
	--version-code "$VERSION_CODE" --version-name "$VERSION_NAME" ${DEBUG:+--debug-mode} $B/res1.zip $B/res2.zip

echo "== the app"
"${J}javac" -source 8 -target 8 -Xlint:-options \
	-bootclasspath "$PLATFORM/android.jar:$BT/core-lambda-stubs.jar" -d $B/classes \
	$(find java -name '*.java')
"${J}java" -cp "$BT/lib/d8.jar" com.android.tools.r8.D8 --release --min-api $MIN_SDK \
	--lib "$PLATFORM/android.jar" --output $B/apk $(find $B/classes -name '*.class')

echo "== pack, align, sign"
(cd $B/apk && zip -q -r ../unsigned.apk classes.dex lib)
"$BT/zipalign" -f 4 $B/unsigned.apk $B/aligned.apk
if [ ! -f "$KEYSTORE" ]; then
	# Android installs a new version over the old one only if both are signed
	# with the same key: this one is made once and kept outside the repository
	echo "   new signing key: $KEYSTORE (keep it, with $KEYSTORE.pass)"
	mkdir -p "$(dirname "$KEYSTORE")"
	(umask 077 && od -An -N24 -tx1 /dev/urandom | tr -d ' \n' > "$KEYSTORE.pass")
	"${J}keytool" -genkeypair -keystore "$KEYSTORE" -storepass:file "$KEYSTORE.pass" -alias chess \
		-keyalg RSA -keysize 3072 -validity 12000 -dname "CN=chess" > /dev/null 2>&1
	chmod 600 "$KEYSTORE"
fi
apksigner() { # quiet unless it fails (newer JDKs warn about its native code)
	"${J}java" -jar "$BT/lib/apksigner.jar" "$@" > $B/apksigner.log 2>&1 || { cat $B/apksigner.log >&2; die "apksigner $1 failed"; }
}
apksigner sign --ks "$KEYSTORE" --ks-pass "file:$KEYSTORE.pass" --ks-key-alias chess --out "$OUT" $B/aligned.apk
rm -f "$OUT.idsig"
apksigner verify "$OUT"
ls -l "$OUT"
