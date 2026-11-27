SHELL = /bin/sh

# make: ../chess-android.apk (see README.md); make install: onto the phone on the USB cable
apk:
	sh build.sh

test:
	sh build.sh test

install: apk
	adb install -r ../chess-android.apk

clean:
	rm -rf build

.PHONY: apk test install clean
