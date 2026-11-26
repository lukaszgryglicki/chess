SHELL = /bin/sh
GO ?= go
JOBS ?= 4
INSTALL_DIR ?= /data/scripts

# make: chess-<os> for this machine; make all: the binaries of all three systems
release:
	@os=$$(uname -s | tr '[:upper:]' '[:lower:]'); [ "$$os" != darwin ] || os=macos; \
	$(MAKE) "chess-$$os"

all: chess-freebsd chess-linux chess-macos

chess-freebsd: chess.go
	CGO_ENABLED=0 GOOS=freebsd GOARCH=amd64 $(GO) build -p $(JOBS) -o $@ chess.go

chess-linux: chess.go
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build -p $(JOBS) -o $@ chess.go

chess-macos: chess.go
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 $(GO) build -p $(JOBS) -o $@ chess.go

# the binaries need no shared library (macOS: only the system library)
static: release

install: release
	@os=$$(uname -s | tr '[:upper:]' '[:lower:]'); [ "$$os" != darwin ] || os=macos; \
	mkdir -p "$(INSTALL_DIR)" && install -m 755 "chess-$$os" "$(INSTALL_DIR)/chess"

# the X11 tests run on a private Xvfb and are skipped without one
test:
	CHESS_HASH_MB=16 $(GO) test -p $(JOBS) -count=1 chess.go chess_test.go
	CHESS_HASH_MB=16 CHESS_SELFTEST=1 $(GO) run chess.go

clean:
	rm -f chess

.PHONY: release all static install test clean
