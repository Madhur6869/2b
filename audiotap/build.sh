#!/usr/bin/env bash
# Build the audiotap helper (mic + system-audio capture; system audio needs macOS 14.2+).
# Needs the Xcode command line tools (xcode-select --install).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
swiftc -O -target "$(uname -m)-apple-macos14.2" -o audiotap audiotap.swift
echo "built $(pwd)/audiotap"
