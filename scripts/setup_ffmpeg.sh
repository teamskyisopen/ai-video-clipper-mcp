#!/usr/bin/env bash
# ==============================================================================
# AI Video Clipper - Linux & macOS FFmpeg Setup Utility
# ==============================================================================
# Installs or downloads FFmpeg configured with libass support into ./bin/
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$PROJECT_ROOT/bin"
FFMPEG_BIN="$BIN_DIR/ffmpeg"
FFPROBE_BIN="$BIN_DIR/ffprobe"

echo "============================================================"
echo "    AI Video Clipper - Linux/macOS FFmpeg Setup Utility     "
echo "============================================================"

# 1. Check if local ./bin/ffmpeg already has libass
if [[ -x "$FFMPEG_BIN" ]]; then
    echo "[*] Checking existing binary in ./bin/ffmpeg..."
    if "$FFMPEG_BIN" -version 2>&1 | grep -q "libass"; then
        echo "[✓] Local FFmpeg with libass is already installed in ./bin/!"
        exit 0
    fi
fi

# 2. Check system FFmpeg for libass support
if command -v ffmpeg &>/dev/null; then
    echo "[*] Found system ffmpeg in PATH. Checking for libass..."
    if ffmpeg -version 2>&1 | grep -q "libass"; then
        echo "[✓] System ffmpeg already has libass support enabled!"
        mkdir -p "$BIN_DIR"
        ln -sf "$(command -v ffmpeg)" "$FFMPEG_BIN"
        ln -sf "$(command -v ffprobe)" "$FFPROBE_BIN"
        echo "[✓] Symlinked system binaries to ./bin/"
        exit 0
    fi
fi

# 3. Handle by OS
OS_TYPE="$(uname -s)"
mkdir -p "$BIN_DIR"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "[1/3] Detected macOS (Darwin). Installing ffmpeg via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "[X] Error: Homebrew is required for automated macOS installation."
        echo "    Install Homebrew or install ffmpeg with libass manually."
        exit 1
    fi
    brew install ffmpeg
    ln -sf "$(command -v ffmpeg)" "$FFMPEG_BIN"
    ln -sf "$(command -v ffprobe)" "$FFPROBE_BIN"

elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "[1/3] Detected Linux. Attempting static build download (with libass)..."
    ARCH="$(uname -m)"
    if [[ "$ARCH" == "x86_64" ]]; then
        URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
    else
        echo "[!] Unsupported architecture ($ARCH) for automated static download."
        echo "    Please install ffmpeg using your distribution package manager:"
        echo "    sudo apt-get install -y ffmpeg libass-dev"
        exit 1
    fi

    echo "[2/3] Downloading static build from John Van Sickle..."
    TMP_DIR="$(mktemp -d)"
    curl -fsSL "$URL" -o "$TMP_DIR/ffmpeg.tar.xz"
    tar -xf "$TMP_DIR/ffmpeg.tar.xz" -C "$TMP_DIR"
    cp "$TMP_DIR"/ffmpeg-*-static/ffmpeg "$FFMPEG_BIN"
    cp "$TMP_DIR"/ffmpeg-*-static/ffprobe "$FFPROBE_BIN"
    chmod +x "$FFMPEG_BIN" "$FFPROBE_BIN"
    rm -rf "$TMP_DIR"
fi

# 4. Final verification
echo "[3/3] Verifying installation..."
if [[ -x "$FFMPEG_BIN" ]] && "$FFMPEG_BIN" -version 2>&1 | grep -q "libass"; then
    echo "============================================================"
    echo "[✓] Setup Complete! FFmpeg with libass is ready."
    echo "    Binary path: $FFMPEG_BIN"
    echo "============================================================"
    exit 0
else
    echo "[!] Installation completed, but libass verification could not be confirmed."
    echo "    Please verify with: ./bin/ffmpeg -version | grep libass"
fi
