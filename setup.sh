#!/bin/bash

set -e

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    OS=$(uname -s)
fi

echo "Detected OS: $OS"
echo ""

if command -v pacman &> /dev/null; then
    echo "Installing system dependencies via pacman"
    
    if [ "$EUID" -ne 0 ]; then
        echo "Note: May need sudo password for package installation"
        sudo pacman -S --needed --noconfirm base-devel cmake git || {
            echo "Package installation failed or packages already installed"
        }
    else
        pacman -S --needed --noconfirm base-devel cmake git || {
            echo "Package installation failed or packages already installed"
        }
    fi
    
    echo "System dependencies installed"
else
    echo "Not running on Arch-based system, skipping package installation"
    echo "Please ensure you have: build-essential, cmake, git"
fi

echo ""

mkdir -p external

install_dependency() {
    local directory=$1 url=$2 tag=$3 expected=$4 marker=$5
    if [ -f "$directory/$marker" ]; then
        local repo_root actual
        repo_root=$(git -C "$directory" rev-parse --show-toplevel 2>/dev/null || true)
        actual=$(git -C "$directory" rev-parse HEAD 2>/dev/null || true)
        if [ "$repo_root" != "$(cd "$directory" && pwd -P)" ] || [ "$actual" != "$expected" ]; then
            echo "Existing dependency at $directory is not the expected checkout; inspect it manually." >&2
            return 1
        fi
        echo "$directory already has the expected revision"
        return
    fi
    if [ -d "$directory" ]; then
        if [ -n "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            echo "$directory is not empty and lacks $marker; inspect it manually." >&2
            return 1
        fi
        rmdir "$directory"
    fi
    git clone --depth 1 --branch "$tag" "$url" "$directory"
    if [ "$(git -C "$directory" rev-parse HEAD)" != "$expected" ]; then
        echo "Unexpected dependency revision at $directory" >&2
        return 1
    fi
}

install_dependency external/cpp-httplib https://github.com/yhirose/cpp-httplib.git \
    v0.14.3 cbca63f091ef1147ff57e90eb1ee5e558aa05d2c httplib.h
install_dependency external/json https://github.com/nlohmann/json.git \
    v3.11.3 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 include/nlohmann/json.hpp

echo "System Information:"
echo "  OS: $OS"
echo "  CMake: $(cmake --version 2>/dev/null | head -n1 || echo 'Not found')"
echo "  GCC: $(gcc --version 2>/dev/null | head -n1 || echo 'Not found')"
echo "  Git: $(git --version 2>/dev/null || echo 'Not found')"
echo ""
echo "Dependencies downloaded:"
echo "  cpp-httplib (HTTP server library)"
echo "  nlohmann-json (JSON library)"
# echo ""
# echo "Next steps:"
# echo "  1. mkdir build && cd build"
# echo "  2. cmake -DCMAKE_BUILD_TYPE=Release .."
# echo "  3. make -j\$(nproc)"
# echo ""
# echo "Then run:"
# echo "  ./worker_node 8001 worker_1"
# echo "  ./worker_node 8002 worker_2"
# echo "  ./worker_node 8003 worker_3"
# echo "  ./gateway localhost:8001 localhost:8002 localhost:8003"