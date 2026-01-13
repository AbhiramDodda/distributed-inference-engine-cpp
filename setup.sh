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
cd external

echo "Downloading cpp-httplib..."
if [ ! -d "cpp-httplib" ]; then
    git clone --depth 1 --branch v0.14.3 https://github.com/yhirose/cpp-httplib.git
    echo "cpp-httplib downloaded"
else
    echo "cpp-httplib exists"
fi

echo "Downloading nlohmann-json..."
if [ ! -d "json" ]; then
    git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
    echo "nlohmann-json downloaded"
else
    echo "nlohmann-json exists"
fi

cd ..

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