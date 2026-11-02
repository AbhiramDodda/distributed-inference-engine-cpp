#!/usr/bin/env bash
set -Eeuo pipefail
WORK=$HOME/benchmark-work
RESULT=$WORK/results
mkdir -p "$WORK" "$RESULT"
export PATH=/usr/local/bin:/usr/bin:/bin

sudo dnf install -y git gcc-c++ make cmake tar gzip procps-ng sysstat python3 python3-pip >/dev/null

if [[ ! -d "$WORK/source" ]]; then
    git clone https://github.com/AbhiramDodda/distributed-inference-engine-cpp.git "$WORK/source"
fi
cd "$WORK/source"
git checkout --detach "$BASE_REVISION"
if [[ -s "$HOME/source.patch" ]]; then
    git apply --check "$HOME/source.patch"
    git apply "$HOME/source.patch"
fi
python3 "$HOME/verify_source.py" source "$WORK/source" "$HOME/source-manifest.json"

rm -rf external/cpp-httplib external/json
git clone --depth 1 --branch v0.14.3 https://github.com/yhirose/cpp-httplib.git external/cpp-httplib
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git external/json
test "$(git -C external/cpp-httplib rev-parse HEAD)" = cbca63f091ef1147ff57e90eb1ee5e558aa05d2c
test "$(git -C external/json rev-parse HEAD)" = 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03

cd "$WORK"
if [[ ! -d "$WORK/onnxruntime-linux-x64-1.23.2" ]]; then
    curl -fsSL --retry 2 --connect-timeout 15 --max-time 180 \
        -o "$WORK/ort.tgz" \
        https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
    printf '%s  %s\n' \
        1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b \
        "$WORK/ort.tgz" | sha256sum -c -
    tar -xzf "$WORK/ort.tgz" -C "$WORK"
fi
ORT="$WORK/onnxruntime-linux-x64-1.23.2"
cmake -S "$WORK/source" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_INCLUDE_DIR="$ORT/include" \
    -DONNXRUNTIME_LIB_DIR="$ORT/lib"
cmake --build "$WORK/build" --parallel 2

if [[ "${MODEL_NAME:-resnet50-v2-7}" != "resnet50-v2-7" ]]; then
    if [[ ! -f "$WORK/model.onnx" ]]; then
        curl -fsSL --retry 2 --connect-timeout 15 --max-time 300 \
            -o "$WORK/model.onnx" "$MODEL_URL"
    fi
    printf '%s  %s\n' "$MODEL_SHA" "$WORK/model.onnx" | sha256sum -c -
fi

echo "prepared $(hostname)"
