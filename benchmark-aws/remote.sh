#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
WORK=$HOME/benchmark-work
RESULT=$WORK/results
mkdir -p "$RESULT"
exec > >(tee -a "$WORK/remote-live.log") 2>&1
set -x
WORKER_PID=
GATEWAY_PID=
MONITOR_PID=
finish() {
    status=$?
    trap - EXIT
    echo "Remote workload exit status: $status"
    for pid in "$GATEWAY_PID" "$WORKER_PID" "$MONITOR_PID"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "$GATEWAY_PID" "$WORKER_PID" "$MONITOR_PID"; do
        if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
    done
    printf '%s\n' "$status" > "$RESULT/exit-status"
    set +x
    cp "$WORK/remote-live.log" "$RESULT/remote.log"
    sync
    tar -C "$WORK" -czf "$HOME/benchmark-results.tar.gz" results
    sha256sum "$HOME/benchmark-results.tar.gz" > "$HOME/benchmark-results.sha256"
    exit "$status"
}
trap finish EXIT
cp "$HOME/source-manifest.json" "$RESULT/source-manifest.json"
cp "$HOME/source.patch" "$RESULT/source.patch"
python3 "$HOME/verify_source.py" payload "$HOME" "$HOME/source-manifest.json"
SOURCE_MODE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mode"])' "$HOME/source-manifest.json")
printf 'Started UTC: '; date -u +%FT%TZ
uname -a
lscpu
free -h
printf '\nInstalling build prerequisites\n'
sudo dnf install -y git gcc-c++ make cmake tar gzip procps-ng sysstat
rpm -q git gcc-c++ make cmake python3 | tee "$RESULT/packages.txt"
curl --version | sed -n '1p' | tee -a "$RESULT/packages.txt"
cd "$WORK"
git clone https://github.com/AbhiramDodda/distributed-inference-engine-cpp.git source
cd source
git checkout --detach c4d8a08b72df810ffad0dce6a993cecae22494bb
git rev-parse HEAD | tee "$RESULT/revision.txt"
if [[ "$SOURCE_MODE" == instrumented ]]; then
    git apply --check "$HOME/source.patch"
    git apply "$HOME/source.patch"
fi
python3 "$HOME/verify_source.py" source "$WORK/source" "$HOME/source-manifest.json"
sha256sum models/resnet50-v2-7.onnx | tee "$RESULT/model.sha256"
rmdir external/cpp-httplib external/json
git clone --depth 1 --branch v0.14.3 https://github.com/yhirose/cpp-httplib.git external/cpp-httplib
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git external/json
test "$(git -C external/cpp-httplib rev-parse HEAD)" = cbca63f091ef1147ff57e90eb1ee5e558aa05d2c
test "$(git -C external/json rev-parse HEAD)" = 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
cd "$WORK"
curl -fsSL --retry 2 --connect-timeout 15 --max-time 180 -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
printf '%s  %s\n' 1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b ort.tgz | sha256sum -c -
tar -xzf ort.tgz
ORT=$WORK/onnxruntime-linux-x64-1.23.2
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_INCLUDE_DIR="$ORT/include" -DONNXRUNTIME_LIB_DIR="$ORT/lib"
cmake --build build --parallel 2
export LD_LIBRARY_PATH="$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ENGINE_BACKEND=ort
stdbuf -oL -eL "$WORK/build/worker_node" 8001 worker_1 "$WORK/source/models/resnet50-v2-7.onnx" > "$RESULT/worker.log" 2>&1 &
WORKER_PID=$!
stdbuf -oL -eL "$WORK/build/gateway" localhost:8001 > "$RESULT/gateway.log" 2>&1 &
GATEWAY_PID=$!
for attempt in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:8001/health > "$RESULT/worker-before.json" && curl -fsS --max-time 2 http://127.0.0.1:8000/stats > "$RESULT/gateway-before.json"; then break; fi
    kill -0 "$WORKER_PID" "$GATEWAY_PID"
    sleep 1
done
curl -fsS --max-time 5 http://127.0.0.1:8001/health > "$RESULT/worker-before.json"
curl -fsS --max-time 5 http://127.0.0.1:8000/stats > "$RESULT/gateway-before.json"
vmstat 1 > "$RESULT/vmstat.txt" &
MONITOR_PID=$!
load_args=("$RESULT")
if [[ "$SOURCE_MODE" == instrumented ]]; then
    load_args+=("--require-timings")
fi
timeout 20m python3 "$HOME/load.py" "${load_args[@]}"
curl -fsS --max-time 5 http://127.0.0.1:8001/health > "$RESULT/worker-after.json"
curl -fsS --max-time 5 http://127.0.0.1:8000/stats > "$RESULT/gateway-after.json"
printf 'Finished UTC: '; date -u +%FT%TZ
