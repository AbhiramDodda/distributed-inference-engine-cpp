#!/usr/bin/env bash
set -Eeuo pipefail
WORK=$HOME/benchmark-work
RESULT=$WORK/results
mkdir -p "$RESULT"
ORT=$WORK/onnxruntime-linux-x64-1.23.2
export LD_LIBRARY_PATH="$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GATEWAY_PORT="${GATEWAY_PORT:-8000}"

setsid vmstat 1 > "$RESULT/gateway-vmstat.txt" < /dev/null &
MONITOR=$!
setsid nohup "$WORK/build/gateway" "$@" < /dev/null > "$RESULT/gateway.log" 2>&1 &
echo $! > "$RESULT/gateway.pid"
echo "$MONITOR" > "$RESULT/gateway-monitor.pid"
