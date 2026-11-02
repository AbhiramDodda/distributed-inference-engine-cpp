#!/usr/bin/env bash
set -Eeuo pipefail
WORK=$HOME/benchmark-work
RESULT=$WORK/results
mkdir -p "$RESULT"

stop_pid_file() {
    local file=$1
    [[ -f "$file" ]] || return 0
    local pid
    pid=$(cat "$file" 2>/dev/null || true)
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

for file in "$RESULT"/*.pid; do
    stop_pid_file "$file"
done

# Also stop any process whose PID file was lost or overwritten.
pkill -TERM -f "$WORK/build/worker_node" 2>/dev/null || true
pkill -TERM -f "$WORK/build/gateway" 2>/dev/null || true

for _ in $(seq 1 40); do
    if ! pgrep -f "$WORK/build/worker_node" >/dev/null 2>&1 \
        && ! pgrep -f "$WORK/build/gateway" >/dev/null 2>&1; then
        rm -f "$RESULT"/*.pid
        exit 0
    fi
    sleep 0.25
done

for file in "$RESULT"/*.pid; do
    [[ -f "$file" ]] || continue
    pid=$(cat "$file" 2>/dev/null || true)
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
done
pkill -KILL -f "$WORK/build/worker_node" 2>/dev/null || true
pkill -KILL -f "$WORK/build/gateway" 2>/dev/null || true
rm -f "$RESULT"/*.pid

if pgrep -f "$WORK/build/worker_node" >/dev/null 2>&1 \
    || pgrep -f "$WORK/build/gateway" >/dev/null 2>&1; then
    echo "services are still running after stop" >&2
    exit 1
fi
