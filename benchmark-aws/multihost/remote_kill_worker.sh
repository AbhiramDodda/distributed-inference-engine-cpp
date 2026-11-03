#!/usr/bin/env bash
set -Eeuo pipefail
WORK=$HOME/benchmark-work
NODE_ID=${1:?node id is required}

pkill -TERM -f "$WORK/build/worker_node [0-9]+ ${NODE_ID} " 2>/dev/null || true
for _ in $(seq 1 40); do
    if ! pgrep -f "$WORK/build/worker_node [0-9]+ ${NODE_ID} " >/dev/null 2>&1; then
        exit 0
    fi
    sleep 0.25
done
pkill -KILL -f "$WORK/build/worker_node [0-9]+ ${NODE_ID} " 2>/dev/null || true
