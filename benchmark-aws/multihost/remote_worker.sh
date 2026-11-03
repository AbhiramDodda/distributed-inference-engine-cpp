#!/usr/bin/env bash
set -Eeuo pipefail
WORK=$HOME/benchmark-work
RESULT=$WORK/results
ROLE=${ROLE:-worker}
mkdir -p "$RESULT"
ORT=$WORK/onnxruntime-linux-x64-1.23.2
export LD_LIBRARY_PATH="$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ORT_INTRA_THREADS="${ORT_INTRA_THREADS:-2}"
export BATCH_DEADLINE_MS="${BATCH_DEADLINE_MS:-10}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
export CACHE_CAPACITY="${CACHE_CAPACITY:-1000}"

if [[ "${MODEL_NAME:-resnet50-v2-7}" == "resnet50-v2-7" ]]; then
    MODEL="$WORK/source/models/resnet50-v2-7.onnx"
else
    MODEL="$WORK/model.onnx"
fi

setsid vmstat 1 > "$RESULT/${ROLE}-vmstat.txt" < /dev/null &
MONITOR=$!
setsid nohup "$WORK/build/worker_node" "$PORT" "$NODE_ID" "$MODEL" \
    < /dev/null > "$RESULT/${ROLE}.log" 2>&1 &
echo $! > "$RESULT/${ROLE}.pid"
echo "$MONITOR" > "$RESULT/${ROLE}-monitor.pid"
