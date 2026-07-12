# Distributed Inference Engine

A high-performance distributed inference system for ONNX models featuring consistent hashing, LRU caching, dynamic batching, and circuit breakers.

### Links to blogs
- [Part 1](https://medium.com/@abhiramdodda/building-a-production-grade-distributed-inference-system-in-c-part-1-11b933d9216f)
- [Part 2](https://medium.com/@abhiramdodda/building-a-production-grade-distributed-inference-system-in-c-part-2-50d0d3ba290d)


## Features

- **Consistent Hashing**: Routes each request by a hash of its input data (via virtual nodes) so identical inputs always reach the same worker, maximizing cache locality across the cluster
- **LRU Cache**: Caches inference results to reduce computation for repeated requests
- **Dynamic Batching**: Automatically batches concurrent requests to improve throughput
- **Circuit Breakers**: Monitors worker health and automatically routes around failed nodes
- **Load Balancing**: Distributes requests across multiple worker nodes
- **CUDA Support**: Optional GPU acceleration with automatic CPU fallback

## Architecture

```
Client -> Gateway (port 8000) -> Worker Nodes (ports 8001, 8002, 8003)
                                       |
                                       +-> ONNX Runtime (CPU/CUDA)
                                       +-> LRU Cache
                                       +-> Batch Processor
```

## System Requirements

### Required Dependencies

- C++17 compatible compiler (GCC 7+ or Clang 5+)
- CMake 3.15 or higher
- ONNX Runtime (CPU or CUDA version)
- pthread library

### Optional Dependencies

- CUDA Toolkit (for GPU acceleration)
- Python 3.6+ with requests library (for benchmarking)

### Arch Linux Installation

```bash
# Install system packages
sudo pacman -S base-devel cmake git onnxruntime

# For CUDA support (optional)
sudo pacman -S cuda onnxruntime-cuda
```

### Ubuntu/Debian Installation

```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential cmake git

# Install ONNX Runtime (see https://onnxruntime.ai for installation)
```

## Build Instructions

### 1. Setup Dependencies

Run the setup script to download required libraries:

```bash
./setup.sh
```

This downloads:
- cpp-httplib (HTTP server library)
- nlohmann-json (JSON parsing library)

### 2. Build the Project

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

Build artifacts:
- `build/worker_node` - Inference worker process
- `build/gateway` - Request gateway/router

## Running the System

### 1. Prepare ONNX Model

Ensure you have an ONNX model file. Set the path:

```bash
export MODEL_PATH=/path/to/your/model.onnx
```

### 2. Start Worker Nodes

Start three worker nodes on different ports:

```bash
./build/worker_node 8001 worker_1 $MODEL_PATH &
./build/worker_node 8002 worker_2 $MODEL_PATH &
./build/worker_node 8003 worker_3 $MODEL_PATH &
```

Worker configuration:
- Cache capacity: 1000 entries
- Max batch size: 32 requests
- Batch timeout: 20ms

### 3. Start Gateway

Start the gateway to route requests:

```bash
./build/gateway localhost:8001 localhost:8002 localhost:8003
```

Gateway configuration:
- Listen port: 8000
- Failure threshold: 5 failures before circuit opens
- Success threshold: 2 successes to close circuit
- Circuit timeout: 30 seconds

## API Endpoints

### Gateway Endpoints

#### POST /infer

Perform inference on input data.

Request:
```json
{
  "request_id": "unique_request_id",
  "input_data": [1.0, 2.0, 3.0, ...]
}
```

Response:
```json
{
  "request_id": "unique_request_id",
  "output_data": [-0.999, 0.452, ...],
  "node_id": "worker_1",
  "cached": false,
  "inference_time_us": 1250
}
```

#### GET /stats

Get gateway statistics and circuit breaker states.

Response:
```json
{
  "total_workers": 3,
  "circuit_breakers": [
    {
      "node": "localhost:8001",
      "state": "CLOSED",
      "failures": 0,
      "successes": 0
    }
  ]
}
```

### Worker Endpoints

#### POST /infer

Direct inference request (bypass gateway).

#### GET /health

Get worker health and performance metrics.

Response:
```json
{
  "healthy": true,
  "node_id": "worker_1",
  "total_requests": 1000,
  "cache_hits": 950,
  "cache_size": 50,
  "cache_hit_rate": 0.95,
  "batch_processor": {
    "total_batches": 100,
    "avg_batch_size": 10.5,
    "timeout_batches": 5,
    "full_batches": 95
  }
}
```

## Testing and Diagnostics

### Run Diagnostic Script

```bash
chmod +x diagnose.sh
./diagnose.sh
```

Checks:
- Process status
- Port availability
- Worker health
- Gateway connectivity
- Direct inference test
- End-to-end inference test

### Manual Testing

Test worker directly:
```bash
curl -X POST http://localhost:8001/infer \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test1", "input_data": [1.0, 2.0, 3.0]}'
```

Test through gateway:
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test1", "input_data": [1.0, 2.0, 3.0]}'
```

Check statistics:
```bash
curl http://localhost:8000/stats
curl http://localhost:8001/health
```

## Benchmarking

### Install Python Dependencies

```bash
pip install requests
```

### Run Standard Benchmark

```bash
python3 benchmark.py --requests 1000 --threads 10
```

### Run Cache Effectiveness Test

```bash
python3 benchmark.py --cache-test --requests 100
```

### Benchmark Options

```
--gateway URL       Gateway URL (default: http://localhost:8000)
--requests N        Total number of requests (default: 1000)
--threads N         Number of concurrent threads (default: 10)
--workers URL...    Worker URLs for statistics
--cache-test        Run cache effectiveness test
--no-stats          Skip system statistics output
```

## Recent Improvements

- **Input-based routing for cache locality**: the gateway now routes on a hash of
  the request's `input_data` (FNV-1a over the float bits) instead of `request_id`.
  Identical inputs always reach the same worker, so each distinct result is computed
  and cached once cluster-wide instead of being duplicated across workers.
- **Thread-safe worker connections**: each worker is backed by a pool of keep-alive
  HTTP clients. Concurrent gateway requests borrow their own connection (RAII),
  removing a data race on the previously shared, non-thread-safe `httplib::Client`.
- **Honest cache-hit metrics**: `inference_time_us` on a cache hit is now the actual
  measured lookup time (single-digit microseconds) rather than a hardcoded constant.
- **Stronger cache hashing**: `VectorHash` hashes the entire input vector rather than
  sampling three elements, eliminating avoidable bucket collisions (and the O(n)
  full-vector comparisons they triggered).
- **Circuit breaker synchronization cleanup**: all breaker state is guarded by a
  single mutex with consistent locking in every accessor, removing redundant and
  inconsistently-used atomics.

## Performance Results

Measured with the gateway and all three workers running locally on a single host
(CPU inference via ONNX Runtime). Because the load generator (`benchmark.py`) shares
the machine with the servers, absolute throughput is bounded by the Python client
rather than the inference engine — see the note below.

### 1,000,000 requests / 500 threads

Throughput
- Total requests: 1,000,000
- Success rate: 100.00% (0 failures)
- Total time: 2359.18s
- Requests/sec: 423.88

Latency
- Mean: 1168.51ms
- Median: 1162.19ms
- P90: 1867.70ms
- P95: 2033.39ms
- P99: 2318.40ms

### 10,000 requests / 50 threads

Throughput
- Success rate: 100.00%
- Requests/sec: 262.77

Latency
- Mean: 187.32ms
- Median: 191.08ms
- P90: 226.22ms
- P99: 357.86ms

### Cache & Load Distribution (input-based routing)

Requests are routed by a hash of the input data, so identical inputs always map to
the same worker. Across the benchmark's 10 distinct inputs, cache entries are
partitioned with no duplication (1M / 500-thread run):

| Worker   | Requests | Cache entries | Hit rate |
|----------|----------|---------------|----------|
| worker_1 | 300,000  | 3             | 100.00%  |
| worker_2 | 400,000  | 4             | 99.99%   |
| worker_3 | 300,000  | 3             | 99.99%   |

The 10 distinct results are stored once cluster-wide (3 + 4 + 3) rather than being
replicated across every worker.

> **Note:** the ~424 req/s ceiling is client-bound, not server-bound. With 500 client
> threads at ~424 req/s, Little's Law gives ~1.18s mean latency, which matches the
> measured mean — each thread spends its time in the queue it creates. The C++ path
> serves cache hits in single-digit microseconds (`total_batches` stays in single
> digits, so only the first occurrence of each input runs real inference). Measuring
> true server throughput requires a keep-alive/async load client.

## Configuration

### Tuning Worker Performance

Edit `worker_node.cpp` to adjust:
- Cache capacity (line 49): `LRUCache cache_(1000)`
- Max batch size (line 51): `32`
- Batch timeout (line 52): `std::chrono::milliseconds(20)`

### Tuning Circuit Breakers

Edit `gateway.cpp` to adjust:
- Failure threshold (line 19): `5`
- Success threshold (line 20): `2`
- Timeout (line 21): `std::chrono::seconds(30)`

### Tuning Consistent Hashing

Edit `gateway.cpp` constructor to adjust virtual nodes per physical node (default: 150).

## Troubleshooting

### All Requests Failing

Check if workers are running:
```bash
ps aux | grep worker_node
```

Check if ports are listening:
```bash
ss -tuln | grep -E "8000|8001|8002|8003"
```

Restart gateway to reset circuit breakers:
```bash
pkill gateway
./build/gateway localhost:8001 localhost:8002 localhost:8003
```

### Circuit Breakers Opening

Circuit breakers open after 5 consecutive failures. Common causes:
- Worker crashed or not responding
- Model file not found or corrupted
- Network connectivity issues

Wait 30 seconds for automatic reset, or restart the gateway.

### Low Cache Hit Rate

Ensure request inputs are identical for cache hits. The cache hashes the full input vector (FNV-1a over the float bits), and the gateway routes by that same hash, so identical inputs consistently reach the same worker's cache.

### CUDA Not Loading

If CUDA provider fails to load, the system automatically falls back to CPU. Check console output for:
```
CUDA failed to load: <error message>
Falling back to CPU Provider...
```

Verify CUDA installation:
```bash
nvidia-smi
```

## Project Structure

```
distributed_inference_engine/
├── src/
│   ├── circuit_breaker.cpp      # Circuit breaker implementation
│   ├── consistent_hash.cpp      # Consistent hashing
│   ├── gateway.cpp              # Gateway server
│   ├── inference_engine.cpp     # ONNX Runtime wrapper
│   └── worker_node.cpp          # Worker node server
├── include/
│   ├── batch_processor.h        # Dynamic batching (header-only)
│   ├── circuit_breaker.h
│   ├── consistent_hash.h
│   ├── inference_engine.h
│   ├── lru_cache.h             # LRU cache (header-only)
│   └── gateway.h
├── external/                    # Third-party dependencies
│   ├── cpp-httplib/
│   └── json/
├── build/                       # Build output
├── setup.sh                     # Dependency setup script
├── diagnose.sh                  # System diagnostic script
├── benchmark.py                 # Performance benchmark
├── CMakeLists.txt              # Build configuration
└── README.md
```
