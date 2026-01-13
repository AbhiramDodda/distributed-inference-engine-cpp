import requests
import time
import threading
import statistics
import json
from collections import defaultdict
import argparse

class BenchmarkRunner:
    def __init__(self, gateway_url, num_requests, num_threads):
        self.gateway_url = gateway_url
        self.num_requests = num_requests
        self.num_threads = num_threads
        self.results = []
        self.lock = threading.Lock()
        self.errors = defaultdict(int)
        
    def send_request(self, request_id):
        start = time.time()
        try:
            payload = {
                "request_id": f"req_{request_id}",
                "input_data": [float(request_id % 10), float((request_id % 10) + 1), float((request_id % 10) + 2)]
            }
            
            response = requests.post(
                f"{self.gateway_url}/infer",
                json=payload,  
                timeout=10
            )
            elapsed = (time.time() - start) * 1000
            
            with self.lock:
                self.results.append({
                    "latency": elapsed,
                    "status": response.status_code,
                    "success": response.status_code == 200
                })
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            with self.lock:
                self.results.append({
                    "latency": elapsed,
                    "status": 0,
                    "success": False
                })
                self.errors[str(type(e).__name__)] += 1
    
    def worker_thread(self, thread_id, requests_per_thread):
        for i in range(requests_per_thread):
            request_id = thread_id * requests_per_thread + i
            self.send_request(request_id)
    
    def run(self):
        print(f"Starting benchmark:")
        print(f"  Gateway: {self.gateway_url}")
        print(f"  Requests: {self.num_requests}")
        print(f"  Threads: {self.num_threads}")
        print()
        
        requests_per_thread = self.num_requests // self.num_threads
        
        start_time = time.time()
        threads = []
        
        for thread_id in range(self.num_threads):
            t = threading.Thread(
                target=self.worker_thread,
                args=(thread_id, requests_per_thread)
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start_time
        self.print_results(total_time)
    
    def print_results(self, total_time):
        successful = [r for r in self.results if r["success"]]
        failed = len(self.results) - len(successful)
        
        if not successful:
            print("ERROR: All requests failed")
            if self.errors:
                print("\nError breakdown:")
                for error_type, count in self.errors.items():
                    print(f"  {error_type}: {count}")
            return
        
        latencies = [r["latency"] for r in successful]
        latencies.sort()
        
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print()
        print("Throughput")
        print(f"  Total requests:     {self.num_requests}")
        print(f"  Successful:         {len(successful)}")
        print(f"  Failed:             {failed}")
        print(f"  Success rate:       {len(successful)/len(self.results)*100:.2f}%")
        print(f"  Total time:         {total_time:.2f}s")
        print(f"  Requests/sec:       {len(successful)/total_time:.2f}")
        print()
        print("Latency (ms):")
        print(f"  Mean:               {statistics.mean(latencies):.2f}")
        print(f"  Median:             {statistics.median(latencies):.2f}")
        print(f"  Stddev:             {statistics.stdev(latencies):.2f}")
        print(f"  Min:                {min(latencies):.2f}")
        print(f"  Max:                {max(latencies):.2f}")
        print()
        print("Percentiles (ms):")
        p50_idx = int(len(latencies) * 0.50)
        p90_idx = int(len(latencies) * 0.90)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        print(f"  p50:                {latencies[p50_idx]:.2f}")
        print(f"  p90:                {latencies[p90_idx]:.2f}")
        print(f"  p95:                {latencies[p95_idx]:.2f}")
        print(f"  p99:                {latencies[p99_idx]:.2f}")
        print()
        if self.errors:
            print("Errors:")
            for error_type, count in self.errors.items():
                print(f"  {error_type}: {count}")
            print()
        
        print("=" * 70)

def get_worker_stats(worker_url):
    try:
        response = requests.get(f"{worker_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_gateway_stats(gateway_url):
    try:
        response = requests.get(f"{gateway_url}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def print_system_stats(gateway_url, worker_urls):
    print()
    print("=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    print()
    
    gateway_stats = get_gateway_stats(gateway_url)
    if gateway_stats:
        print("Gateway Circuit Breakers:")
        for breaker in gateway_stats.get("circuit_breakers", []):
            print(f"  {breaker['node']}: {breaker['state']} (failures: {breaker['failures']}, successes: {breaker['successes']})")
        print()
    
    for worker_url in worker_urls:
        stats = get_worker_stats(worker_url)
        if stats:
            node_id = stats.get('node_id', 'unknown')
            print(f"Worker {node_id} ({worker_url}):")
            print(f"  Total requests:    {stats.get('total_requests', 0)}")
            print(f"  Cache size:        {stats.get('cache_size', 0)}")
            print(f"  Cache hits:        {stats.get('cache_hits', 0)}")
            print(f"  Cache hit rate:    {stats.get('cache_hit_rate', 0)*100:.2f}%")
            bp = stats.get('batch_processor', {})
            print(f"  Avg batch size:    {bp.get('avg_batch_size', 0):.2f}")
            print(f"  Total batches:     {bp.get('total_batches', 0)}")
            print(f"  Full batches:      {bp.get('full_batches', 0)}")
            print(f"  Timeout batches:   {bp.get('timeout_batches', 0)}")
            print()
    
    print("=" * 70)

def run_cache_effectiveness_test(gateway_url):
    print()
    print("=" * 70)
    print("CACHE EFFECTIVENESS TEST")
    print("=" * 70)
    print()
    
    print("Phase 1: Initial requests (cache misses expected)")
    latencies_miss = []
    for i in range(100):
        start = time.time()
        payload = {
            "request_id": f"cache_miss_{i}",
            "input_data": [float(i % 10), float((i % 10) + 1), float((i % 10) + 2)]
        }
        requests.post(f"{gateway_url}/infer", json=payload, timeout=10)
        latencies_miss.append((time.time() - start) * 1000)
    
    print(f"  Mean latency: {statistics.mean(latencies_miss):.2f}ms")
    print()
    
    print("Phase 2: Repeated requests (cache hits expected)")
    time.sleep(1)
    
    latencies_hit = []
    for i in range(100):
        start = time.time()
        payload = {
            "request_id": f"cache_hit_{i}",
            "input_data": [float(i % 10), float((i % 10) + 1), float((i % 10) + 2)]
        }
        requests.post(f"{gateway_url}/infer", json=payload, timeout=10)
        latencies_hit.append((time.time() - start) * 1000)
    
    print(f"  Mean latency: {statistics.mean(latencies_hit):.2f}ms")
    print()
    
    speedup = statistics.mean(latencies_miss) / statistics.mean(latencies_hit)
    print(f"Cache speedup: {speedup:.2f}x")
    print()
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed inference system")
    parser.add_argument("--gateway", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests")
    parser.add_argument("--threads", type=int, default=10, help="Concurrent threads")
    parser.add_argument("--workers", nargs="+", 
                       default=["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"],
                       help="Worker URLs")
    parser.add_argument("--cache-test", action="store_true", help="Run cache effectiveness test")
    parser.add_argument("--no-stats", action="store_true", help="Skip system statistics")
    
    args = parser.parse_args()
    
    if args.cache_test:
        run_cache_effectiveness_test(args.gateway)
    
    benchmark = BenchmarkRunner(args.gateway, args.requests, args.threads)
    benchmark.run()
    
    if not args.no_stats:
        print_system_stats(args.gateway, args.workers)

if __name__ == "__main__":
    main()