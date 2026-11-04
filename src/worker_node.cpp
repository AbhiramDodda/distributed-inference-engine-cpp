#include "engine_backend.h"
#include "lru_cache.h"
#include "batch_processor.h"
#include "shutdown_signal.h"
#include "tensor_payload.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// HTTP server worker threads. Kept above the batch processor's max_batch_size
// (32) so that enough /infer handlers can be in flight to actually fill a batch.
static constexpr size_t kServerThreads = 64;

// Batch deadline. Ten milliseconds keeps the low-load queueing tax small while
// still filling the measured 2- and 4-request batches; the environment override
// supports latency/throughput experiments and operational tuning.
static int resolveBatchDeadlineMs() {
    if (const char* value = std::getenv("BATCH_DEADLINE_MS")) {
        try {
            const int requested = std::stoi(value);
            if (requested > 0) {
                return requested;
            }
        } catch (const std::exception&) {
            // Use the default below.
        }
    }
    return 10;
}

static size_t resolvePositiveSize(const char* name, size_t fallback) {
    if (const char* value = std::getenv(name)) {
        try {
            const unsigned long long requested = std::stoull(value);
            if (requested > 0) {
                return static_cast<size_t>(requested);
            }
        } catch (const std::exception&) {
            // Use the fallback below.
        }
    }
    return fallback;
}

// request/response types for batch processor
struct InferenceRequest {
    std::string request_id;
    std::vector<float> input_data;
};

struct InferenceResponse {
    std::string request_id;
    std::vector<float> output_data;
    int64_t inference_time_us;
    int64_t batch_execution_us;
    size_t batch_size;
    bool cached;
};

class WorkerNode {
public:
    WorkerNode(const std::string& node_id, int port, const std::string& model_path)
        : node_id_(node_id),
          port_(port),
          // Backend chosen by ENGINE_BACKEND (ort|compiled); compiled loads the
          // tensorforge build dir from TF_BUILD_DIR.
          engine_(make_engine(
              std::getenv("ENGINE_BACKEND") ? std::getenv("ENGINE_BACKEND") : "ort",
              model_path,
              std::getenv("TF_BUILD_DIR") ? std::getenv("TF_BUILD_DIR") : "",
              port % 3)),
          cache_(resolvePositiveSize("CACHE_CAPACITY", 1000)),
          batch_deadline_ms_(resolveBatchDeadlineMs()),
          max_batch_size_(
              std::min(
                  resolvePositiveSize("MAX_BATCH_SIZE", 32),
                  engine_->maxBatchSize()
              )
          ),
          batch_processor_(
              max_batch_size_,
              std::chrono::milliseconds(batch_deadline_ms_),
              [this](const std::vector<InferenceRequest>& reqs) {
                  return this->processBatch(reqs);
              }
          ) {
        total_requests_.store(0);
        cache_hits_.store(0);
        std::cout << "Engine backend: " << engine_->name() << std::endl;
        std::cout << "Model batch limit: " << engine_->maxBatchSize() << std::endl;
        std::cout << "Effective batch size: " << max_batch_size_ << std::endl;
        batch_processor_.start();
    }
    
    ~WorkerNode() {
        batch_processor_.stop();
    }
    
    struct TimedResult {
        InferenceResponse response;
        int64_t queue_wait_us;
    };

    TimedResult processInput(
        const std::string& request_id,
        const std::vector<float>& input_data
    ) {
        total_requests_++;

        const auto cache_start = std::chrono::high_resolution_clock::now();
        auto cached = cache_.get(input_data);
        if (cached.has_value()) {
            const auto cache_end = std::chrono::high_resolution_clock::now();
            cache_hits_++;
            InferenceResponse response;
            response.request_id = request_id;
            response.output_data = *cached;
            response.inference_time_us = std::chrono::duration_cast<
                std::chrono::microseconds>(cache_end - cache_start).count();
            response.batch_execution_us = 0;
            response.batch_size = 0;
            response.cached = true;
            return TimedResult{std::move(response), 0};
        }

        InferenceRequest inf_req{request_id, input_data};
        auto timed = batch_processor_.process(inf_req);
        cache_.put(input_data, timed.response.output_data);
        return TimedResult{timed.response, timed.queue_wait_us};
    }

    json handleInfer(const json& request) {
        const std::string request_id = request["request_id"];
        const std::vector<float> input_data = request["input_data"];
        const TimedResult timed = processInput(request_id, input_data);
        const InferenceResponse& response_data = timed.response;

        json response;
        response["request_id"] = response_data.request_id;
        response["output_data"] = response_data.output_data;
        response["node_id"] = node_id_;
        response["cached"] = response_data.cached;
        response["inference_time_us"] = response_data.inference_time_us;
        if (response_data.cached) {
            response["queue_wait_us"] = nullptr;
            response["batch_execution_us"] = nullptr;
            response["batch_size"] = nullptr;
        } else {
            response["queue_wait_us"] = timed.queue_wait_us;
            response["batch_execution_us"] = response_data.batch_execution_us;
            response["batch_size"] = response_data.batch_size;
        }
        return response;
    }

    int getBatchDeadlineMs() const { return batch_deadline_ms_; }
    size_t getMaxBatchSize() const { return max_batch_size_; }
    size_t getCacheCapacity() const { return cache_.capacity(); }
    const std::string& getNodeId() const { return node_id_; }

    json getHealth() {
        auto batch_metrics = batch_processor_.getMetrics();
        json health;
        health["healthy"] = true;
        health["node_id"] = node_id_;
        health["backend"] = engine_->name();
        health["total_requests"] = total_requests_.load();
        health["cache_hits"] = cache_hits_.load();
        health["cache_size"] = cache_.size();
        health["cache_hit_rate"] = cache_.getHitRate();
        // batch processor metrics
        json batch_stats;
        batch_stats["total_batches"] = batch_metrics.total_batches;
        batch_stats["avg_batch_size"] = batch_metrics.avg_batch_size;
        batch_stats["timeout_batches"] = batch_metrics.timeout_batches;
        batch_stats["full_batches"] = batch_metrics.full_batches;
        health["batch_processor"] = batch_stats;
        
        return health;
    }
    
private:
    std::vector<InferenceResponse> processBatch(
        const std::vector<InferenceRequest>& requests) {
        const auto legacy_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> inputs;
        inputs.reserve(requests.size());
        for (const auto& req : requests) {
            inputs.push_back(req.input_data);
        }
        
        // batch inference
        const auto batch_start = std::chrono::steady_clock::now();
        auto outputs = engine_->batchPredict(inputs);
        const auto batch_end = std::chrono::steady_clock::now();
        
        const auto legacy_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            legacy_end - legacy_start).count();
        const auto batch_execution_us = std::chrono::duration_cast<std::chrono::microseconds>(
            batch_end - batch_start).count();
        std::vector<InferenceResponse> responses;
        responses.reserve(requests.size());
        int64_t per_request_time = duration / requests.size();
        for (size_t i = 0; i < requests.size(); ++i) {
            InferenceResponse resp;
            resp.request_id = requests[i].request_id;
            resp.output_data = outputs[i];
            resp.inference_time_us = per_request_time;
            resp.batch_execution_us = batch_execution_us;
            resp.batch_size = requests.size();
            resp.cached = false;
            responses.push_back(resp);
        }
        return responses;
    }
    
    std::string node_id_;
    int port_;
    std::unique_ptr<IEngine> engine_;
    LRUCache<std::vector<float>, std::vector<float>, VectorHash> cache_;
    int batch_deadline_ms_;
    size_t max_batch_size_;
    BatchProcessor<InferenceRequest, InferenceResponse> batch_processor_;
    
    std::atomic<int64_t> total_requests_;
    std::atomic<int64_t> cache_hits_;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <port> <node_id> [model_path]" << std::endl;
        std::cerr << "  Or set MODEL_PATH environment variable" << std::endl;
        return 1;
    }
    int port = std::stoi(argv[1]);
    std::string node_id = argv[2];
    
    // model path from argument or environment
    std::string model_path;
    if (argc >= 4) {
        model_path = argv[3];
    } else {
        const char* env_path = std::getenv("MODEL_PATH");
        if (env_path) {
            model_path = env_path;
        } else {
            std::cerr << "Error: No model path provided!" << std::endl;
            std::cerr << "  Provide as: " << argv[0] << " <port> <node_id> <model_path>" << std::endl;
            std::cerr << "  Or set: export MODEL_PATH=/path/to/model.onnx" << std::endl;
            return 1;
        }
    }
    
    std::cout << "Using model: " << model_path << std::endl;
    // Block shutdown signals before ORT starts its thread pools so every thread
    // inherits the mask. The main thread then waits for a signal explicitly.
    try {
        block_shutdown_signals();
    } catch (const std::exception& e) {
        std::cerr << "Failed to install shutdown handler: " << e.what() << std::endl;
        return 1;
    }
    // Construct the worker (and its engine) up front so a bad ENGINE_BACKEND or
    // a load failure exits cleanly with a message instead of a core dump.
    std::unique_ptr<WorkerNode> worker_ptr;
    try {
        worker_ptr = std::make_unique<WorkerNode>(node_id, port, model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to start worker: " << e.what() << std::endl;
        return 1;
    }
    WorkerNode& worker = *worker_ptr;
    httplib::Server server;
    // Responses are written as headers followed by the body. Without TCP_NODELAY,
    // Nagle plus delayed ACK can hold the body for a fixed interval on keep-alive
    // connections, which dominates cache-hit latency. Enable it explicitly.
    server.set_tcp_nodelay(true);
    // cpp-httplib's default server thread pool is hardware_concurrency-1 (~15).
    // Each /infer handler blocks in the batch processor waiting for its result,
    // so that default caps the number of requests that can be queued for
    // batching at once -- below max_batch_size (32), which makes full batches
    // structurally unreachable and stalls the tail in 5s keep-alive quanta once
    // concurrency exceeds the pool. Size the pool above the batch size so the
    // batcher can actually fill and the server admits real concurrency.
    server.new_task_queue = [] { return new httplib::ThreadPool(kServerThreads); };
    // inference endpoint
    server.Post("/infer", [&worker](const httplib::Request& req, httplib::Response& res) {
        try {
            auto request = json::parse(req.body);
            auto response = worker.handleInfer(request);
            
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error;
            error["error"] = e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });
    // raw float32 tensor endpoint
    server.Post("/infer_tensor", [&worker](const httplib::Request& req, httplib::Response& res) {
        try {
            const std::string request_id = req.get_header_value("X-Request-Id");
            if (request_id.empty() || request_id.size() > 256) {
                throw std::invalid_argument("X-Request-Id must be 1-256 characters");
            }
            constexpr size_t kMaxTensorBytes = 16'000'000;
            if (req.body.empty() || req.body.size() % sizeof(float) != 0 ||
                req.body.size() > kMaxTensorBytes) {
                throw std::invalid_argument(
                    "tensor body must be a non-empty multiple of 4 bytes, at most 16 MB"
                );
            }
            const auto input = decode_float32_le(req.body);
            const auto timed = worker.processInput(request_id, input);
            const auto& data = timed.response;
            res.set_header("X-Request-Id", data.request_id);
            res.set_header("X-Node-Id", worker.getNodeId());
            res.set_header("X-Cached", data.cached ? "true" : "false");
            res.set_header("X-Output-Elements", std::to_string(data.output_data.size()));
            res.set_header("X-Batch-Size", std::to_string(data.batch_size));
            res.set_header("X-Queue-Wait-Us", std::to_string(timed.queue_wait_us));
            res.set_header("X-Batch-Execution-Us", std::to_string(data.batch_execution_us));
            res.set_content(encode_float32_le(data.output_data), "application/octet-stream");
        } catch (const std::invalid_argument& e) {
            json error;
            error["error"] = e.what();
            res.status = 400;
            res.set_content(error.dump(), "application/json");
        } catch (const std::exception& e) {
            json error;
            error["error"] = e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });
    // health endpoint
    server.Get("/health", [&worker](const httplib::Request&, httplib::Response& res) {
        auto health = worker.getHealth();
        res.set_content(health.dump(), "application/json");
    });
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Worker Node: " << node_id << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "   Port:              " << port << std::endl;
    std::cout << "   Cache Capacity:    " << worker.getCacheCapacity() << " entries" << std::endl;
    std::cout << "   Batch Size:        " << worker.getMaxBatchSize() << " requests" << std::endl;
    std::cout << "   Batch Timeout:     " << worker.getBatchDeadlineMs() << "ms" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Ready to accept requests!" << std::endl;
    std::cout << std::endl;
    std::thread server_thread([&server, port] {
        server.listen("0.0.0.0", port);
    });
    server.wait_until_ready();
    if (!server.is_running()) {
        std::cerr << "Worker server failed to start on port " << port << std::endl;
        server_thread.join();
        return 1;
    }
    const int signal_number = wait_for_shutdown_signal();
    std::cout << "Received signal " << signal_number << ", shutting down" << std::endl;
    server.stop();
    server_thread.join();
    worker_ptr.reset();
    return 0;
}
