#include "inference_engine.h"
#include "lru_cache.h"
#include "batch_processor.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// request/response types for batch processor
struct InferenceRequest {
    std::string request_id;
    std::vector<float> input_data;
};

struct InferenceResponse {
    std::string request_id;
    std::vector<float> output_data;
    int64_t inference_time_us;
    bool cached;
};

class WorkerNode {
public:
    WorkerNode(const std::string& node_id, int port, const std::string& model_path)
        : node_id_(node_id),
          port_(port),
          engine_(model_path, port % 3),
          cache_(1000),  // capacity: 1000 entries
          batch_processor_(
              32,  // max_batch_size
              std::chrono::milliseconds(20),  // timeout
              [this](const std::vector<InferenceRequest>& reqs) {
                  return this->processBatch(reqs);
              }
          ) {
        total_requests_.store(0);
        cache_hits_.store(0);
        batch_processor_.start();
    }
    
    ~WorkerNode() {
        batch_processor_.stop();
    }
    
    json handleInfer(const json& request) {
        total_requests_++;
        
        std::string request_id = request["request_id"];
        std::vector<float> input_data = request["input_data"];
        
        // Check cache first
        auto cached = cache_.get(input_data);
        if (cached.has_value()) {
            cache_hits_++;
            json response;
            response["request_id"] = request_id;
            response["output_data"] = *cached;
            response["node_id"] = node_id_;
            response["cached"] = true;
            response["inference_time_us"] = 50;  // Cache hit is very fast
            
            return response;
        }
        
        // Cache miss - use batch processor
        InferenceRequest inf_req{request_id, input_data};
        InferenceResponse inf_resp = batch_processor_.process(inf_req);
        cache_.put(input_data, inf_resp.output_data);
        
        json response;
        response["request_id"] = inf_resp.request_id;
        response["output_data"] = inf_resp.output_data;
        response["node_id"] = node_id_;
        response["cached"] = false;
        response["inference_time_us"] = inf_resp.inference_time_us;
        
        return response;
    }
    
    json getHealth() {
        auto batch_metrics = batch_processor_.getMetrics();
        json health;
        health["healthy"] = true;
        health["node_id"] = node_id_;
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
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> inputs;
        inputs.reserve(requests.size());
        for (const auto& req : requests) {
            inputs.push_back(req.input_data);
        }
        
        // batch inference
        auto outputs = engine_.batchPredict(inputs);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        std::vector<InferenceResponse> responses;
        responses.reserve(requests.size());
        int64_t per_request_time = duration / requests.size();
        for (size_t i = 0; i < requests.size(); ++i) {
            InferenceResponse resp;
            resp.request_id = requests[i].request_id;
            resp.output_data = outputs[i];
            resp.inference_time_us = per_request_time;
            resp.cached = false;
            responses.push_back(resp);
        }
        return responses;
    }
    
    std::string node_id_;
    int port_;
    InferenceEngine engine_;
    LRUCache<std::vector<float>, std::vector<float>, VectorHash> cache_;
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
    WorkerNode worker(node_id, port, model_path);
    httplib::Server server;
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
    // health endpoint
    server.Get("/health", [&worker](const httplib::Request&, httplib::Response& res) {
        auto health = worker.getHealth();
        res.set_content(health.dump(), "application/json");
    });
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Worker Node: " << node_id << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "   Port:              " << port << std::endl;
    std::cout << "   Cache Capacity:    1000 entries" << std::endl;
    std::cout << "   Batch Size:        32 requests" << std::endl;
    std::cout << "   Batch Timeout:     20ms" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Ready to accept requests!" << std::endl;
    std::cout << std::endl;
    server.listen("0.0.0.0", port);
    return 0;
}