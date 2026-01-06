#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"
#include "inference_engine.h"
#include "batch_processor.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <atomic>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceRequest;
using inference::InferenceResponse;
using inference::HealthCheckRequest;
using inference::HealthCheckResponse;

class InferenceServiceImpl final : public InferenceService::Service {
public:
    InferenceServiceImpl(const std::string& node_id, int port)
        : node_id_(node_id),
          engine_("model.onnx", port % 3),
          batch_processor_(32, 20, // max_batch=32, timeout=20ms
              [this](const std::vector<InferenceRequest>& reqs) {
                  return this->processBatch(reqs);
              }) {
        batch_processor_.start();
    }
    
    ~InferenceServiceImpl() {
        batch_processor_.stop();
    }
    
    Status Predict(ServerContext* context,
                   const InferenceRequest* request,
                   InferenceResponse* response) override {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        total_requests_++;
        active_requests_++;
        
        try {
            // Use batch processor for dynamic batching
            auto result = batch_processor_.process(*request);
            *response = result;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
            
            response->set_inference_time_us(duration);
            response->set_node_id(node_id_);
            response->set_request_id(request->request_id());
            
        } catch (const std::exception& e) {
            active_requests_--;
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
        
        active_requests_--;
        return Status::OK;
    }
    
    Status HealthCheck(ServerContext* context,
                       const HealthCheckRequest* request,
                       HealthCheckResponse* response) override {
        response->set_healthy(true);
        response->set_active_requests(active_requests_.load());
        response->set_cpu_usage(0.5f); // Simulated
        response->set_total_requests(total_requests_.load());
        return Status::OK;
    }
    
private:
    std::vector<InferenceResponse> processBatch(
        const std::vector<InferenceRequest>& requests) {
        
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<int>> shapes;
        
        for (const auto& req : requests) {
            inputs.push_back(
                std::vector<float>(req.input_data().begin(), req.input_data().end())
            );
            shapes.push_back(
                std::vector<int>(req.input_shape().begin(), req.input_shape().end())
            );
        }
        
        auto results = engine_.batchPredict(inputs, shapes);
        
        std::vector<InferenceResponse> responses;
        for (size_t i = 0; i < results.size(); ++i) {
            InferenceResponse resp;
            for (float val : results[i]) {
                resp.add_output_data(val);
            }
            resp.add_output_shape(results[i].size());
            responses.push_back(resp);
        }
        
        return responses;
    }
    
    std::string node_id_;
    InferenceEngine engine_;
    BatchProcessor<InferenceRequest, InferenceResponse> batch_processor_;
    std::atomic<int> active_requests_{0};
    std::atomic<int64_t> total_requests_{0};
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <port>" << std::endl;
        return 1;
    }
    
    int port = std::stoi(argv[1]);
    std::string node_id = "worker_" + std::to_string(port);
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    
    InferenceServiceImpl service(node_id, port);
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Worker node " << node_id << " listening on " << server_address << std::endl;
    
    server->Wait();
    return 0;
}