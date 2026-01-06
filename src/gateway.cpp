#include "consistent_hash.h"
#include "circuit_breaker.h"
#include <iostream>
#include <memory>
#include <map>
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Gateway {
public:
    explicit Gateway(const std::vector<std::string>& workers) {
        // Initialize consistent hash
        for (const auto& worker : workers) {
            hash_ring_.addNode(worker);
            
            // Create circuit breaker for each worker
            circuit_breakers_[worker] = std::make_unique<CircuitBreaker>(
                5,   // failure_threshold
                2,   // success_threshold
                std::chrono::seconds(30)
            );
            
            // Create HTTP client for each worker
            auto url_parts = parseUrl(worker);
            clients_[worker] = std::make_unique<httplib::Client>(
                url_parts.first, url_parts.second
            );
            clients_[worker]->set_connection_timeout(5, 0);  // 5 seconds
            clients_[worker]->set_read_timeout(5, 0);
            
            std::cout << "Connected to worker: " << worker << std::endl;
        }
    }
    
    json routeRequest(const json& request) {
        std::string request_id = request["request_id"];
        
        // Get target node using consistent hashing
        std::string target_node = hash_ring_.getNode(request_id);
        
        if (target_node.empty()) {
            throw std::runtime_error("No workers available");
        }
        
        // Try primary node with circuit breaker
        auto result = tryNode(target_node, request);
        if (result.has_value()) {
            return *result;
        }
        
        // Primary failed, try other nodes
        auto all_nodes = hash_ring_.getAllNodes();
        for (const auto& node : all_nodes) {
            if (node != target_node) {
                auto retry_result = tryNode(node, request);
                if (retry_result.has_value()) {
                    return *retry_result;
                }
            }
        }
        
        throw std::runtime_error("All workers failed or circuit breakers open");
    }
    
    json getStats() {
        json stats;
        stats["total_workers"] = hash_ring_.getAllNodes().size();
        
        json circuit_states = json::array();
        for (const auto& [node, breaker] : circuit_breakers_) {
            json state;
            state["node"] = node;
            state["state"] = breaker->getStateString();
            state["failures"] = breaker->getFailureCount();
            state["successes"] = breaker->getSuccessCount();
            circuit_states.push_back(state);
        }
        stats["circuit_breakers"] = circuit_states;
        
        return stats;
    }
    
private:
    std::optional<json> tryNode(const std::string& node, const json& request) {
        auto breaker_it = circuit_breakers_.find(node);
        if (breaker_it == circuit_breakers_.end()) {
            return std::nullopt;
        }
        
        auto& breaker = breaker_it->second;
        
        // Check circuit breaker
        if (!breaker->allowRequest()) {
            std::cout << "Circuit breaker OPEN for " << node << ", skipping" << std::endl;
            return std::nullopt;
        }
        
        // Try request
        auto client_it = clients_.find(node);
        if (client_it == clients_.end()) {
            breaker->recordFailure();
            return std::nullopt;
        }
        
        try {
            auto result = client_it->second->Post(
                "/infer",
                request.dump(),
                "application/json"
            );
            
            if (result && result->status == 200) {
                breaker->recordSuccess();
                return json::parse(result->body);
            } else {
                breaker->recordFailure();
                return std::nullopt;
            }
        } catch (const std::exception& e) {
            std::cerr << "Request to " << node << " failed: " << e.what() << std::endl;
            breaker->recordFailure();
            return std::nullopt;
        }
    }
    
    std::pair<std::string, int> parseUrl(const std::string& url) {
        // Simple URL parser for localhost:port format
        size_t colon_pos = url.find_last_of(':');
        if (colon_pos == std::string::npos) {
            return {"localhost", 8080};
        }
        
        std::string host = url.substr(0, colon_pos);
        int port = std::stoi(url.substr(colon_pos + 1));
        
        // Remove http:// if present
        size_t proto_pos = host.find("://");
        if (proto_pos != std::string::npos) {
            host = host.substr(proto_pos + 3);
        }
        
        return {host, port};
    }
    
    ConsistentHash hash_ring_;
    std::map<std::string, std::unique_ptr<CircuitBreaker>> circuit_breakers_;
    std::map<std::string, std::unique_ptr<httplib::Client>> clients_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <worker1:port> [worker2:port] ..." << std::endl;
        return 1;
    }
    
    std::vector<std::string> workers;
    for (int i = 1; i < argc; ++i) {
        workers.push_back(argv[i]);
    }
    
    Gateway gateway(workers);
    httplib::Server server;
    
    // Inference endpoint
    server.Post("/infer", [&gateway](const httplib::Request& req, httplib::Response& res) {
        try {
            auto request = json::parse(req.body);
            auto response = gateway.routeRequest(request);
            
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error;
            error["error"] = e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });
    
    // Stats endpoint
    server.Get("/stats", [&gateway](const httplib::Request&, httplib::Response& res) {
        auto stats = gateway.getStats();
        res.set_content(stats.dump(), "application/json");
    });
    
    std::cout << "Gateway listening on port 8000" << std::endl;
    std::cout << "Workers: " << workers.size() << std::endl;
    std::cout << "Circuit breakers enabled" << std::endl;
    std::cout << "Ready!" << std::endl;
    
    server.listen("0.0.0.0", 8000);
    
    return 0;
}