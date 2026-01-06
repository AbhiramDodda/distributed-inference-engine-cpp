#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <chrono>
#include <future>
#include <atomic>

template<typename Request, typename Response>
class BatchProcessor {
public:
    using BatchCallback = std::function<std::vector<Response>(const std::vector<Request>&)>;
    
    BatchProcessor(
        size_t max_batch_size,
        std::chrono::milliseconds timeout,
        BatchCallback callback
    );
    
    ~BatchProcessor();
    
    // Process single request (batched internally)
    Response process(const Request& request);
    
    void start();
    void stop();
    
    // Metrics
    struct Metrics {
        std::atomic<int64_t> total_requests{0};
        std::atomic<int64_t> total_batches{0};
        std::atomic<int64_t> timeout_batches{0};
        std::atomic<int64_t> full_batches{0};
        double avg_batch_size{0.0};
    };
    
    Metrics getMetrics() const;
    
private:
    void processingLoop();
    void processBatch(
        std::vector<std::pair<Request, std::promise<Response>>>& batch,
        bool is_timeout
    );
    
    size_t max_batch_size_;
    std::chrono::milliseconds timeout_;
    BatchCallback callback_;
    
    std::queue<std::pair<Request, std::promise<Response>>> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    
    Metrics metrics_;
    mutable std::mutex metrics_mutex_;
};

// Template implementation
template<typename Request, typename Response>
BatchProcessor<Request, Response>::BatchProcessor(
    size_t max_batch_size,
    std::chrono::milliseconds timeout,
    BatchCallback callback
) : max_batch_size_(max_batch_size),
    timeout_(timeout),
    callback_(callback) {}

template<typename Request, typename Response>
BatchProcessor<Request, Response>::~BatchProcessor() {
    stop();
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::start() {
    running_ = true;
    worker_thread_ = std::thread(&BatchProcessor::processingLoop, this);
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::stop() {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

template<typename Request, typename Response>
Response BatchProcessor<Request, Response>::process(const Request& request) {
    std::promise<Response> promise;
    std::future<Response> future = promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push({request, std::move(promise)});
        metrics_.total_requests++;
    }
    
    queue_cv_.notify_one();
    
    // Wait for result
    return future.get();
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::processingLoop() {
    std::vector<std::pair<Request, std::promise<Response>>> batch;
    batch.reserve(max_batch_size_);
    
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for requests or timeout
        bool timeout = !queue_cv_.wait_for(
            lock,
            timeout_,
            [this] { return !request_queue_.empty() || !running_; }
        );
        
        if (!running_) break;
        
        // Collect batch
        batch.clear();
        while (!request_queue_.empty() && batch.size() < max_batch_size_) {
            batch.push_back(std::move(request_queue_.front()));
            request_queue_.pop();
        }
        
        lock.unlock();
        
        if (!batch.empty()) {
            processBatch(batch, timeout);
        }
    }
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::processBatch(
    std::vector<std::pair<Request, std::promise<Response>>>& batch,
    bool is_timeout
) {
    if (batch.empty()) return;
    
    try {
        // Extract requests
        std::vector<Request> requests;
        requests.reserve(batch.size());
        for (auto& item : batch) {
            requests.push_back(item.first);
        }
        
        // Process batch
        auto responses = callback_(requests);
        
        // Fulfill promises
        for (size_t i = 0; i < batch.size() && i < responses.size(); ++i) {
            batch[i].second.set_value(responses[i]);
        }
        
        // Update metrics
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.total_batches++;
        
        if (is_timeout) {
            metrics_.timeout_batches++;
        } else {
            metrics_.full_batches++;
        }
        
        // Update average batch size
        double prev_total = metrics_.avg_batch_size * (metrics_.total_batches - 1);
        metrics_.avg_batch_size = (prev_total + batch.size()) / metrics_.total_batches;
        
    } catch (const std::exception& e) {
        // Set exception for all promises
        for (auto& item : batch) {
            try {
                item.second.set_exception(std::current_exception());
            } catch (...) {
                // Promise already set or moved
            }
        }
    }
}

template<typename Request, typename Response>
typename BatchProcessor<Request, Response>::Metrics 
BatchProcessor<Request, Response>::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

#endif 