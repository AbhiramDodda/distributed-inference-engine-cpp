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
#include <stdexcept>

template<typename Request, typename Response>
class BatchProcessor {
public:
    using BatchCallback = std::function<std::vector<Response>(const std::vector<Request>&)>;

    struct Result {
        Response response;
        int64_t queue_wait_us;
    };

    // `latency_budget` is the target batching deadline for the oldest request.
    // A busy worker can delay processing beyond that deadline.
    BatchProcessor(
        size_t max_batch_size,
        std::chrono::milliseconds latency_budget,
        BatchCallback callback
    );

    ~BatchProcessor();
    Result process(const Request& request);

    void start();
    void stop();
    struct Metrics {
        int64_t total_requests;
        int64_t total_batches;
        int64_t timeout_batches;  // batches flushed at the SLO deadline (partial)
        int64_t full_batches;     // batches flushed because they hit max size
        double avg_batch_size;
    };

    Metrics getMetrics() const;

private:
    // A queued request carries the promise its caller is blocked on plus the
    // time it was enqueued, so the batching loop can bound each request's
    // queueing latency against a fixed budget.
    struct QueuedRequest {
        Request request;
        std::promise<Result> promise;
        std::chrono::steady_clock::time_point arrival;
    };

    void processingLoop();
    void processBatch(
        std::vector<QueuedRequest>& batch,
        bool is_deadline_flush
    );
    size_t max_batch_size_;
    std::chrono::milliseconds latency_budget_;
    BatchCallback callback_;
    std::queue<QueuedRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int64_t> total_requests_{0};
    std::atomic<int64_t> total_batches_{0};
    std::atomic<int64_t> timeout_batches_{0};
    std::atomic<int64_t> full_batches_{0};
    mutable std::mutex metrics_mutex_;
    double avg_batch_size_{0.0};
};

template<typename Request, typename Response>
BatchProcessor<Request, Response>::BatchProcessor(
    size_t max_batch_size,
    std::chrono::milliseconds latency_budget,
    BatchCallback callback
) : max_batch_size_(max_batch_size),
    latency_budget_(latency_budget),
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
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        running_ = false;
    }
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // The processing loop exits on running_ == false. Any request that reached
    // the queue after the loop stopped must be failed explicitly; otherwise
    // its handler would wait forever.
    std::queue<QueuedRequest> abandoned;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        abandoned.swap(request_queue_);
    }
    while (!abandoned.empty()) {
        QueuedRequest item = std::move(abandoned.front());
        abandoned.pop();
        try {
            item.promise.set_exception(
                std::make_exception_ptr(std::runtime_error("batch processor stopped"))
            );
        } catch (const std::future_error&) {
            // A promise can be completed only once. Preserve the first result.
        }
    }
}

template<typename Request, typename Response>
typename BatchProcessor<Request, Response>::Result
BatchProcessor<Request, Response>::process(const Request& request) {
    std::promise<Result> promise;
    std::future<Result> future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!running_) {
            throw std::runtime_error("batch processor is stopped");
        }
        request_queue_.push(QueuedRequest{
            request,
            std::move(promise),
            std::chrono::steady_clock::now()
        });
        total_requests_++;
    }
    queue_cv_.notify_one();
    return future.get();
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::processingLoop() {
    std::vector<QueuedRequest> batch;
    batch.reserve(max_batch_size_);
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Block until there is at least one request to serve (or shutdown).
        queue_cv_.wait(lock, [this] {
            return !request_queue_.empty() || !running_;
        });
        if (!running_) break;

        // Anchor the flush deadline to the OLDEST waiting request's arrival
        // time, not to when this loop happened to start waiting. If the worker
        // was busy running the previous batch, the deadline may
        // already have passed. The budget is not a bound on actual queue wait.
        auto deadline = request_queue_.front().arrival + latency_budget_;

        // Grow the batch until it fills or the deadline passes. Under high load
        // the queue reaches max_batch_size before the deadline (throughput
        // mode: large, full batches). Under low load we flush a small batch the
        // instant the oldest request would blow its budget (latency mode). The
        // effective batching window is therefore adaptive to offered load.
        queue_cv_.wait_until(lock, deadline, [this] {
            return request_queue_.size() >= max_batch_size_ || !running_;
        });

        batch.clear();
        while (!request_queue_.empty() && batch.size() < max_batch_size_) {
            batch.push_back(std::move(request_queue_.front()));
            request_queue_.pop();
        }

        lock.unlock();

        if (!batch.empty()) {
            bool full = batch.size() >= max_batch_size_;
            processBatch(batch, /*is_deadline_flush=*/!full);
        }
    }
}

template<typename Request, typename Response>
void BatchProcessor<Request, Response>::processBatch(
    std::vector<QueuedRequest>& batch,
    bool is_deadline_flush
) {
    if (batch.empty()) return;

    try {
        // Queue wait ends when this batch begins processing, before the callback.
        const auto batch_start = std::chrono::steady_clock::now();
        std::vector<int64_t> queue_waits;
        queue_waits.reserve(batch.size());
        std::vector<Request> requests;
        requests.reserve(batch.size());
        for (auto& item : batch) {
            queue_waits.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                batch_start - item.arrival).count());
            requests.push_back(item.request);
        }
        auto responses = callback_(requests);
        for (size_t i = 0; i < batch.size(); ++i) {
            if (i < responses.size()) {
                batch[i].promise.set_value(Result{std::move(responses[i]), queue_waits[i]});
            } else {
                // If callback returned fewer results, fail the remaining ones
                // rather than letting them hang indefinitely
                batch[i].promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("No response for batched request"))
                );
            }
        }

        // Update metrics
        total_batches_++;

        if (is_deadline_flush) {
            timeout_batches_++;
        } else {
            full_batches_++;
        }

        // Update average batch size (thread-safe)
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        int64_t batches = total_batches_.load();
        avg_batch_size_ = (avg_batch_size_ * (batches - 1) + batch.size()) / batches;

    } catch (const std::exception& e) {
        // Set exception for all promises
        for (auto& item : batch) {
            try {
                item.promise.set_exception(std::current_exception());
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
    return Metrics{
        total_requests_.load(),
        total_batches_.load(),
        timeout_batches_.load(),
        full_batches_.load(),
        avg_batch_size_
    };
}

#endif
