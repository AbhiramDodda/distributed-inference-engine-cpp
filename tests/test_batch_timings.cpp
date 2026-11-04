#include "batch_processor.h"

#include <stdexcept>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void require(bool condition) {
    if (!condition) throw std::runtime_error("batch timing check failed");
}

int main() {
    // Hold the first callback so a second request must wait behind it.
    std::promise<void> first_started;
    std::promise<void> release_first;
    auto release = release_first.get_future().share();
    BatchProcessor<int, int> processor(1, 0ms, [&](const std::vector<int>& values) {
        if (values[0] == 1) {
            first_started.set_value();
            release.wait();
        }
        return values;
    });
    processor.start();
    auto first = std::async(std::launch::async, [&] { return processor.process(1); });
    if (first_started.get_future().wait_for(2s) != std::future_status::ready) {
        release_first.set_value();
        throw std::runtime_error("first batch did not start");
    }
    auto second = std::async(std::launch::async, [&] { return processor.process(2); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (processor.getMetrics().total_requests < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool second_enqueued = processor.getMetrics().total_requests == 2;
    // Keep the callback blocked for a known interval after enqueue observation.
    const auto observed = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < observed + 1ms) {
        std::this_thread::yield();
    }
    release_first.set_value();
    require(second_enqueued);
    require(first.wait_for(2s) == std::future_status::ready);
    require(second.wait_for(2s) == std::future_status::ready);
    auto first_result = first.get();
    auto second_result = second.get();
    require(first_result.response == 1);
    require(first_result.queue_wait_us >= 0);
    require(second_result.response == 2);
    require(second_result.queue_wait_us > 0);
    processor.stop();

    // Two queued requests share one callback without mixing response values.
    BatchProcessor<int, int> paired(2, 2s, [](const std::vector<int>& values) {
        require(values.size() == 2);
        return values;
    });
    paired.start();
    auto a = std::async(std::launch::async, [&] { return paired.process(10); });
    auto b = std::async(std::launch::async, [&] { return paired.process(20); });
    require(a.wait_for(3s) == std::future_status::ready);
    require(b.wait_for(3s) == std::future_status::ready);
    require(a.get().response == 10);
    require(b.get().response == 20);
    paired.stop();

    // Shutdown fails queued requests and rejects new work instead of hanging.
    {
        std::promise<void> started;
        std::promise<void> release;
        auto release_future = release.get_future().share();
        BatchProcessor<int, int> shutdown_processor(1, 2s, [&](const std::vector<int>& values) {
            if (values[0] == 1) {
                started.set_value();
                release_future.wait();
            }
            return values;
        });
        shutdown_processor.start();
        auto first = std::async(std::launch::async, [&] { return shutdown_processor.process(1); });
        require(started.get_future().wait_for(2s) == std::future_status::ready);
        auto second = std::async(std::launch::async, [&] { return shutdown_processor.process(2); });
        const auto enqueue_deadline = std::chrono::steady_clock::now() + 2s;
        while (shutdown_processor.getMetrics().total_requests < 2 &&
               std::chrono::steady_clock::now() < enqueue_deadline) {
            std::this_thread::yield();
        }
        require(shutdown_processor.getMetrics().total_requests == 2);
        auto stopping = std::async(std::launch::async, [&] { shutdown_processor.stop(); });
        const auto stop_deadline = std::chrono::steady_clock::now() + 2s;
        while (shutdown_processor.isRunning() &&
               std::chrono::steady_clock::now() < stop_deadline) {
            std::this_thread::yield();
        }
        require(!shutdown_processor.isRunning());
        auto concurrent_stop = std::async(std::launch::async, [&] { shutdown_processor.stop(); });
        release.set_value();
        require(first.wait_for(2s) == std::future_status::ready);
        require(stopping.wait_for(2s) == std::future_status::ready);
        require(concurrent_stop.wait_for(2s) == std::future_status::ready);
        bool second_failed = false;
        try {
            second.get();
        } catch (const std::runtime_error&) {
            second_failed = true;
        }
        require(second_failed);
        bool new_work_rejected = false;
        try {
            shutdown_processor.process(3);
        } catch (const std::runtime_error&) {
            new_work_rejected = true;
        }
        require(new_work_rejected);
    }
}
