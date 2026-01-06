#ifndef CIRCUIT_BREAKER_H
#define CIRCUIT_BREAKER_H

#include <string>
#include <chrono>
#include <mutex>
#include <atomic>

enum class CircuitState {
    CLOSED,      // Normal operation
    OPEN,        // Failing, reject requests
    HALF_OPEN    // Testing if service recovered
};

class CircuitBreaker {
public:
    CircuitBreaker(
        int failure_threshold = 5,
        int success_threshold = 2,
        std::chrono::seconds timeout = std::chrono::seconds(30)
    );
    
    // Check if request is allowed
    bool allowRequest();
    
    // Record request result
    void recordSuccess();
    void recordFailure();
    
    // Get current state
    CircuitState getState() const;
    std::string getStateString() const;
    
    // Get metrics
    int getFailureCount() const { return failure_count_; }
    int getSuccessCount() const { return success_count_; }
    
private:
    void transitionToOpen();
    void transitionToHalfOpen();
    void transitionToClosed();
    bool shouldAttemptReset();
    
    std::atomic<CircuitState> state_;
    std::atomic<int> failure_count_;
    std::atomic<int> success_count_;
    
    int failure_threshold_;
    int success_threshold_;
    std::chrono::seconds timeout_;
    
    std::chrono::steady_clock::time_point last_failure_time_;
    mutable std::mutex mutex_;
};

#endif // CIRCUIT_BREAKER_H