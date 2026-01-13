#include "circuit_breaker.h"

CircuitBreaker::CircuitBreaker(int failure_threshold, int success_threshold, std::chrono::seconds timeout)
    : state_(CircuitState::CLOSED),
      failure_count_(0),
      success_count_(0),
      failure_threshold_(failure_threshold),
      success_threshold_(success_threshold),
      timeout_(timeout),
      last_failure_time_(std::chrono::steady_clock::now()) {}

bool CircuitBreaker::allowRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == CircuitState::OPEN) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_failure_time_ >= timeout_) {
            state_ = CircuitState::HALF_OPEN;
            success_count_ = 0;
            return true;
        }
        return false;
    }
    return true;
}

void CircuitBreaker::recordSuccess() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == CircuitState::HALF_OPEN) {
        success_count_++;
        if (success_count_ >= success_threshold_) {
            state_ = CircuitState::CLOSED;
            failure_count_ = 0;
        }
    } else {
        failure_count_ = 0;
    }
}

void CircuitBreaker::recordFailure() {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_count_++;
    last_failure_time_ = std::chrono::steady_clock::now();
    
    if (failure_count_ >= failure_threshold_ || state_ == CircuitState::HALF_OPEN) {
        state_ = CircuitState::OPEN;
    }
}

CircuitState CircuitBreaker::getState() const {
    return state_.load();
}

std::string CircuitBreaker::getStateString() const {
    CircuitState s = getState();
    switch (s) {
        case CircuitState::CLOSED: return "CLOSED";
        case CircuitState::OPEN: return "OPEN";
        case CircuitState::HALF_OPEN: return "HALF_OPEN";
        default: return "UNKNOWN";
    }
}

int CircuitBreaker::getSuccessCount() const {
    return success_count_.load();
}

int CircuitBreaker::getFailureCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_count_;
}