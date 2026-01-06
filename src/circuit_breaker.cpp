#include "circuit_breaker.h"

CircuitBreaker::CircuitBreaker(
    int failure_threshold,
    int success_threshold,
    std::chrono::seconds timeout
) : state_(CircuitState::CLOSED),
    failure_count_(0),
    success_count_(0),
    failure_threshold_(failure_threshold),
    success_threshold_(success_threshold),
    timeout_(timeout),
    last_failure_time_(std::chrono::steady_clock::now()) {}

bool CircuitBreaker::allowRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CircuitState current_state = state_.load();
    
    switch (current_state) {
        case CircuitState::CLOSED:
            return true;
            
        case CircuitState::OPEN:
            if (shouldAttemptReset()) {
                transitionToHalfOpen();
                return true;
            }
            return false;
            
        case CircuitState::HALF_OPEN:
            return true;
            
        default:
            return false;
    }
}

void CircuitBreaker::recordSuccess() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CircuitState current_state = state_.load();
    
    if (current_state == CircuitState::HALF_OPEN) {
        success_count_++;
        
        if (success_count_ >= success_threshold_) {
            transitionToClosed();
        }
    } else if (current_state == CircuitState::CLOSED) {
        // Reset failure count on success
        failure_count_ = 0;
    }
}

void CircuitBreaker::recordFailure() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CircuitState current_state = state_.load();
    
    last_failure_time_ = std::chrono::steady_clock::now();
    
    if (current_state == CircuitState::HALF_OPEN) {
        // Failure in half-open state, go back to open
        transitionToOpen();
    } else if (current_state == CircuitState::CLOSED) {
        failure_count_++;
        
        if (failure_count_ >= failure_threshold_) {
            transitionToOpen();
        }
    }
}

CircuitState CircuitBreaker::getState() const {
    return state_.load();
}

std::string CircuitBreaker::getStateString() const {
    switch (state_.load()) {
        case CircuitState::CLOSED: return "CLOSED";
        case CircuitState::OPEN: return "OPEN";
        case CircuitState::HALF_OPEN: return "HALF_OPEN";
        default: return "UNKNOWN";
    }
}

void CircuitBreaker::transitionToOpen() {
    state_ = CircuitState::OPEN;
    success_count_ = 0;
}

void CircuitBreaker::transitionToHalfOpen() {
    state_ = CircuitState::HALF_OPEN;
    failure_count_ = 0;
    success_count_ = 0;
}

void CircuitBreaker::transitionToClosed() {
    state_ = CircuitState::CLOSED;
    failure_count_ = 0;
    success_count_ = 0;
}

bool CircuitBreaker::shouldAttemptReset() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_failure_time_
    );
    
    return elapsed >= timeout_;
}