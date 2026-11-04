#ifndef SHUTDOWN_SIGNAL_H
#define SHUTDOWN_SIGNAL_H

#include <csignal>
#include <pthread.h>
#include <stdexcept>

// Block SIGINT/SIGTERM in the process before worker or ORT threads start. A
// dedicated wait then receives the signal without interrupting a thread pool.
inline void block_shutdown_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) {
        throw std::runtime_error("failed to block shutdown signals");
    }
}

inline int wait_for_shutdown_signal() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    int signal_number = 0;
    if (sigwait(&set, &signal_number) != 0) {
        throw std::runtime_error("failed to wait for a shutdown signal");
    }
    return signal_number;
}

#endif
