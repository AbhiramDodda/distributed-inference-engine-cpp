#include "consistent_hash.h"
#include <sstream>

ConsistentHash::ConsistentHash(int virtual_nodes) : virtual_nodes_(virtual_nodes) {}

uint32_t ConsistentHash::hash(const std::string& key) const {
    // FNV-1a hash
    uint32_t h = 2166136261u;
    for (char c : key) {
        h ^= static_cast<uint32_t>(c);
        h *= 16777619u;
    }
    return h;
}

void ConsistentHash::addNode(const std::string& node) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < virtual_nodes_; ++i) {
        std::string vnode = node + "#" + std::to_string(i);
        uint32_t h = hash(vnode);
        ring_[h] = node;
    }
}

void ConsistentHash::removeNode(const std::string& node) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < virtual_nodes_; ++i) {
        std::string vnode = node + "#" + std::to_string(i);
        uint32_t h = hash(vnode);
        ring_.erase(h);
    }
}

std::string ConsistentHash::getNode(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ring_.empty()) {
        return "";
    }
    uint32_t h = hash(key);
    auto it = ring_.lower_bound(h);
    if (it == ring_.end()) {
        it = ring_.begin();
    }
    return it->second;
}

std::vector<std::string> ConsistentHash::getAllNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> nodes;
    std::map<std::string, bool> seen;
    
    for (const auto& pair : ring_) {
        if (!seen[pair.second]) {
            nodes.push_back(pair.second);
            seen[pair.second] = true;
        }
    }
    return nodes;
}

std::map<std::string, int> ConsistentHash::getDistribution(
    const std::vector<std::string>& keys) const {
    std::map<std::string, int> dist;
    
    for (const auto& key : keys) {
        std::string node = getNode(key);
        dist[node]++;
    }
    return dist;
}
