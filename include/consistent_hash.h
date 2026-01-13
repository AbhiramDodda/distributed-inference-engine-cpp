#ifndef CONSISTENT_HASH_H
#define CONSISTENT_HASH_H

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cstdint>

class ConsistentHash {
public:
    explicit ConsistentHash(int virtual_nodes = 150);
    void addNode(const std::string& node);
    void removeNode(const std::string& node);
    std::string getNode(const std::string& key) const;
    std::vector<std::string> getAllNodes() const;
    std::map<std::string, int> getDistribution(const std::vector<std::string>& keys) const;
    
private:
    uint32_t hash(const std::string& key) const;
    int virtual_nodes_;
    std::map<uint32_t, std::string> ring_;
    mutable std::mutex mutex_;
};

#endif