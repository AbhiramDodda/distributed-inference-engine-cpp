#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <unordered_map>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <functional>

// LRU Cache for inference results
template<typename Key, typename Value>
class LRUCache {
public:
    explicit LRUCache(size_t capacity) 
        : capacity_(capacity), hits_(0), misses_(0) {}
    
    // Get value from cache
    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            misses_++;
            return std::nullopt;
        }
        
        // Move to front (most recently used)
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        
        hits_++;
        return it->second->second;
    }
    
    // Put value in cache
    void put(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        
        if (it != cache_map_.end()) {
            // Update existing entry
            it->second->second = value;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }
        
        // Add new entry
        if (cache_list_.size() >= capacity_) {
            // Evict least recently used
            auto last = cache_list_.back();
            cache_map_.erase(last.first);
            cache_list_.pop_back();
        }
        
        cache_list_.emplace_front(key, value);
        cache_map_[key] = cache_list_.begin();
    }
    
    // Clear cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_map_.clear();
        cache_list_.clear();
        hits_ = 0;
        misses_ = 0;
    }
    
    // Get statistics
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_list_.size();
    }
    
    size_t capacity() const { return capacity_; }
    
    size_t getHits() const { return hits_.load(); }
    size_t getMisses() const { return misses_.load(); }
    
    double getHitRate() const {
        size_t total = hits_ + misses_;
        return total > 0 ? (double)hits_ / total : 0.0;
    }
    
private:
    size_t capacity_;
    std::list<std::pair<Key, Value>> cache_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> cache_map_;
    
    mutable std::mutex mutex_;
    std::atomic<size_t> hits_;
    std::atomic<size_t> misses_;
};

// Hash function for vector<float> to use as cache key
struct VectorHash {
    size_t operator()(const std::vector<float>& vec) const {
        size_t hash = 0;
        std::hash<float> hasher;
        
        // Sample hash - hash first, last, and middle elements
        if (!vec.empty()) {
            hash ^= hasher(vec[0]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hasher(vec[vec.size()/2]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hasher(vec.back()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        
        return hash;
    }
};

#endif 