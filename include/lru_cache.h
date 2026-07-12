#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <unordered_map>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <cstdint>
#include <cstring>

template<typename Key, typename Value, typename Hash = std::hash<Key>>
class LRUCache {
public:
    explicit LRUCache(size_t capacity) 
        : capacity_(capacity), hits_(0), misses_(0) {}
    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            misses_++;
            return std::nullopt;
        }
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        hits_++;
        return it->second->second;
    }
    void put(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {

            it->second->second = value;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }
        // new entry
        if (cache_list_.size() >= capacity_) {
            // kick least recently used
            auto last = cache_list_.back();
            cache_map_.erase(last.first);
            cache_list_.pop_back();
        }
        cache_list_.emplace_front(key, value);
        cache_map_[key] = cache_list_.begin();
    }
    
    // clear cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_map_.clear();
        cache_list_.clear();
        hits_ = 0;
        misses_ = 0;
    }
    
    // stats
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
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator, Hash> cache_map_;
    
    mutable std::mutex mutex_;
    std::atomic<size_t> hits_;
    std::atomic<size_t> misses_;
};

// hash for vector<float> to use as cache key
struct VectorHash {
    size_t operator()(const std::vector<float>& vec) const {
        // FNV-1a over the raw bits of every element. Hashing the whole vector
        // (rather than sampling a few elements) keeps distinct inputs from
        // colliding into the same bucket, which would otherwise force an O(n)
        // full-vector equality check on every lookup. Matches the routing-key
        // hash used by the gateway so the two views of "the same input" agree.
        uint64_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
        for (float f : vec) {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;  // FNV-1a 64-bit prime
        }
        return static_cast<size_t>(h);
    }
};

#endif 