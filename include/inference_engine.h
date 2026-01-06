#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <vector>
#include <string>
#include <random>
#include <mutex>

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_name, int shard_id = 0);
    
    std::vector<float> predict(const std::vector<float>& input);
    
    std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& inputs
    );
    
    const std::string& getModelName() const { return model_name_; }
    int getShardId() const { return shard_id_; }
    
private:
    void simulateComputation(size_t ops_count);
    
    std::string model_name_;
    int shard_id_;
    int num_classes_;
    
    std::mt19937 rng_;
    std::mutex mutex_;
};

#endif 