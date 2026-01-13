#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path, int shard_id = 0);
    ~InferenceEngine();
    std::vector<float> predict(const std::vector<float>& input);
    std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& inputs
    );
    const std::string& getModelPath() const { return model_path_; }
    int getShardId() const { return shard_id_; }
    std::vector<int64_t> getInputShape() const;
    std::vector<int64_t> getOutputShape() const;
    
private:
    void initializeSession();
    
    std::string model_path_;
    int shard_id_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::mutex mutex_;
};

#endif