#include "inference_engine.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

// Keep ONNX Runtime's intra-op pool within the host's CPU capacity. The
// previous hardcoded value of 4 oversubscribed 2-vCPU hosts (for example
// c6i.large), where two threads measured about twice as fast as four. The cap
// preserves the existing behavior on hosts with four or more CPUs. The
// environment override exists for experiments and operational tuning.
int resolveIntraOpThreads() {
    if (const char* value = std::getenv("ORT_INTRA_THREADS")) {
        try {
            const int requested = std::stoi(value);
            if (requested > 0) {
                return requested;
            }
        } catch (const std::exception&) {
            // Use the capacity-aware default below.
        }
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, std::min(4u, hardware == 0 ? 1u : hardware)));
}

}  // namespace

InferenceEngine::InferenceEngine(const std::string& model_path, int shard_id)
    : model_path_(model_path), shard_id_(shard_id) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine");
    initializeSession();
}

InferenceEngine::~InferenceEngine() {
    // cleanup handled by unique_ptr
}

void InferenceEngine::initializeSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    session_options_ = std::make_unique<Ort::SessionOptions>();
    const int intra_op_threads = resolveIntraOpThreads();
    session_options_->SetIntraOpNumThreads(intra_op_threads);
    std::cout << "ORT intra-op threads: " << intra_op_threads
              << " (hardware_concurrency=" << std::thread::hardware_concurrency() << ")"
              << std::endl;
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_->AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "CUDA Provider successfully loaded." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "CUDA failed to load: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU Provider..." << std::endl;
    }
    // session creation
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_options_);
    
    // input info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    
    if (num_input_nodes > 0) {
        // Get input name - FIXED: Store the string, not just the pointer
        auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
        input_name_strings_.push_back(std::string(input_name_allocated.get()));
        input_names_.push_back(input_name_strings_[0].c_str());
        // input shape
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        // A static first dimension is the model's maximum batch size. Dynamic
        // batch models (-1) keep the large default and let the batch processor
        // choose. This prevents a batch >1 from being sent to a static batch-1
        // model, which ONNX Runtime rejects.
        if (!input_shape_.empty()) {
            if (input_shape_[0] > 0) {
                max_batch_size_ = static_cast<size_t>(input_shape_[0]);
            } else {
                max_batch_size_ = 1024;
            }
        }
        // Handle dynamic dimensions (-1)
        for (auto& dim : input_shape_) {
            if (dim == -1) {
                dim = 1; // Default batch size
            }
        }
    }
    size_t num_output_nodes = session_->GetOutputCount();
    
    if (num_output_nodes > 0) {
        // Get output name - FIXED: Store the string, not just the pointer
        auto output_name_allocated = session_->GetOutputNameAllocated(0, allocator);
        output_name_strings_.push_back(std::string(output_name_allocated.get()));
        output_names_.push_back(output_name_strings_[0].c_str());
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = tensor_info.GetShape();
        //dynamic dimensions
        for (auto& dim : output_shape_) {
            if (dim == -1) {
                dim = 1;
            }
        }
    }
    
    std::cout << "ONNX model loaded: " << model_path_ << std::endl;
    std::cout << "  Input name: " << (input_names_.empty() ? "NONE" : input_names_[0]) << std::endl;
    std::cout << "  Input shape: [";
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        std::cout << input_shape_[i];
        if (i < input_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Output name: " << (output_names_.empty() ? "NONE" : output_names_[0]) << std::endl;
    std::cout << "  Output shape: [";
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::cout << output_shape_[i];
        if (i < output_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

std::vector<float> InferenceEngine::predict(const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int64_t expected_size = std::accumulate(
        input_shape_.begin(), 
        input_shape_.end(), 
        1LL, 
        std::multiplies<int64_t>()
    );
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_copy = input;
    if (static_cast<int64_t>(input_copy.size()) != expected_size) {
        input_copy.resize(expected_size, 0.0f);
    }
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_copy.data(),
        input_copy.size(),
        input_shape_.data(),
        input_shape_.size()
    );
    
    // inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        1
    );
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t output_size = std::accumulate(
        output_shape.begin(),
        output_shape.end(),
        1LL,
        std::multiplies<int64_t>()
    );
    std::vector<float> result(output_data, output_data + output_size);
    return result;
}

std::vector<std::vector<float>> InferenceEngine::batchPredict(
    const std::vector<std::vector<float>>& inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inputs.empty()) {
        return {};
    }
    
    size_t batch_size = inputs.size();
    
    // input size per sample
    int64_t per_sample_size = std::accumulate(
        input_shape_.begin() + 1,  // batch dimension skipped
        input_shape_.end(),
        1LL,
        std::multiplies<int64_t>()
    );
    
    // Flatten each sample into exactly one model-sized slot. Padding short
    // inputs is intentional; truncating oversized inputs prevents them from
    // shifting every later sample in the batch.
    if (per_sample_size <= 0) {
        throw std::runtime_error("model input shape has no per-sample elements");
    }
    const size_t sample_size = static_cast<size_t>(per_sample_size);
    std::vector<float> batch_input;
    batch_input.reserve(batch_size * sample_size);
    for (const auto& input : inputs) {
        const size_t copy_size = std::min(input.size(), sample_size);
        batch_input.insert(batch_input.end(), input.begin(), input.begin() + copy_size);
        if (copy_size < sample_size) {
            batch_input.resize(batch_input.size() + (sample_size - copy_size), 0.0f);
        }
    }
    
    // batch input shape creation
    std::vector<int64_t> batch_input_shape = input_shape_;
    batch_input_shape[0] = batch_size;  // batch dimension set
    // input tensor creation
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        batch_input.data(),
        batch_input.size(),
        batch_input_shape.data(),
        batch_input_shape.size()
    );
    
    // inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        1
    );
    
    if (output_tensors.empty()) {
        throw std::runtime_error("model returned no output tensors");
    }
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.empty() || output_shape[0] != static_cast<int64_t>(batch_size)) {
        throw std::runtime_error("model output batch dimension does not match the input batch");
    }

    int64_t per_output_size = std::accumulate(
        output_shape.begin() + 1,  // batch dimension skipped
        output_shape.end(),
        1LL,
        std::multiplies<int64_t>()
    );
    
    // batch output is split into individual results
    std::vector<std::vector<float>> results;
    results.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        size_t offset = i * per_output_size;
        std::vector<float> result(
            output_data + offset,
            output_data + offset + per_output_size
        );
        results.push_back(std::move(result));
    }
    
    return results;
}

std::vector<int64_t> InferenceEngine::getInputShape() const {
    return input_shape_;
}

std::vector<int64_t> InferenceEngine::getOutputShape() const {
    return output_shape_;
}

size_t InferenceEngine::maxBatchSize() const {
    return max_batch_size_;
}