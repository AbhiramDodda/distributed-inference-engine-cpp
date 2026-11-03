#ifndef ENGINE_BACKEND_H
#define ENGINE_BACKEND_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Backend-agnostic inference interface the worker talks to. Lets the worker run
// either the ONNX Runtime engine or the tensorforge CompiledEngine unchanged.
struct IEngine {
    virtual ~IEngine() = default;
    virtual std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& inputs) = 0;
    // Conservative default for backends that cannot report a static batch
    // limit. The ORT backend overrides this with the model's actual limit.
    virtual size_t maxBatchSize() const { return 1; }
    virtual const char* name() const = 0;   // active backend, for logs / health
};

// backend: "" or "ort" -> ONNX Runtime (the default); "compiled" -> tensorforge
// CompiledEngine (build_dir = a tensorforge build dir; needs -DWITH_COMPILED_BACKEND).
// Any other name throws (fail loudly rather than silently defaulting).
std::unique_ptr<IEngine> make_engine(const std::string& backend,
                                     const std::string& model_path,
                                     const std::string& build_dir,
                                     int shard_id);

#endif
