#include "engine_backend.h"
#include "inference_engine.h"
#include <stdexcept>

#ifdef WITH_COMPILED_BACKEND
#include "tensorforge/compiled_engine.h"
#endif

namespace {

// Wraps the ONNX Runtime engine.
struct OrtEngine : IEngine {
    InferenceEngine eng;
    OrtEngine(const std::string& model_path, int shard_id)
        : eng(model_path, shard_id) {}
    std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& in) override {
        return eng.batchPredict(in);
    }
};

#ifdef WITH_COMPILED_BACKEND
// Wraps the tensorforge CompiledEngine (loads plan.json + weights.bin).
struct CompiledEngineAdapter : IEngine {
    tensorforge::CompiledEngine eng;
    explicit CompiledEngineAdapter(const std::string& build_dir) : eng(build_dir) {}
    std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& in) override {
        return eng.batchPredict(in);
    }
};
#endif

}  // namespace

std::unique_ptr<IEngine> make_engine(const std::string& backend,
                                     const std::string& model_path,
                                     const std::string& build_dir, int shard_id) {
    if (backend == "compiled") {
#ifdef WITH_COMPILED_BACKEND
        if (build_dir.empty())
            throw std::runtime_error("ENGINE_BACKEND=compiled requires TF_BUILD_DIR");
        return std::make_unique<CompiledEngineAdapter>(build_dir);
#else
        throw std::runtime_error(
            "compiled backend not built (rebuild with -DWITH_COMPILED_BACKEND=ON)");
#endif
    }
    return std::make_unique<OrtEngine>(model_path, shard_id);
}
