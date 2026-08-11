#ifndef ENGINE_BACKEND_H
#define ENGINE_BACKEND_H

#include <memory>
#include <string>
#include <vector>

// Backend-agnostic inference interface the worker talks to. Lets the worker run
// either the ONNX Runtime engine or the tensorforge CompiledEngine unchanged.
struct IEngine {
    virtual ~IEngine() = default;
    virtual std::vector<std::vector<float>> batchPredict(
        const std::vector<std::vector<float>>& inputs) = 0;
};

// backend: "ort" (default) or "compiled". For "compiled", build_dir points at a
// tensorforge build directory (plan.json + weights.bin); requires the binary to
// be built with -DWITH_COMPILED_BACKEND=ON.
std::unique_ptr<IEngine> make_engine(const std::string& backend,
                                     const std::string& model_path,
                                     const std::string& build_dir,
                                     int shard_id);

#endif
