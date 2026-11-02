#include "inference_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float max_difference(const std::vector<float>& left, const std::vector<float>& right) {
    require(left.size() == right.size(), "output dimension mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) {
        result = std::max(result, std::fabs(left[i] - right[i]));
    }
    return result;
}

}  // namespace

int main() {
    const char* model = std::getenv("BENCH_MODEL");
    if (!model || !*model) {
        std::cout << "SKIP: set BENCH_MODEL to run batch-shape checks\n";
        return 0;
    }

    InferenceEngine engine(model, 0);
    std::vector<float> oversized(200000);
    for (size_t i = 0; i < oversized.size(); ++i) {
        oversized[i] = 1.0f + static_cast<float>(i % 97) * 0.01f;
    }
    std::vector<float> small{1.0f, 2.0f, 3.0f};
    std::vector<float> another_small{4.0f, 5.0f, 6.0f};

    const auto reference_oversized = engine.batchPredict({oversized});
    const auto reference_small = engine.batchPredict({small});
    const auto reference_another = engine.batchPredict({another_small});

    const auto oversized_first = engine.batchPredict({oversized, small});
    require(oversized_first.size() == 2, "expected two outputs");
    require(
        max_difference(oversized_first[0], reference_oversized[0]) <= 1e-4f,
        "oversized sample output changed"
    );
    require(
        max_difference(oversized_first[1], reference_small[0]) <= 1e-4f,
        "sample after an oversized input was misaligned"
    );

    const auto small_first = engine.batchPredict({small, another_small});
    require(small_first.size() == 2, "expected two outputs");
    require(
        max_difference(small_first[0], reference_small[0]) <= 1e-4f,
        "first small sample changed"
    );
    require(
        max_difference(small_first[1], reference_another[0]) <= 1e-4f,
        "second small sample changed"
    );

    std::cout << "batch shape checks passed\n";
    return 0;
}
