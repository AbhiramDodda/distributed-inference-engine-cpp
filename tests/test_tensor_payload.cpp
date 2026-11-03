#include "tensor_payload.h"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    const std::vector<float> values{0.0f, 1.5f, -2.25f, 3.125f, -0.0f};
    const auto encoded = encode_float32_le(values);
    if (encoded.size() != values.size() * sizeof(float)) {
        throw std::runtime_error("encoded size mismatch");
    }
    const auto decoded = decode_float32_le(encoded);
    if (decoded.size() != values.size()) {
        throw std::runtime_error("decoded size mismatch");
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::fabs(decoded[i] - values[i]) > 1e-6f) {
            throw std::runtime_error("decoded value mismatch");
        }
    }
    bool rejected = false;
    try {
        decode_float32_le("bad");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("invalid payload was accepted");
    }
    return 0;
}
