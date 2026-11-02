#ifndef TENSOR_PAYLOAD_H
#define TENSOR_PAYLOAD_H

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Raw float32 little-endian tensor payload. The HTTP layer validates the
// maximum size; this codec validates only that the body is a non-empty
// multiple of four bytes.
inline std::vector<float> decode_float32_le(const std::string& body) {
    if (body.empty() || body.size() % sizeof(float) != 0) {
        throw std::invalid_argument("tensor body must be a non-empty multiple of 4 bytes");
    }
    std::vector<float> values(body.size() / sizeof(float));
    const auto* bytes = reinterpret_cast<const unsigned char*>(body.data());
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t offset = i * 4;
        const uint32_t bits = static_cast<uint32_t>(bytes[offset]) |
                              (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                              (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                              (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        std::memcpy(&values[i], &bits, sizeof(bits));
    }
    return values;
}

inline std::string encode_float32_le(const std::vector<float>& values) {
    std::string body(values.size() * sizeof(float), '\0');
    auto* bytes = reinterpret_cast<unsigned char*>(body.data());
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof(bits));
        const size_t offset = i * 4;
        bytes[offset] = static_cast<unsigned char>(bits & 0xff);
        bytes[offset + 1] = static_cast<unsigned char>((bits >> 8) & 0xff);
        bytes[offset + 2] = static_cast<unsigned char>((bits >> 16) & 0xff);
        bytes[offset + 3] = static_cast<unsigned char>((bits >> 24) & 0xff);
    }
    return body;
}

#endif
