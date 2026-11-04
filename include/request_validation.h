#ifndef REQUEST_VALIDATION_H
#define REQUEST_VALIDATION_H

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace inference {

class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& message) : std::runtime_error(message) {}
};

inline void validateInferRequest(const nlohmann::json& request) {
    if (!request.is_object()) {
        throw ValidationError("request must be a JSON object");
    }
    const auto request_id = request.find("request_id");
    if (request_id == request.end() || !request_id->is_string()) {
        throw ValidationError("request_id must be a string");
    }
    const std::string id = request_id->get<std::string>();
    if (id.empty() || id.size() > 256) {
        throw ValidationError("request_id must be 1-256 characters");
    }
    const auto input_data = request.find("input_data");
    if (input_data == request.end() || !input_data->is_array()) {
        throw ValidationError("input_data must be an array");
    }
    if (input_data->empty() || input_data->size() > 1'000'000) {
        throw ValidationError("input_data must contain 1-1000000 numbers");
    }
    for (const auto& value : *input_data) {
        if (!value.is_number()) {
            throw ValidationError("input_data must contain only numbers");
        }
        try {
            if (!std::isfinite(value.get<double>())) {
                throw ValidationError("input_data must contain only finite numbers");
            }
        } catch (const nlohmann::json::exception&) {
            throw ValidationError("input_data must contain only finite numbers");
        }
    }
}

}  // namespace inference

#endif
