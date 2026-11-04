#include "request_validation.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using nlohmann::json;

void require_invalid(const json& value) {
    try {
        inference::validateInferRequest(value);
    } catch (const inference::ValidationError&) {
        return;
    }
    throw std::runtime_error("expected a validation error");
}

int main() {
    inference::validateInferRequest(json{{"request_id", "r"}, {"input_data", {1, 2, 3}}});
    require_invalid(json::object());
    require_invalid(json{{"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", 1}, {"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", ""}, {"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", std::string(257, 'x')}, {"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", "r"}});
    require_invalid(json{{"request_id", "r"}, {"input_data", {}}});
    require_invalid(json{{"request_id", "r"}, {"input_data", {"x"}}});
    require_invalid(json{{"request_id", "r"}, {"input_data", {1, nullptr}}});
    return 0;
}
