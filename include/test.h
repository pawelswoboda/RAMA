#pragma once

#include <stdexcept>
#include <string>

struct test_exception
{
    test_exception(const std::string& s) : error_message(s) {}
    std::string error_message;
};

inline void test(const bool pred, const std::string& error_message = "")
{
    std::string msg;
    if(!error_message.empty()) {
        msg = "test failed: " + error_message;
    } else {
        msg = "test failed.";
    }

    if (!pred)
        throw std::runtime_error(msg);
}
