//
// Created by macross on 8/7/22.
//

#ifndef ORION_ORION_ASSERT_HPP
#define ORION_ORION_ASSERT_HPP

#ifndef ORION_NO_DEBUG

#include <iostream>

#define orion_assert(condition, message) \
    do { \
        if (!(condition)) { \
        std::cerr << "\nAssertion failed: " << #condition << ", file " << \
        __FILE__ ", " << __PRETTY_FUNCTION__ << ", line " \
        << __LINE__ << ",\n" << message << " "; \
        std::abort(); \
    } \
} while (false)
#else
#define orion_assert(condition, message)
#endif

#endif //ORION_ORION_ASSERT_HPP
