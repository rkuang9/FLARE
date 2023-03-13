//
// Created by macross on 8/7/22.
//

#ifndef FLARE_FLARE_ASSERT_HPP
#define FLARE_FLARE_ASSERT_HPP

#ifndef FLARE_NO_DEBUG

#include <iostream>

#define fl_assert(condition, message) \
    do { \
        if (!(condition)) { \
        std::cerr << "\nAssertion failed: " << #condition << ", file " << \
        __FILE__ ", " << __PRETTY_FUNCTION__ << ", line " \
        << __LINE__ << ",\n" << message << " "; \
        std::abort(); \
    } \
} while (false)
#else
#define fl_assert(condition, message)
#endif

#endif //FLARE_FLARE_ASSERT_HPP
