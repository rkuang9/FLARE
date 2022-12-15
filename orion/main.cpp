#include <iostream>
#include <chrono>
#include <orion/orion.hpp>
#include "examples/MNIST_digit_detection.h"


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    MNIST_CNN();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic