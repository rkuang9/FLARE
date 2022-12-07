#include <iostream>
#include <chrono>
#include <orion/orion.hpp>
#include "examples/heart_attack_prediction.h"

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    HeartAttackPrediction();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() << "s";

    return 0;
}