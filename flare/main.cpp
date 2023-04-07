#include <iostream>
#include <chrono>
#include <flare/flare.hpp>
#include "examples/sarcasm_detection_nlp.hpp"

void Run()
{
    SarcasmDetection();
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    Run();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic