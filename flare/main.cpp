#include <iostream>
#include <chrono>
#include <flare/flare.hpp>


void test()
{
    std::vector<std::string> texts {
            "I lOVe my dog",
            "I love my cat",
            "You love my dog!",
            "Do you think my dog is amazing?"
    };

    fl::Tokenizer dict;
    dict.Fit(texts);
    dict.Compile();
    dict.Print();
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    test();

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