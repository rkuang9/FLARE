#include <iostream>
#include <chrono>
#include <vector>


#include <orion/orion.hpp>


using namespace orion;




int main()
{
    auto start = std::chrono::high_resolution_clock::now();



    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}