#include <iostream>
#include <chrono>
#include <vector>


#include <orion/orion.hpp>


using namespace orion;


void convolution_dev()
{
    Tensor<4> img(2, 2, 3, 5);
    img.setRandom();
    std::cout << img << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();


    convolution_dev();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}