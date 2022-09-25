#include <iostream>
#include <orion/orion.hpp>
#include <iomanip>

using namespace orion;


void dim_test()
{
    std::cout << GlorotUniform().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";
    std::cout << GlorotNormal().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";

    std::cout << LecunUniform().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";
    std::cout << LecunNormal().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";

    std::cout << HeUniform().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";
    std::cout << HeNormal().Initialize(Tensor<3>::Dimensions(2, 3, 5), 1, 1) << "\n\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    dim_test();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}