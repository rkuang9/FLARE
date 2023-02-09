#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


using namespace orion;


void serialize()
{
    Layer *conv = new Conv2D<Linear>(2, 3, Kernel(4, 4), Stride(1, 1),
                                     Dilation(1, 1), Padding::PADDING_SAME);
    
    Tensor<1> flattened_kernels = conv->GetWeights4D()
            .reshape(Dims<1>(conv->GetWeights4D().size()));

    conv->Save(conv->name + ".weights");
    conv->Load(conv->name + ".weights");

    Tensor<1> loaded_flattened_kernels = conv->GetWeights4D()
            .reshape(Dims<1>(conv->GetWeights4D().size()));

    for (int i = 0; i < flattened_kernels.size(); i++) {
        orion_assert(
                std::abs(flattened_kernels(i) - loaded_flattened_kernels(i)) < 1e-7,
                "SAVED WEIGHTS NOT EQUAL");
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    serialize();

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