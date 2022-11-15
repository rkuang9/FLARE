#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
//#include "examples/xor_classifier.hpp"

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;


void maxpool_op()
{
    Tensor<2> quad(3, 3);
    quad.setValues({{1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}});
    Tensor<4> fakeimg = quad
            .reshape(Dims<4>(1, 3, 3, 1))
            .broadcast(Dims<4>(2, 1, 1, 3));
    Tensor<4> fakelabels(2, 2, 2, 1);
    fakelabels.setZero();


    Sequential model {
            new Conv2D<Linear>(1, Input {3, 3, 3}, Kernel {1, 1}, Stride {1, 1},
                               Dilation {1, 1}, Padding::PADDING_VALID),
            new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_VALID),
    };

    MeanSquaredError loss;
    SGD opt(1);

    Tensor<4> kernel(1, 1, 1, 3);
    kernel.setConstant(1);

    model.layers[0]->SetWeights(kernel);

    model.Compile(loss, opt);
    model.Fit(std::vector<Tensor<4>> {fakeimg}, std::vector<Tensor<4>> {fakelabels},
              1, 1);
    std::cout << "updated kernels: " << model.layers[0]->GetWeights4D() << ", expect 3 x [-308]\n";
    std::cout << "loss: " << loss.GetLoss() << ", expect 463.5000\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    maxpool_op();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}