#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
//#include "examples/xor_classifier.hpp"
#include <utility>

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;

Dims<4> NCHW(0, 3, 1, 2);


void conv_backpropagation()
{
    int filters = 2;
    int channels = 3;
    int batch_size = 5;

    std::cout << "filters: ";
    std::cin >> filters;

    std::cout << "channels: ";
    std::cin >> channels;

    std::cout << "batch_size: ";
    std::cin >> batch_size;
    std::cout << "\n";

    Tensor<2> preimage(5, 5);
    preimage.setValues({{1, 2,  3,  4,  5},
                        {2, 4,  6,  8,  10},
                        {3, 6,  9,  12, 15},
                        {5, 10, 15, 20, 25},
                        {6, 12, 18, 24, 36}});

    Tensor<4> image = preimage
            .reshape(Dims<4>(1, preimage.dimension(0), preimage.dimension(1), 1))
            .broadcast(Dims<4>(batch_size, 1, 1, channels));

    Tensor<1> prelabel(9 * filters * batch_size);
    prelabel.setConstant(30.0);
    Tensor<4> label(batch_size, 3, 3, filters);
    label.setConstant(30.0);

    Tensor<2> prekernel(3, 3);
    prekernel.setValues({{1, 0, 1},
                         {2, 1, 3},
                         {3, 1, 2}});

    Tensor<4> kernel = prekernel
            .reshape(Dims<4>(1, prekernel.dimension(0), prekernel.dimension(1), 1))
            .broadcast(Dims<4>(filters, 1, 1, channels));

    SGD sgd(1);
    MeanSquaredError loss;

    Layer *conv = new Conv2D<Swish>(filters, Input(5, 5, channels), Kernel(3, 3),
                                     Padding::PADDING_VALID, Stride(1, 1));
    conv->SetWeights(kernel);

    conv->Forward(image);
    loss.CalculateLoss(conv->GetOutput4D(), label);
    conv->Backward(loss);
    conv->Update(sgd);
    std::cout << "updated kernels: " << conv->GetWeights4D().dimensions() << "\n" << conv->GetWeights4D().shuffle(NCHW) << "\n";
}



int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    //padding();
    conv_backpropagation();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}