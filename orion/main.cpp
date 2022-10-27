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
    int batch_size = 4;

    int iterations = 1;

    std::cout << "filters: ";
    std::cin >> filters;

    std::cout << "channels: ";
    std::cin >> channels;

    std::cout << "batch_size: ";
    std::cin >> batch_size;

    std::cout << "times to run: ";
    std::cin >> iterations;
    std::cout << "\n";

    Tensor<2> preimage(5, 5);
    preimage.setValues({{1,    2,    3,    4,    5},
                        {2,    4,    6,    8,    10},
                        {3,    5,    6,    1,    2},
                        {2,    1,    0.3,  4,    0.01},
                        {0.03, 0.01, 0.31, 0.19, 0.26}});

    Tensor<4> image = preimage
            .reshape(Dims<4>(1, preimage.dimension(0), preimage.dimension(1), 1))
            .broadcast(Dims<4>(batch_size, 1, 1, channels));

    Tensor<1> prelabel(9 * filters * batch_size);
    prelabel.setConstant(30.0);

    Tensor<4> label(batch_size, 5, 5, filters);
    label.setConstant(30.0);

    Tensor<2> prekernel(3, 3);
    prekernel.setValues({{0.1, 0,   0.8},
                         {0.6, 1,   0.3},
                         {0.3, 0.1, 0.12}});

    Tensor<4> kernel = prekernel
            .reshape(Dims<4>(1, prekernel.dimension(0), prekernel.dimension(1), 1))
            .broadcast(Dims<4>(filters, 1, 1, channels));

    ///////////////////////////////////////////////////////////////

    std::vector<Tensor<4>> training_samples {image};
    std::vector<Tensor<4>> training_labels {label};

    Sequential model {
            new Conv2D<Swish>(filters, Input(5, 5, channels), Kernel(3, 3),
                              Padding::PADDING_SAME, Stride(1, 1)),
    };

    model.layers[0]->SetWeights(kernel);

    SGD sgd(1);
    MeanSquaredError loss;

    model.Compile(loss, sgd);
    model.Fit(training_samples, training_labels, iterations, 1);

    std::cout << "updated kernels: " << model.layers[0]->GetWeights4D().dimensions() << "\n"
              << model.layers[0]->GetWeights4D().shuffle(NCHW) << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    conv_backpropagation();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}