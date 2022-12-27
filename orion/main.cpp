#include <iostream>
#include <chrono>
#include <orion/orion.hpp>
#include "examples/MNIST_digit_detection.h"

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void conv2d()
{
    using namespace orion;

    int filters = 2;
    int batch_size = 2;
    int channels = 2;

    Tensor<2> _img_input(3, 3);
    _img_input.setValues({{1, 1, 1},
                          {2, 2, 2},
                          {3, 3, 3}});
    Tensor<4> img_input = _img_input.reshape(Dims<4>(1, 3, 3, 1))
            .broadcast(Dims<4>(batch_size, 1, 1, channels));

    Tensor<2> _kernels(2, 2);
    _kernels.setValues({{1, 0},
                        {2, 3}});
    Tensor<4> kernels = _kernels.reshape(Dims<4>(1, 2, 2, 1))
            .broadcast(Dims<4>(filters, 1, 1, channels));

    Tensor<4> label(batch_size, 13, 13, filters);
    label.setZero();

    Sequential model {
            new Conv2DTranspose<Linear>(
                    filters, Input(3, 3, channels), Kernel(2, 2),
                    Stride(1, 1), Dilation(3, 3), Padding::PADDING_VALID),
            new Conv2DTranspose<Linear>(
                    filters, Input(3, 3, filters), Kernel(2, 2),
                    Stride(2, 2), Dilation(1, 1), Padding::PADDING_VALID),
            new Conv2DTranspose<Linear>(
                    filters, Input(3, 3, filters), Kernel(2, 2),
                    Stride(1, 1), Dilation(2, 2), Padding::PADDING_VALID),
            new Conv2D<Linear>(filters, Input(3, 3, filters), Kernel(2, 2),
                               Stride(1, 1), Dilation(1, 1), Padding::PADDING_VALID),
    };

    for (Layer *layer: model.layers) {
        layer->SetWeights(kernels);
    }

    MeanSquaredError mse;
    SGD opt(1.0);

    model.Compile(mse, opt);
    model.Fit(std::vector<Tensor<4>> {img_input}, std::vector<Tensor<4>> {label}, 1);

    for (Layer *layer: model.layers) {
        std::cout << layer->name << "\n" << layer->GetWeights4D().shuffle(NCHW)
                  << "\n\n";
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    MNIST_CNN();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic