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


void maxpool()
{
    int batches = 2;
    int channels = 3;
    int filters = 2;

    // hard code a small image
    Tensor<2> _fakeimg(3, 3);
    _fakeimg.setValues({{0.321, 0.542,  0.876},
                        {0.056, 0.0312, 0.432},
                        {0.432, 0.654,  0.192}});
    Tensor<4> fakeimg = _fakeimg
            .reshape(Dims<4>(1, 3, 3, 1))
            .broadcast(Dims<4>(batches, 1, 1, channels));

    // hard code the labels
    Tensor<4> fakelabels(batches, 2, 2, filters);
    fakelabels.setZero();

    std::vector<Tensor<4>> training_samples {fakeimg};
    std::vector<Tensor<4>> training_labels {fakelabels};

    Sequential model {
            new Conv2D<TanH>(filters, Input {3, 3, channels}, Kernel {2, 2},
                             Stride {1, 1}, Dilation {1, 1},
                             Padding::PADDING_VALID),
            new MaxPooling2D(PoolSize(3, 3), Stride(1, 1), Padding::PADDING_SAME),
    };

    MeanSquaredError loss;
    SGD opt(1);

    // hard code the kernel values
    Tensor<4> kernel(filters, 2, 2, channels);
    kernel.setConstant(1);
    model.layers[0]->SetWeights(kernel);

    model.Compile(loss, opt);
    model.Fit(training_samples, training_labels, 100, 1);

    std::cout << "loss: " << loss.GetLoss() << "\n";

    std::cout << "updated kernels:\n"
              << model.layers[0]->GetWeights4D().shuffle(Dims<4>(3, 0, 1, 2))
              << "\n";

    std::cout << "network output: \n"
              << model.Predict(fakeimg).shuffle(Dims<4>(3, 0, 1, 2)) << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    maxpool();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}