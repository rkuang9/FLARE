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
    int iterations = 100;

    // create a sample image (batch_size, 5, 5, channels)
    Tensor<2> preimage(5, 5);
    preimage.setValues({
                               {1,    2,    3,    4,    5},
                               {2,    4,    6,    8,    10},
                               {3,    5,    6,    1,    2},
                               {2,    1,    0.3,  4,    0.01},
                               {0.03, 0.01, 0.31, 0.19, 0.26}});

    Tensor<4> image = preimage
            .reshape(Dims<4>(1, preimage.dimension(0), preimage.dimension(1), 1))
            .broadcast(Dims<4>(batch_size, 1, 1, channels));
    std::cout << "image dims: " << image.dimensions() << "\n";

    // create a sample label
    Tensor<2> label(9 * filters, batch_size);
    label.setConstant(30.0);
    std::cout << "label dims: " << label.dimensions() << "\n";

    // create sample kernels
    Tensor<2> prekernel(3, 3);
    prekernel.setValues({{0.1, 0,   0.8},
                         {0.6, 1,   0.3},
                         {0.3, 0.1, 0.12}});

    Tensor<4> kernel = prekernel
            .reshape(Dims<4>(1, prekernel.dimension(0), prekernel.dimension(1), 1))
            .broadcast(Dims<4>(filters, 1, 1, channels));

    Tensor<4> kernel2 = prekernel
            .reshape(Dims<4>(1, prekernel.dimension(0), prekernel.dimension(1), 1))
            .broadcast(Dims<4>(filters, 1, 1, filters));

    std::cout << "kernel dims: " << kernel.dimensions() << "\n";
    std::cout << "kernel dims2: " << kernel2.dimensions() << "\n";

    //////////////////////// begin actual api ///////////////////////////////

    std::vector<Tensor<4>> training_samples {image};
    std::vector<Tensor<2>> training_labels {label};

    Sequential model {
            new Conv2D<TanH>(filters, Input(5, 5, channels), Kernel(3, 3),
                             Stride(1, 1), Dilation(2, 2), Padding::PADDING_SAME),
            new Conv2D<Sigmoid>(filters, Input(5, 5, filters), Kernel(3, 3),
                                Stride(1, 1), Dilation(1, 1),
                                Padding::PADDING_VALID),
            new Flatten<4>(),
    };

    model.layers[0]->SetWeights(kernel);
    model.layers[1]->SetWeights(kernel2);

    SGD sgd(1);
    MeanSquaredError loss;

    model.Compile(loss, sgd);
    model.Fit(training_samples, training_labels, iterations, 1);

    std::cout << "updated " << model.layers[0]->name << " kernels: "
              << model.layers[0]->GetWeights4D().dimensions() << "\n"
              << model.layers[0]->GetWeights4D().shuffle(NCHW) << "\n";
}


void maxpool_op()
{

    Tensor<2> preimg(3, 3);
    preimg.setValues({
                             {1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9}});

    Tensor<4> eeemg = preimg.reshape(Dims<4>(1, 3, 3, 1));
    Tensor<5> p = eeemg.extract_image_patches(2, 2, 1, 1, 1, 1, Eigen::PADDING_SAME);

    std::cout << p.shuffle(Dims<5>(0, 1, 4, 2, 3));


    //https://github.com/madlib/eigen/blob/master/unsupported/test/cxx11_tensor_argmax.cpp
    //Eigen::Tensor<Eigen::DenseIndex, 1, Eigen::RowMajor> tensor_argmax(1);
    //Tensor<1, Eigen::DenseIndex> omfg = preimage.argmax(0);


    return;
    Tensor<2> a(4, 3);
    a.setValues({
                        {0,   100,  200},
                        {300, 400,  300},
                        {600, 700,  1100},
                        {900, 1000, 1100}});

    Tensor<0> maxval = a.maximum();
    std::cout << "maxval: " << maxval(0) << "\n\n";

    Tensor<4> img = a.reshape(Dims<4>(1, 4, 3, 1))
            .broadcast(Dims<4>(1, 1, 1, 1));


    Stride stride(1, 1);

    Tensor<5> patches = img.extract_image_patches(
            2, 2,
            stride[0], stride[1],
            1, 1, Eigen::PADDING_VALID);

    std::cout << "patches: " << patches.dimensions() << "\n"
              << patches.shuffle(Dims<5>(0, 1, 4, 2, 3)) << "\n";


    Tensor<4> maxtensor = patches
            .maximum(Dims<2>(2, 3))
            .reshape(Dims<4>(1, 3, 2, 1));
    std::cout << "maxtensor: " << maxtensor.dimensions() << "\n"
              << maxtensor.shuffle(Dims<4>(0, 3, 1, 2));
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