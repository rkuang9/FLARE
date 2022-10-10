#include <iostream>
#include <chrono>

#include <orion/orion.hpp>
#include "examples/xor_classifier.hpp"

using namespace orion;


void convolve_test()
{
    int batch = 2;
    int channels = 3;
    int filters = 3;
    int filter_size = 2;

    Tensor<3> preimage(1, 3, 3);
    preimage.setValues({{{1, 1, 3}, {1, 1, 2}, {2, 4, 0}}});
    Tensor<3> prekernel(1, 2, 2);
    prekernel.setValues({{{1, 0}, {2, 1}}});

    Tensor<4> image = preimage.reshape(Dims<4>(1, 3, 3, 1)).broadcast(
            Dims<4>(batch, 1, 1, channels));
    Tensor<4> kernel = prekernel.reshape(
            Dims<4>(1, filter_size, filter_size, 1)).broadcast(
            Dims<4>(filters, 1, 1, channels));


    Layer *conv = new Conv2D<ReLU>(filters, Input(3, 3, channels), Kernel(2, 2), Padding::PADDING_VALID);
    conv->SetWeights(kernel);
    conv->Forward(image);
    std::cout << conv->GetOutput4D().shuffle(Dims<4>(0, 3, 1, 2)) << "\n" << conv->GetOutput4D().dimensions() << "\n";
}


void pooling2D()
{
    Tensor<2> img(3, 3);
    img.setValues({{1, 2, 3},
                   {4, 5, 6},
                   {7, 8, 9}});

    Tensor<4> patches = img.reshape(Dims<4>(1, 3, 3, 1))
            .extract_image_patches(2, 2, 1, 1, 1, 1, Eigen::PADDING_VALID)
            .reshape(Dims<3>(1, 4, 4))
            .maximum(Dims<1>(2))
            .reshape(Dims<4>(1, 2, 2, 1));

    std::cout << patches.shuffle(Dims<4>(0, 3, 1, 2)) << "\n" << patches.dimensions()
            << "\n\n";

}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();


    //pooling2D();
    convolve_test();



    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}