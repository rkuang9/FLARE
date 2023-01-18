#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void GAN()
{
    using namespace orion;

    Tensor<4> image(2, 3, 3, 1);
    image.setRandom();

    Tensor<2> labels(2, 9);
    labels.setZero();

    MeanSquaredError loss;

    // flatten an image into a rank 2 tensor
    Reshape<4, 2> reshape({2, -1});
    reshape.Forward(image);
    loss.CalculateLoss(reshape.GetOutput2D(), labels);
    reshape.Backward(loss);

    assert(reshape.GetInputGradients4D().dimensions() == image.dimensions());
    std::cout << "reshape layer output: " << reshape.GetOutput2D().dimensions() <<
              "\n" << reshape.GetOutput2D() << "\n";
    std::cout << "reshape layer gradient dimensions: "
              << reshape.GetInputGradients4D().dimensions() << "\n\n";

    // the above reshaping is equivalent to flattening
    Flatten<4> flatten;
    flatten.Forward(image);
    loss.CalculateLoss(flatten.GetOutput2D(), labels);
    flatten.Backward(loss);

    assert(flatten.GetInputGradients4D().dimensions() == image.dimensions());
    std::cout << "flatten layer output: " << flatten.GetOutput2D().dimensions() <<
              "\n" << flatten.GetOutput2D() << "\n";
    std::cout << "flatten layer gradient dimensions: "
              << flatten.GetInputGradients4D().dimensions() << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    GAN();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic