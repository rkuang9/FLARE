#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void GAN()
{
    using namespace orion;

    Tensor<2> image(2, 2);
    image.setValues({{1, 2}, {4, 5}});

    Tensor<2> labels = image.constant(0.0);

    MeanSquaredError loss;
    std::cout << "input\n" << image << "\n";

    Activation<Sigmoid, 2> transfer;
    transfer.Forward(image);
    std::cout << "sigmoid\n" << transfer.GetOutput2D() << "\n";

    loss.CalculateLoss(transfer.GetOutput2D(),  labels);

    transfer.Backward(loss);
    std::cout << "input grads\n" << transfer.GetInputGradients2D() << "\n";
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