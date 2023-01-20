#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void test(const orion::Sigmoid &sigmoid)
{

}


void Generator()
{
    using namespace orion;

    Tensor<2> noise(2, 3);
    noise.setValues({{0.5, -1, 0.7}, {-1, -0.5, -3}});

    Tensor<2> label = noise.constant(0.0);
    std::cout << "noise\n" << noise << "\n";

    MeanSquaredError loss;


    LeakyReLU<2> leaky;
    leaky.Forward(noise);
    loss.CalculateLoss(leaky.GetOutput2D(), label);
    leaky.Backward(loss);

    std::cout << "output\n" << leaky.GetOutput2D() << "\n";
    std::cout << "input gradients\n" << leaky.GetInputGradients2D() << "\n";
    return;

    Sequential model {
            new Dense<Linear>(100, 7 * 7 * 256, false),
            new BatchNormalization<2, 1>(Dims<1>(1)),
            new Activation<ReLU, 2>,

            new Reshape<2, 4>({1, 7, 7, 256}),

            new Conv2DTranspose<Linear>(128, Input(7, 7, 256), Kernel(5, 5),
                                        Stride(1, 1), Dilation(1, 1),
                                        Padding::PADDING_SAME),
            new BatchNormalization<4, 1>(Dims<1>(3)),
            new Activation<ReLU, 4>,

            new Conv2DTranspose<Linear>(64, Input(7, 7, 128), Kernel(5, 5),
                                        Stride(2, 2), Dilation(1, 1),
                                        Padding::PADDING_SAME),
            new BatchNormalization<4, 1>(Dims<1>(3)),
            new Activation<ReLU, 4>,

            new Conv2DTranspose<TanH>(1, Input(14, 14, 64), Kernel(5, 5),
                                      Stride(2, 2), Dilation(1, 1),
                                      Padding::PADDING_SAME),
    };

    std::cout << "noise\n" << noise << "\n";

    model.Predict<4>(noise);
}


void GAN()
{
    Generator();
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