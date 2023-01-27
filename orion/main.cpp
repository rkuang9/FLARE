#include <iostream>
#include <chrono>
#include <orion/orion.hpp>
#include <examples/MNIST_digit_detection.h>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


using namespace orion;


/*
Sequential Generator()
{
    Tensor<2> noise = RandomUniform(Dims<2>(1, 100), -1, 1);

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

    return model;
}


Sequential Discriminator()
{
    Sequential model {
            new Conv2D<Linear>(64, Input(28, 28, 1), Kernel(5, 5),
                               Stride(2, 2), Dilation(1, 1),
                               Padding::PADDING_SAME),
            new LeakyReLU<4>(),
            new Dropout<4>(0.3),

            new Conv2D<Linear>(128, Input(14, 14, 64), Kernel(5, 5),
                               Stride(2, 2), Dilation(1, 1),
                               Padding::PADDING_SAME),
            new LeakyReLU<4>(),
            new Dropout<4>(0.3),

            new Flatten<4>(),
            new Dense<Sigmoid>(7 * 7 * 128, 1, false),
    };

    return model;
}*/


void GAN()
{
    using namespace orion;

    MNIST_CNN();
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