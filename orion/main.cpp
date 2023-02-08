#include <iostream>
#include <chrono>
#include <orion/orion.hpp>
#include "examples/CIFAR_image_detection.h"

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


using namespace orion;


BinaryCrossEntropy<2> DiscriminatorLoss(const Tensor<2> &real_output,
                                        const Tensor<2> &fake_output)
{
    BinaryCrossEntropy<2> real_loss = BinaryCrossEntropy<2>(
            real_output, real_output.constant(1.0));
    BinaryCrossEntropy<2> fake_loss = BinaryCrossEntropy<2>(
            fake_output, fake_output.constant(0.0));

    return real_loss + fake_loss;
}


// From the example https://www.tensorflow.org/tutorials/generative/dcgan
// Our MNIST dataset's images have 3 channels, mean() will combine it into 1 channel
// This GAN currently performs 3x slower than TensorFlow, specifically during convolutions
// Areas for improvement are:
// - optimize the 3 convolution operations or swap them for TensorFlow's kernels
// - add multi-threading to activation, loss, and optimizer
// - parallelize layer update
// - add compile time check to skip activation for linear layers
void GAN()
{
    using namespace orion;

    Sequential generator {
            new Dense<Linear>(100, 7 * 7 * 256, false),
            new BatchNormalization<2, 1>(Dims<1>(1)),
            new LeakyReLU<2>(),

            new Reshape<2, 4>({-1, 7, 7, 256}),

            new Conv2DTranspose<Linear, 2>(128, 256, Kernel(5, 5),
                                           Stride(1, 1), Dilation(1, 1),
                                           Padding::PADDING_SAME),
            new BatchNormalization<4, 1>(Dims<1>(3)),
            new LeakyReLU<4>(),

            new Conv2DTranspose<Linear, 2>(64, 128, Kernel(5, 5),
                                           Stride(2, 2), Dilation(1, 1),
                                           Padding::PADDING_SAME),
            new BatchNormalization<4, 1>(Dims<1>(3)),
            new LeakyReLU<4>(),

            new Conv2DTranspose<TanH, 2>(1, 64, Kernel(5, 5),
                                         Stride(2, 2), Dilation(1, 1),
                                         Padding::PADDING_SAME),
    };

    Sequential discriminator {
            new Conv2D<Linear, 4>(64, 1, Kernel(5, 5),
                                  Stride(2, 2), Dilation(1, 1),
                                  Padding::PADDING_SAME),
            new LeakyReLU<4>(),
            new Dropout<4>(0.3),

            new Conv2D<Linear, 4>(128, 64, Kernel(5, 5),
                                  Stride(2, 2), Dilation(1, 1),
                                  Padding::PADDING_SAME),
            new LeakyReLU<4>(),
            new Dropout<4>(0.3),

            new Flatten<4>(),
            new Dense<Sigmoid>(7 * 7 * 128, 1, false),
    };

    Adam discriminator_opt(1e-4);
    Adam generator_opt(1e-4);

    // load the dataset MNIST images
    Dataset dataset(Dims<3>(28, 28, 3), Dims<1>(1));

    int epochs = 50;
    int batch_size = 256;
    int noise_dim = 100;

    for (int i = 0; i < 10; i++) {
        // label arg is unused, discriminator loss sets it to 1 since they are all real
        for (const auto &entry: std::filesystem::directory_iterator(
                "MNIST/archive/trainingSet/trainingSet/" + std::to_string(i))) {
            dataset.Add(entry.path(), Tensor<1>(Dims<1>(1)).setConstant(1.0));
        }
    }

    dataset.Batch(batch_size, false, false);
    std::cout << "dataset batched, total mini-batches: "
              << dataset.training_samples.size() << "\n";

    // normalize images to [-1, 1], combine channels since they're all black/white
    for (auto &image: dataset.training_samples) {
        image = (image - 127.5) / 127.5;
        image = image.mean(Dims<1>(3))
                .reshape(Dims<4>(image.dimensions().front(), 28, 28, 1));
    }

    BinaryCrossEntropy<2> disc_gen_loss;

    // training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < dataset.training_samples.size(); i++) {
            Tensor<2> noise_discriminator = RandomNormal(
                    Dims<2>(dataset.training_samples[i].dimensions().front(),
                            noise_dim), 0, 1);
            const Tensor<4> &fake_image = generator.Predict<4>(noise_discriminator,
                                                               true);

            // train the discriminator first
            const Tensor<2> &mnist_output = discriminator.Predict<2>(
                    dataset.training_samples[i], true);
            const Tensor<2> &fake_output = discriminator.Predict<2>(fake_image,
                                                                    true);

            BinaryCrossEntropy<2> discriminator_loss = DiscriminatorLoss(
                    mnist_output, fake_output);
            discriminator.Backward(discriminator_loss.GetGradients());
            discriminator.Update(discriminator_opt);

            Tensor<2> noise_generator = RandomNormal(
                    Dims<2>(dataset.training_samples[i].dimensions().front(),
                            noise_dim), 0, 1);

            const Tensor<4> &false_image = generator.Predict<4>(noise_generator,
                                                                true);
            const Tensor<2> &is_fake = discriminator.Predict<2>(false_image, true);

            disc_gen_loss(is_fake, is_fake.constant(1.0));
            discriminator.Backward(disc_gen_loss.GetGradients());

            generator.Backward(discriminator.layers.front()->GetInputGradients4D());

            generator.Update(generator_opt);
            // not sure discriminator is updated here since it the paper's algorithm does say so
            //discriminator.Update(discriminator_opt);

            auto stop = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    stop - start);

            std::cout << "\rEpoch " << epoch << ", mini-batch " << i << "/"
                      << dataset.training_samples.size() << ", "
                      << time.count() / 1000.0 << " s";
        }

        std::cout << "\n";
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    GAN();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic