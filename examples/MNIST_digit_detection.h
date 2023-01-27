//
// Created by Macross on 12/11/22.
//

#ifndef ORION_MNIST_DIGIT_DETECTION_H
#define ORION_MNIST_DIGIT_DETECTION_H

#include <orion/orion.hpp>

void MNIST_CNN()
{
    using namespace orion;
    namespace fs = std::filesystem;

    Dataset dataset(Dims<3>(28, 28, 3), Dims<1>(10));
    std::string path = "MNIST/archive/trainingSet/trainingSet/";

    for (int i = 0; i < 10; i++) {
        Tensor<1> label(10);
        label.setZero();
        label(i) = 1;

        for (const auto &entry: fs::directory_iterator(path + std::to_string(i))) {
            dataset.Add(entry.path(), label);
        }
    }

    dataset.Batch(16, true, false);

    for (auto &image: dataset.training_samples) {
        image = image / 255.0;
    }

    std::cout << "total number of batched samples: "
              << dataset.training_samples.size() << "\n";

    Sequential model {
            new Conv2D<ReLU>(32, Input(28, 28, 3), Kernel(3, 3),
                             Padding::PADDING_VALID),
            // 16, 26, 26, 32
            new MaxPooling2D(PoolSize(2, 2)),
            // 16, 13, 13, 32
            new Flatten<4>(),
            new Dense<Softmax>(5408, 10, false),
    };

    CategoricalCrossEntropy<2> loss;
    Adam opt;

    model.Fit(dataset.training_samples, dataset.training_labels, 15, loss, opt);

    std::cout << model.Predict<2>(dataset.training_samples.front()) << "\nexpect\n"
              << dataset.training_labels.front() << "\n";
    std::cout << model.Predict<2>(dataset.training_samples.back()) << "\nexpect\n"
              << dataset.training_labels.back() << "\n\n";
}

#endif //ORION_MNIST_DIGIT_DETECTION_H
