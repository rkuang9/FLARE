//
// Created by Macross on 12/12/22.
//

#ifndef ORION_CIFAR_IMAGE_DETECTION_H
#define ORION_CIFAR_IMAGE_DETECTION_H

#include <orion/orion.hpp>

// https://github.com/YoongiKim/CIFAR-10-images
// if using OpenCV to display images from a Tensor, unexpected results will occur, see
// https://stackoverflow.com/questions/54971083/how-to-use-cvmat-and-eigenmatrix-correctly-opencv-eigen
// this is just a working example and not tuned for accuracy!!!
void CIFAR10_CNN()
{
    using namespace orion;
    namespace fs = std::filesystem;

    Dataset cifar(Dims<3>(32, 32, 3), Dims<1>(10));
    std::string path = "CIFAR-10-images/train/";

    std::vector<std::string> folders = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"};

    for (int i = 0; i < folders.size(); i++) {
        Tensor<1> label(10);
        label.setZero();
        label(i) = 1;
        std::cout << "\rAdding to dataset CIFAR class " << i + 1 << "/" << folders.size();

        for (const auto &entry: fs::directory_iterator(path + folders[i])) {
            cifar.Add(entry.path(), label);
        }
    }

    cifar.Batch(16, true, false);

    for (auto &image: cifar.training_samples) {
        image = image / 255.0;
    }

    std::cout << "\nTotal number of training samples: " << cifar.training_samples.size() << "\n";

    Sequential model {
            new Conv2D<ReLU>(32, 3, Kernel(3, 3),
                             Padding::PADDING_VALID),
            new MaxPooling2D(PoolSize(2, 2)),
            new Flatten<4>(),
            new Dense<Softmax>(7200, 10, false),
    };

    CategoricalCrossEntropy<2> loss;
    SGD opt(0.001, 0.9);

    model.Fit(cifar.training_samples, cifar.training_labels, 10, loss, opt);
}

#endif //ORION_CIFAR_IMAGE_DETECTION_H
