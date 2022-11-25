#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;


void mnist()
{
    namespace fs = std::filesystem;

    Dims<3> sample_dims(28, 28, 3);
    Dims<1> label_dims(1);

    Dataset dataset(sample_dims, label_dims);

    std::string mnist_path =
            fs::current_path().string() + "/mnist/trainingSet/trainingSet/";


    Tensor<1> img_label(1);

    // go through folders 0-9, labels will be the folder number
    for (int i = 0; i < 10; i++) {
        std::string img_path = mnist_path + std::to_string(i);
        img_label.setValues({i == 1 ? Scalar(1) : Scalar(0)}); // 1 digit is true, not 1 digit is false

        // give Dataset::Add the image path and it will create a tensor
        for (auto &file: std::filesystem::directory_iterator(img_path)) {
            dataset.Add(file.path(), img_label);
        }

    }

    dataset.Batch(1, true);

    std::cout << "total samples: " << dataset.training_samples.size() << "\n";

    // haven't implemented softmax activation so sigmoid (true/false is one) will have to do
    Sequential model {
            new Conv2D<ReLU>(16, Input(28, 28, 3),
                             Kernel(5, 5), Padding::PADDING_VALID),
            new MaxPooling2D(PoolSize(3, 3)),
            new Conv2D<ReLU>(32, Input(22, 22, 16),
                             Kernel(3, 3), Padding::PADDING_VALID),
            new Flatten<4>(),
            new Dense<ReLU>(1152, 32, false),
            new Dense<Sigmoid>(32, 1, false),
    };

    Adam opt;
    MeanSquaredError loss;

    model.Compile(loss, opt, {new Loss});
    model.Fit(dataset.training_samples, dataset.training_labels, 15, 1);

    // prediction on a test sample
    Tensor<3> image_tensor;
    cv::Mat cv_matrix = cv::imread("/Users/macross/Desktop/OrionNN/bin/mnist/testSample/img_62.jpg");
    cv::cv2eigen(cv_matrix, image_tensor);
    Tensor<4> cv_image = image_tensor.reshape(Dims<4>(1, 28, 28, 3));


    std::cout << model.Predict<2>(cv_image) << ", expected 1";
}


/*void debug()
{
    Tensor<2> input(3, 3);
    input.setValues({{1, 2, 1},
                     {3, 2, 1},
                     {3, 4, 4}});
    Tensor<4> image = input.reshape(Dims<4>(1, 3, 3, 1));
    Tensor<2> label(1, 1);
    label.setValues({{0.5}});
    Tensor<1> pweights(4);
    pweights.setValues(
            {0.3, 0.8, 0.5, 0.1});
    Tensor<2> weights = pweights.reshape(Dims<2>(1, 4));
    Tensor<2> pkernels(2, 2);
    pkernels.setValues({{0.3, 0.2},
                        {0.5, 0.1}});
    Tensor<4> kernels = pkernels.reshape(Dims<4>(1, 2, 2, 1)).broadcast(
            Dims<4>(1, 1, 1, 1));
    std::cout << "hard code setup complete\n";
    Sequential tf {
            new Conv2D<ReLU>(1, Input(3, 3, 1), Kernel(2, 2),
                             Padding::PADDING_VALID),
            new Flatten<4>(),
            new Dense<Sigmoid>(4, 1, false),
    };

    MeanSquaredError mse;
    SGD sgd(1);

    tf.layers[0]->SetWeights(kernels);
    tf.layers[2]->SetWeights(weights);

    std::cout << "predict: " << tf.Predict<2>(image) << "\n";
    tf.Compile(mse, sgd);
    tf.Fit(std::vector<Tensor<4>> {image}, std::vector<Tensor<2>> {label}, 1, 1);
    std::cout << tf.layers[0]->GetWeights4D().shuffle(Dims<4>(3, 0, 1, 2)) << "\n";
    std::cout << tf.layers[2]->GetWeights() << "\n";
    std::cout << mse.GetGradients2D() << "\n";
}*/


int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    //reorder_dense_batch_dim();
    mnist();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() << "s";

    return 0;
}