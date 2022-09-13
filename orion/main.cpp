#include <iostream>
#include <orion/orion.hpp>
#include <iomanip>
#include <fstream>


void compare_with_tensorflow()
{
    using namespace orion;

    int input_features = 100;
    int label_features = 1;
    int batch_size = 1;

    Tensor<2> w1(1, input_features);
    w1.setConstant(0.3);

    Tensor<2> w2(label_features, 1);
    w2.setConstant(0.6);

    std::vector<std::vector<Scalar>> training_set{
        std::vector<std::vector<Scalar>>(batch_size, std::vector<Scalar>(input_features, 0.5))
    };

    std::vector<std::vector<Scalar>> training_labels{
            std::vector<std::vector<Scalar>>(batch_size, std::vector<Scalar>(label_features, 0.6))
    };

    TensorMap<2> infer_input(training_set.front().data(), input_features, 1);
    TensorMap<2> infer_label(training_labels.front().data(), label_features, 1);

    Sequential model {
        new Dense<Sigmoid>(input_features, label_features, true),
        new Dense<Sigmoid>(1, 1, true),
        new Dense<Sigmoid>(1, label_features, true)
    };

    model[0].SetWeights(w1);
    model[1].SetWeights(w2);
    model[2].SetWeights(w2);

    BinaryCrossEntropy loss;
    MeanSquaredError mse;
    SGD opt;

    std::cout << "predict before fit: " << model.Predict(infer_input) << "\n";
    model.Compile(loss, opt);
    model.Fit(training_set, training_labels, 1000, 1);
    std::cout << "predict after fit: " << model.Predict(infer_input) << "\n";

    std::cout << "L1 output: " << model[0].GetOutput2D() << "\n";
    std::cout << "L1 dL/dZ: " << model[0].GetInputGradients2D() << "\n";
    std::cout << "L1 weights: " << model[0].GetWeights() << "\n";
    std::cout << "L1 weight gradients: " << model[0].GetWeightGradients() << "\n";
    std::cout << "L1 bias: " << model[0].GetBias()<< "\n";

    std::cout << "L2 output: " << model[1].GetOutput2D() << "\n";
    std::cout << "L2 dL/dZ: " << model[1].GetInputGradients2D() << "\n";
    std::cout << "L2 weights: " << model[1].GetWeights()<< "\n";
    std::cout << "L2 bias: " << model[1].GetBias()<< "\n";

    std::cout << "L3 output: " << model[2].GetOutput2D() << "\n";
    std::cout << "L3 dL/dZ: " << model[2].GetInputGradients2D() << "\n";
    std::cout << "L3 weights: " << model[2].GetWeights()<< "\n";
    std::cout << "L3 bias: " << model[2].GetBias()<< "\n";
    std::cout << "---------------------\n";
    std::cout << "gradient check:   " << model.GradientCheck(infer_input, infer_label, 1e-7) << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    compare_with_tensorflow();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}