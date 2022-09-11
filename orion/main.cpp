#include <iostream>
#include <orion/orion.hpp>
#include <iomanip>
#include <fstream>


void linear_sigmoid_test()
{
    using namespace orion;

    Tensor<2> w1(1, 2);
    w1.setConstant(0.3);

    Tensor<2> w2(1, 1);
    w2.setConstant(0.6);

    std::vector<std::vector<Scalar>> training_set{{0.5, 0.5}};
    std::vector<std::vector<Scalar>> training_labels{{0.6}};

    Sequential model {
        new Dense<Sigmoid>(2, 1, false),
        new Dense<Sigmoid>(1, 1, false)
    };

    model[0].SetWeights(w1);
    model[1].SetWeights(w2);

    BinaryCrossEntropy loss;
    SGD opt;

    model.Compile(loss, opt);
    model.Fit(training_set, training_labels, 20, 1);

    std::cout << "L1 output: " << model[0].GetOutput2D() << "\n";
    std::cout << "L1 dL/dZ: " << model[0].GetInputGradients2D() << "\n";
    std::cout << "L1 weights: " << model[0].GetWeights() << "\n";

    std::cout << "L2 output: " << model[1].GetOutput2D() << "\n";
    std::cout << "L2 dL/dZ: " << model[1].GetInputGradients2D() << "\n";
    std::cout << "L2 weights: " << model[1].GetWeights()<< "\n";

    TensorMap<2> infer_input(training_set.front().data(), 2, 1);
    TensorMap<2> infer_label(training_labels.front().data(), 1, 1);

    std::cout << "gradient check:   " << model.GradientCheck(infer_input, infer_label, 1e-7);
}


void learn_xor()
{
    using namespace orion;
    std::random_device random;
    std::mt19937_64 mt(random());

    std::vector<std::vector<Scalar>> training_set;
    std::vector<std::vector<Scalar>> labels;

    for (int i = 0; i < 1000; i++) {
        Scalar x = std::uniform_int_distribution<int>(0, 1)(mt);
        Scalar y = std::uniform_int_distribution<int>(0, 1)(mt);
        Scalar z = (x != y);

        training_set.push_back({x, y});
        labels.push_back({z});
    }

    Sequential model{
            new Dense<TanH>(2, 1, false),
            /*new Dense<ReLU>(2, 10, false),
            new Dense<ReLU>(10, 3, false),
            new Dense<ReLU>(3, 1, false),*/
    };

    BinaryCrossEntropy loss;
    SGD opt;


    model.Compile(loss, opt);
    model.Fit(training_set, labels, 1, 1);

    Tensor<2> infer(2, 1);
    infer.setValues({{2}, {0}});

    Tensor<2> infer_label(1, 1);
    infer_label.setValues({{1}});

    std::cout << "gradient check: " << model.GradientCheck(infer, infer_label, 1e-7);

    std::cout << "\n\n\n";

    Tensor<2> pred(1, 3);
    pred.setValues({{1, 1, 1}});

    Tensor<2> label(1, 3);
    label.setValues({{1, 1, 0}});

    return;
    for (int i = 0; i < 10; i++) {
        std::cout << training_set[i][0] << " xor " << training_set[i][1]
                << " = "
                << model.Predict(TensorMap<2>(training_set[i].data(), 2, 1))
                << "\n";
    }
}


void grad_check_test()
{
    using namespace orion;

    // generating addition dataset
    std::vector<std::vector<Scalar>> training_set;
    std::vector<std::vector<Scalar>> labels;

    std::random_device random;
    std::mt19937_64 mt(random());

    for (int i = 0; i < 1000; i++) {
        Scalar one = std::uniform_real_distribution<Scalar>(-1, 1)(mt);
        Scalar two = std::uniform_real_distribution<Scalar>(-1, 1)(mt);
        Scalar three = std::uniform_real_distribution<Scalar>(-1, 1)(mt);
        Scalar sum = one + two + three;

        training_set.push_back({one, two, three});
        labels.push_back({sum});
    }


    Sequential model{
            new Dense<ReLU>(3, 1, false),
    };

    MeanSquaredError mse;
    SGD sgd;

    model.Compile(mse, sgd);
    model.Fit(training_set, labels, 15, 1);

    Tensor<2> input(3, 1);
    input.setValues({{3},
                     {2},
                     {6}});

    Tensor<2> expected(1, 1);
    expected.setValues({{11}});

    std::cout << "predict: " << model.Predict(input).format(
            Eigen::TensorIOFormat(Eigen::FullPrecision)) << "\n";
    std::cout << model[0].GetWeights() << "\n";
    model.GradientCheck(input, expected, 1e-7);
}


// https://colab.research.google.com/drive/1aRv7pK2ClYi0os3Be8-5Hxy9c_t55ATg
void test()
{
    using namespace orion;

    int num_inputs = 1;
    int num_outputs_layer1 = 1;

    int layer1_weight_constant = 1;
    int layer2_weight_constant = 1;

    Sequential model{
            //new Dense<Sigmoid>(num_inputs, num_outputs_layer1, false),
            new Dense<Sigmoid>(num_outputs_layer1, 1, false),
    };

    std::vector<std::vector<Scalar>> inputs{
            std::vector<Scalar>(num_inputs, 0.4)};
    std::vector<std::vector<Scalar>> labels{{1}};

    Tensor<2> weights1(num_outputs_layer1, num_inputs);
    weights1.setConstant(layer1_weight_constant);

    Tensor<2> weights2(1, num_outputs_layer1);
    weights2.setConstant(layer2_weight_constant);

    model[0].SetWeights(weights1);
    //model[1].SetWeights(weights2);

    SGD opt;
    BinaryCrossEntropy loss(1e-7);

    TensorMap<2> infer(inputs.front().data(), inputs.front().size(), 1);

    std::cout << "Predit before fit:\n" << model.Predict(infer) << "\n\n";

    std::cout << "loss: " << loss(model.Predict(infer),
                                  TensorMap<2>(labels.front().data(), 1, 1))
            << "\n\n";

    model.Compile(loss, opt);
    model.Fit(inputs, labels, 1, 1);

    std::cout << "Weights1:\n" << model[0].GetWeights() << "\n\n";
    //std::cout << "Weights2:\n" << model[1].GetWeights() << "\n\n";

    std::cout << "Predit before fit:\n" << model.Predict(infer) << "\n\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    learn_xor();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "\n\n" << "Execution Time: " << ms.count() / 1e6 << " ms";

    return 0;
}