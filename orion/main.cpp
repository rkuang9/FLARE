#include <iostream>
#include <orion/orion.hpp>
#include <iomanip>


void dense_batch_test()
{
    using namespace orion;

    Tensor<2> input(3, 2);
    input.setRandom();

    Tensor<2> label(1, 2);
    label.setRandom();

    Sequential model{
            new Dense<Sigmoid>(3, 2, false),
            new Dense<Sigmoid>(2, 1, false),
    };

    BinaryCrossEntropy loss;
    Adam opt;

    model.Compile(loss, opt);
    model.Fit({input}, {label}, 1, 1);
    std::cout << "gradient check\n" << model.GradientCheck(input, label, 1e-7);
}


void embedding_development()
{
    using namespace orion;


    BinaryCrossEntropy loss;
    Adam opt(1);

    Tensor<2> embed_weights(5, 1);
    embed_weights.setRandom();
    std::cout << "embed weights\n" << embed_weights << "\n";


    Tensor<2> inputs(3, 1);
    inputs.setValues({{0},
                      {1},
                      {2}});

    Tensor<2> label(1, 1);
    label.setValues({{0}});


    Sequential model{
            new Embedding(5, 1, 3),
            new GlobalAveragePooling1D(),
    };
    std::cout << "predict " << model.Predict(inputs) << "\n";
    model[0].SetWeights(embed_weights);
    model.Compile(loss, opt);
    model.Fit({inputs}, {label}, 1, 1);

    std::cout << "updated weights\n" << model[0].GetWeights() << "\n";
    std::cout << std::setprecision(13) << "loss " << loss.GetLoss() << "\n";
    std::cout << "loss gradients " << loss.GetGradients2D() << "\n";
    std::cout << "gradient check: " << model.GradientCheck(inputs, label, 1e-7) << "\n";

}


void adam_compare_with_tf()
{
    using namespace orion;

    SGD sgd;
    Adam adam(0.001, 0.9, 0.999);

    BinaryCrossEntropy bce;
    MeanSquaredError mse;

    int input_features = 18;
    int label_features = 14;
    int batch_size = 1;
    int epochs = 10;

    std::vector<std::vector<Scalar>> training_set{
            std::vector<std::vector<Scalar>>(batch_size,
                                             std::vector<Scalar>(input_features,
                                                                 0.2))
    };

    std::vector<std::vector<Scalar>> training_labels{
            std::vector<std::vector<Scalar>>(batch_size,
                                             std::vector<Scalar>(label_features,
                                                                 0.5))
    };

    Tensor<2> w1(input_features, input_features);
    w1.setConstant(0.3);

    Tensor<2> w2(label_features, input_features);
    w2.setConstant(0.43);

    Tensor<2> w_temp1(13, input_features);
    w_temp1.setConstant(0.1);

    Tensor<2> w_temp2(input_features, 13);
    w_temp2.setConstant(0.7);

    Sequential model{
            new Dense<Sigmoid>(input_features, input_features, false),
            new Dense<Sigmoid>(input_features, 13, false),
            new Dense<Sigmoid>(13, input_features, false),
            new Dense<Sigmoid>(input_features, label_features, false),
    };

    model[0].SetWeights(w1);
    model[1].SetWeights(w_temp1);
    model[2].SetWeights(w_temp2);
    model[3].SetWeights(w2);

    model.Compile(mse, adam);
    model.Fit(training_set, training_labels, epochs, batch_size);
    std::cout << "gradient check: "
            << model.GradientCheck(
                    TensorMap<2>(training_set.front().data(), input_features,
                                 batch_size),
                    TensorMap<2>(training_labels.front().data(), label_features,
                                 batch_size)) << "\n";


    for (Layer *layer: model.layers) {
        std::cout << layer->name << " weights:\n" << layer->GetWeights() << "\n";
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    embedding_development();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}