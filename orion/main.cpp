#include <iostream>
#include <orion/orion.hpp>

void gap1d_development()
{
    using namespace orion;

    Tensor<2> tensor2(2, 3);
    tensor2.setValues({
        {1, 2, 3},
        {2, 4, 6},
    });

    std::cout << tensor2.mean(Tensor<1>::Dimensions(2));
}


void embedding_development()
{
    using namespace orion;

    Tensor<2> weights(4, 3);
    weights.setValues({{1, 1, 2},
                       {2, 2, 3},
                       {3, 3, 4},
                       {4, 4, 5}});

    Tensor<2> input(2, 1);
    input.setValues({{0}, {1}});

    Layer *embedding = new Embedding(4, 3, 2);
    embedding->SetWeights(weights);

    embedding->Forward(input);


    std::cout << "embedding weights\n" << embedding->GetWeights() << "\n";
    std::cout << "embedding output\n" << embedding->GetOutput3D() << "\n";
    std::cout << "embedding output dimensions: " << embedding->GetOutput3D().dimensions() << "\n";



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

    gap1d_development();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}