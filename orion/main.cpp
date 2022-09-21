#include <iostream>
#include <orion/orion.hpp>
#include <Eigen/Dense>


void gap1d_development()
{
    using namespace orion;

    Tensor<2> tensor2d(2, 4);
    tensor2d.setValues({{1, 1, 1, 1},
                        {1, 1, 1, 2}});


    Tensor<2> label(2, 1);
    label.setZero();

    GlobalAveragePooling1D gap1d;
    gap1d.Forward(tensor2d.reshape(Tensor<3>::Dimensions(2, 4, 1)));

    SGD opt(1);

    MeanSquaredError loss;
    loss.CalculateLoss(gap1d.GetOutput2D(), label);
    std::cout << "loss\n" << loss.GetGradients2D() << "\n\n";

    std::cout << "gap1d output\n" << gap1d.GetOutput2D() << "\n\n";

    gap1d.Backward(loss);

    std::cout << "gap1d dL_dZ\n" << gap1d.GetInputGradients2D() << "\ndimensions\n"
            << gap1d.GetInputGradients2D().dimensions() << "\n\n";

    Tensor<2> gap1d_grad = gap1d.GetInputGradients2D().broadcast(
            Tensor<2>::Dimensions(1, 4));
    std::cout << "bcast gap1d dL_dZ\n" << gap1d_grad << "\n\n";
}


void embedding_development()
{
    using namespace orion;

    int batch_size = 1;
    int num_inputs = 3;
    int label_size = 3;

    MeanSquaredError loss;
    SGD opt(1);

    Tensor<2> embed_weights(5, 3);
    embed_weights.setValues({{0.01246039,  0.01246468,  -0.04545361},
                             {0.04710307,  -0.01240801, -0.03737292},
                             {-0.02046142, 0.04035002,  -0.0373662},
                             {-0.0009598,  -0.00118566, 0.02462603},
                             {0.04840514,  0.01805499,  0.01568809}});
    embed_weights.setValues({{0, 0, 0},
                             {1, 1, 2},
                             {2, 2, 2},
                             {3, 3, 3},
                             {4, 4, 4}});
    //embed_weights.setConstant(1);

    Tensor<2> inputs(num_inputs, batch_size);
    inputs.setValues({{0},
                      {1},
                      {2}});

    Tensor<2> label(batch_size, label_size);
    label.setValues({{0, 0, 0}});


    /*Sequential model{
            new Embedding(5, 3, 3),
            new GlobalAveragePooling1D(),
    };

    model[0].SetWeights(embed_weights);

    model.Compile(loss, opt);
    model.Fit({inputs}, {label}, 1, 1);
    std::cout << "loss: " << loss.GetLoss() << "\n";
    std::cout << model[0].GetWeights() << "\n";
    std::cout << "gradient check: " << model.GradientCheck(inputs, label, 1e-7);
    return;*/


    Layer *embed = new Embedding(5, 3, num_inputs);
    Layer *gap1d = new GlobalAveragePooling1D();
    embed->SetWeights(embed_weights);

    embed->Forward(inputs);
    std::cout << "embedding outputs\n" << embed->GetOutput3D() << "\n";
    gap1d->Forward(*embed);
    std::cout << "gap1d outputs\n" << gap1d->GetOutput2D() << "\n";
    loss.CalculateLoss(gap1d->GetOutput2D(), label);
    std::cout << "loss " << loss.GetLoss() << "\nloss gradients\n"
            << loss.GetGradients2D() << "\n";
    gap1d->Backward(loss);
    embed->Backward(*gap1d);
    embed->Update(opt);
    std::cout << "embedding updated weights\n" << embed->GetWeights() << "\n";


    return;
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