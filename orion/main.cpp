#include <iostream>
#include <chrono>
#include <vector>
#include <orion/orion.hpp>
#include <filesystem>

/*#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>*/

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic

using namespace orion;


void softmax()
{
    Tensor<2> weights(3, 3);
    weights.setValues({{1, 2, 3},
                       {1, 2, 3},
                       {1, 2, 3}});

    Tensor<2> labels(2, 3);
    labels.setValues({{0, 0, 1},
                      {0, 0, 1}});

    CategoricalCrossEntropy cce;
    SGD sgd(1);

    Tensor<2> X(2, 3);
    X.setValues({{0.2, 0.1, 0.3},
                 {0.4, 0.5, 0.6}});

    Layer *dense = new Dense<Softmax>(3, 3, false);
    dense->SetWeights(weights);

    dense->Forward(X);
    cce.CalculateLoss(dense->GetOutput2D(), labels);
    Tensor<2> loss_grad = cce.GetGradients2D();
    dense->Backward(cce);
    dense->Update(sgd);
    std::cout << "updated weights\n" << dense->GetWeights() << "\n";
    return;

    Tensor<3> softmax_grad = Softmax::Gradients(dense->GetOutput2D());

    std::cout << "loss gradients\n" << cce.GetGradients2D() << "\n\n";

    std::cout << "jacobian: " << softmax_grad << "\n\n";

    std::cout << "chip\n" << loss_grad.chip(1, 0) << "\n" << softmax_grad.chip(1, 0)
              << "\n";
    Tensor<1> chipy = loss_grad.chip(1, 0);
    std::cout << "chipy dim: " << chipy.dimensions() << "\n";
    Tensor<2> chipper = softmax_grad.chip(1, 0);
    std::cout << "chipper dim: " << chipper.dimensions() << "\n";

    std::cout << "chippest dim: "
              << chipy.contract(chipper, ContractDim {Axes(0, 0)}) << "\n";

    Tensor<2> dL_dZ(2, 3);
    dL_dZ.chip(0, 0) = loss_grad.chip(0, 0).contract(softmax_grad.chip(0, 0),
                                                     ContractDim {Axes(0, 1)});
    std::cout << "chip 1\n";
    dL_dZ.chip(1, 0) = loss_grad.chip(1, 0).contract(softmax_grad.chip(1, 0),
                                                     ContractDim {Axes(0, 1)});
    std::cout << "chip 2\n";
    std::cout << "dL_dZ\n" << dL_dZ << "\n\n";


}


void crossentropy()
{
    Tensor<1> input1(3);
    input1.setValues({0.2, 0.1, 0.3});
    Tensor<1> label1(3);
    label1.setValues({0, 0, 1});

    Tensor<1> input2(3);
    input2.setValues({0.4, 0.5, 0.6});
    Tensor<1> label2(3);
    label2.setValues({0, 0, 1});

    Dataset dataset(Dims<1>(3), Dims<1>(3));
    dataset.Add(input1, label1);
    dataset.Add(input2, label2);
    dataset.Batch(2);

    Tensor<2> custom_weights(3, 3);
    custom_weights.setValues({{1, 2, 3},
                              {1, 2, 3},
                              {1, 2, 3}});

    Sequential model {
            new Dense<Softmax>(3, 3, false),
    };

    model.layers[0]->SetWeights(custom_weights);

    CategoricalCrossEntropy cce;
    SGD sgd(1);

    model.Compile(cce, sgd, {new Loss});
    model.Fit(dataset.training_samples, dataset.training_labels, 1, 1);

    std::cout << "prediction\n" << model.layers.back()->GetOutput2D() << "\n";

    std::cout << "gradient check (1e-7 and smaller means backpropagation is accurate: "
              << model.GradientCheck(dataset.training_samples.front(),
                                     dataset.training_labels.front());
}



int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    //mnist();
    //softmax();
    crossentropy();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() << "s";

    return 0;
}