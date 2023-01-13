#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

orion::Dims<4> NCHW(0, 3, 1, 2);
orion::Dims<5> NPCHW(0, 1, 4, 2, 3);


void BatchNorm()
{
    using namespace orion;

    MeanSquaredError loss;
    Adam opt(1.0);

    Tensor<2> img(3, 3), labels(img.dimensions());
    img.setValues({{2, 3, 1},
                   {1, 1, 4},
                   {2, 6, 3}});
    labels.setValues({{5, 6, 1},
                      {1, 6, 1},
                      {8, 2, 6}});

    Tensor<2> weights(3, 3);
    weights.setConstant(1.0);
    weights.setValues({{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}});


    Sequential model {
            new Dense<Sigmoid>(3, 3, false),
            new Dense<Sigmoid>(3, 3, false),
            new BatchNormalization<2, 1>(Dims<1>(1), 0.99, 0, true),
    };

    model[0].SetWeights(weights);
    model[1].SetWeights(weights);

    model.Compile(loss, opt);
    model.Fit(std::vector<Tensor<2>> {img}, std::vector<Tensor<2>> {labels}, 1);
    std::cout << "output grad\n" << loss.GetGradients2D() << "\n";
    std::cout << "dense weights\n" << model.layers.front()->GetWeights() << "\n";
    std::cout << "dense weights\n" << model.layers[1]->GetWeights() << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    BatchNorm();

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