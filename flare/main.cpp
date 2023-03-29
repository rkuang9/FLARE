#include <iostream>
#include <chrono>
#include <flare/flare.hpp>


void test()
{
    fl::MeanSquaredError<3> loss;
    fl::RMSprop opt(1.0);

    fl::Sequential model {
            new fl::Embedding(10, 4, 3),
    };

    fl::Tensor<2> input(2, 3);
    input.setValues({{7, 8, 9},
                     {0, 1, 2}});

    for (int epoch = 0; epoch < 1; epoch++) {
        model.Forward(input);
    std::cout << model.layers.back()->GetOutput3D() << "\n";
        loss(model.layers.back()->GetOutput3D(),
             model.layers.back()->GetOutput3D().constant(1.0));

        model.Backward(loss.GetGradients());
        model.Update(opt);
    }

    std::cout << model.layers.back()->GetWeightGradients() << "\n\n";
    std::cout << "embed weight \n" << model.layers.back()->GetWeights() << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    test();

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