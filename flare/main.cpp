#include <iostream>
#include <chrono>
#include <flare/flare.hpp>
#include "examples/heart_attack_prediction.h"


/*void test()
{
    fl::MeanSquaredError<3> loss;
    fl::SGD opt(1.0);

    fl::Tensor<3> x(1, 4, 1);
    x.setConstant(1.0);

    fl::LSTM<fl::TanH, fl::Sigmoid, true> lstm(1, 1);

    for (int i = 0; i < 2; i++) {
        lstm.Forward(x);
        std::cout << "output:\n" << lstm.GetOutput3D() << "\n";


        loss(lstm.GetOutput3D(), lstm.GetOutput3D().constant(1.0));
        std::cout << "loss gradients: " << loss.GetGradients() << "\n";

        lstm.Backward(loss.GetGradients());
        lstm.Update(opt);
        std::cout << "updated weights\n" << lstm.GetWeights() << "\n";
    }
}*/


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    HeartAttackPrediction();
    //test();

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