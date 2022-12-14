#include <iostream>
#include <chrono>
#include <orion/orion.hpp>


void mean_abs_error()
{
    using namespace orion;

    Tensor<1> sample(20);
    sample.setRandom();
    Tensor<1> label(20);
    label.setRandom();


    Dataset dataset(Dims<1>(20), Dims<1>(20));
    dataset.Add(sample, label);
    dataset.Batch(1);

    Sequential model {
            new Dense<Sigmoid>(20, 20, false),
    };

    KLDivergence loss;
    Adam opt;


    model.Compile(loss, opt);
    model.Fit(dataset.training_samples, dataset.training_labels, 1);
    std::cout << "loss: " << loss.GetLoss() << "\n";
    std::cout << "gradients: " << loss.GetGradients2D() << "\n";


    std::cout << model.GradientCheck(dataset.training_samples.front(),
                                     dataset.training_labels.front());
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    mean_abs_error();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic