#include <iostream>
#include <chrono>
#include <orion/orion.hpp>

void dropout()
{
    using namespace orion;

    Tensor<2> weights(10, 10);
    weights.setConstant(1.0);

    Tensor<1> sample(10), label(10);
    sample.setValues({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    label.setZero();
    Dataset dataset(sample.dimensions(), label.dimensions());
    dataset.Add(sample, label);
    dataset.Batch(1);

    Sequential model {
        new Dense<Linear>(10, 10, false),
        new Dropout<2>(0.5),
    };

    model.layers[0]->SetWeights(weights);

    MeanSquaredError loss;
    SGD opt(1);

    model.Compile(loss, opt);
    model.Fit(dataset.training_samples, dataset.training_labels, 1);
    std::cout << model.Predict(dataset.training_samples.front());
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    dropout();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic