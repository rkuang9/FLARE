#include <iostream>
#include <chrono>

#include <orion/orion.hpp>

int XOR()
{
    using namespace orion;

    // random number generator
    std::random_device random;
    std::mt19937_64 mt(random());

    std::vector<std::vector<Scalar>> training_set;
    std::vector<std::vector<Scalar>> labels;

    // generate the dataset
    for (int i = 0; i < 10000; i++) {
        Scalar x = std::uniform_int_distribution<int>(0, 1)(mt);
        Scalar y = std::uniform_int_distribution<int>(0, 1)(mt);
        Scalar z = ((bool)x != (bool)y);

        training_set.push_back({x, y});
        labels.push_back({z});
    }

    Sequential model{
            new Dense<Sigmoid>(2, 4, false),
            new Dense<Sigmoid>(4, 1, false),
    };

    BinaryCrossEntropy loss;
    SGD opt(0.01);

    model.Compile(loss, opt);

    // train for 30 epochs with batch size 1
    model.Fit(training_set, labels, 30, 1);


    Tensor<2> p1(2, 1), p2(2, 1), p3(2, 1), p4(2, 1);
    p1.setValues({{1}, {1}});
    p2.setValues({{1}, {0}});
    p3.setValues({{0}, {1}});
    p4.setValues({{0}, {0}});

    std::cout << p1.reshape(Tensor<2>::Dimensions(1, 2)) << ", expects 0: " << model.Predict(p1) << "\n";
    std::cout << p2.reshape(Tensor<2>::Dimensions(1, 2)) << ", expects 1: " << model.Predict(p2) << "\n";
    std::cout << p3.reshape(Tensor<2>::Dimensions(1, 2)) << ", expects 1: " << model.Predict(p3) << "\n";
    std::cout << p4.reshape(Tensor<2>::Dimensions(1, 2)) << ", expects 0: " << model.Predict(p4) << "\n";
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    XOR();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";
}