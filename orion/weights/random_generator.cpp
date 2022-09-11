//
// Created by RKuang on 8/30/2022.
//

#include "random_generator.hpp"

namespace orion::random
{
    std::random_device random;
    std::mt19937_64 mt(random());


    Tensor2D RandomUniform(int rows, int cols, Scalar min, Scalar max)
    {
        return Tensor2D(rows, cols).nullaryExpr([&]() {
            return std::uniform_real_distribution<Scalar>(min, max)(mt);
        });
    }


    Tensor2D RandomNormal(int rows, int cols, Scalar mean, Scalar stddev)
    {
        return Tensor2D(rows, cols).nullaryExpr([&]() {
            return std::normal_distribution<Scalar>(mean, stddev)(mt);
        });
    }
}