//
// Created by macross on 8/8/22.
//

#ifndef ORION_OPTIMIZER_HPP
#define ORION_OPTIMIZER_HPP

#include "orion/orion_types.hpp"
#include <map>


namespace orion
{

// https://arxiv.org/pdf/1609.04747.pdf
class Optimizer
{
public:
    explicit Optimizer(Scalar learning_rate) : learning_rate(learning_rate) {}

    virtual void Step() = 0;

    // skip bias correction for most implementations
    virtual void Minimize(Tensor2D &W, const Tensor2D &dL_dW) = 0;

    virtual void Minimize(Tensor1D &b, const Tensor1D &dL_db) = 0;

    Scalar GetLearningRate() const { return this->learning_rate; };

protected:
    Scalar learning_rate;
};

} // namespace orion

#endif //ORION_OPTIMIZER_HPP
