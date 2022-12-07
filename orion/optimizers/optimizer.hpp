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
    explicit Optimizer(Scalar learning_rate) : learning_rate(learning_rate)
    {}

    virtual void Step() = 0;

    // skip bias correction for most implementations
    virtual void Minimize(Tensor<2> &W, const Tensor<2> &dL_dW) = 0;

    virtual void Minimize(Tensor<1> &b, const Tensor<1> &dL_db) = 0;

    virtual void Minimize(Tensor<4> &W, const Tensor<4> &dL_dW)
    { throw std::logic_error("Optimizer::Minimize based class called"); };

    Scalar GetLearningRate() const
    { return this->learning_rate; };

protected:
    Scalar learning_rate;
};

} // namespace orion

#endif //ORION_OPTIMIZER_HPP
