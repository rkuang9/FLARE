//
// Created by macross on 8/8/22.
//

#ifndef FLARE_OPTIMIZER_HPP
#define FLARE_OPTIMIZER_HPP

#include "flare/fl_types.hpp"
#include <map>


namespace fl
{

// https://arxiv.org/pdf/1609.04747.pdf
class Optimizer
{
public:
    explicit Optimizer(Scalar learning_rate) : learning_rate(learning_rate)
    {}


    virtual void Step() = 0;


    virtual void Minimize(Tensor<1> &W, const Tensor<1> &dL_dW) = 0;

    virtual void Minimize(Tensor<2> &W, const Tensor<2> &dL_dW) = 0;

    virtual void Minimize(Tensor<3> &W, const Tensor<3> &dL_dW) = 0;

    virtual void Minimize(Tensor<4> &W, const Tensor<4> &dL_dW) = 0;


    Scalar GetLearningRate() const
    { return this->learning_rate; };

protected:
    Scalar learning_rate;

    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(new Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency()), 2);
};

} // namespace fl

#endif //FLARE_OPTIMIZER_HPP
