//
// Created by macross on 8/18/22.
//

#ifndef ORION_LOSS_HPP
#define ORION_LOSS_HPP

#include <vector>

#include "orion/orion_types.hpp"
#include "orion/orion_assert.hpp"

namespace orion
{

class Loss
{
public:
    Loss() = default;


    explicit Loss(Scalar epsilon) : epsilon(epsilon)
    {}


    virtual void CalculateLoss(const Tensor<2> &predict,
                               const Tensor<2> &label) = 0;

    virtual Scalar operator()(const Tensor<2> &predict,
                              const Tensor<2> &label) const = 0;


    const std::vector<Tensor<2>> &GradientHistory() const
    { return this->gradient_history; }


    const std::vector<Scalar> &LossHistory() const
    { return this->loss_history; }


    virtual Scalar GetLoss() const
    { return this->loss_history.back(); }


    virtual Tensor<2> GetGradients2D() const
    { return this->gradient_history.back(); }

public:
    Scalar epsilon = 1e-07; // numeric stability constant
    Scalar clip_min = epsilon;
    Scalar clip_max = 1 - epsilon;
    std::vector<Scalar> loss_history;
    std::vector<Tensor<2>> gradient_history;
};

}

#endif //ORION_LOSS_HPP
