//
// Created by macross on 8/18/22.
//

#ifndef ORION_LOSS_FUNCTION_HPP
#define ORION_LOSS_FUNCTION_HPP

#include <vector>

#include "orion/orion_types.hpp"
#include "orion/orion_assert.hpp"

namespace orion
{

 /*
  * The loss function classes can contain up to 4 hard coded functions for tensor
  * returns (2D, 3D, 4D, 5D)
  * Currently unable to find a better way to implement this given the layer
  * base class and virtual functions
  */
class LossFunction
{
public:
    LossFunction() = default;


    explicit LossFunction(Scalar epsilon) : epsilon(epsilon) {}


    virtual void CalculateLoss(const Tensor<2> &predict,
                               const Tensor<2> &label) = 0;

    virtual void CalculateLoss(const Tensor<4> &predict,
                               const Tensor<4> &label) = 0;

    const std::vector<Scalar> &LossHistory() const
    { return this->loss_history; }

    Scalar GetLoss() const
    { return this->loss_history.back(); }

    const std::vector<Tensor<2>> &GradientHistory2D() const
    { return this->gradient_history2D; }


    const std::vector<Tensor<4>> &GradientHistory4D() const
    { return this->gradient_history4D; }


    const Tensor<2> &GetGradients2D() const
    { return this->gradient_history2D.back(); }


    const Tensor<4> &GetGradients4D() const
    { return this->gradient_history4D.back(); }


protected:
    Scalar epsilon = 1e-07; // numeric stability constant
    Scalar clip_min = epsilon;
    Scalar clip_max = 1 - epsilon;
    std::vector<Scalar> loss_history;
    std::vector<Tensor<2>> gradient_history2D;
    std::vector<Tensor<4>> gradient_history4D;
};

}

#endif //ORION_LOSS_FUNCTION_HPP
