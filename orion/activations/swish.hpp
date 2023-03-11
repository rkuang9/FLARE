//
// Created by RKuang on 9/22/2022.
//

#ifndef ORION_SWISH_HPP
#define ORION_SWISH_HPP

#include "orion/orion_types.hpp"
#include "sigmoid.hpp"

namespace orion
{
// TODO: prone to exploding gradients, need to find a way to clip in optimizers
// TODO:
// https://arxiv.org/pdf/1710.05941.pdf
class Swish
{
public:
    /**
     * Compute the activation of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Activate(const TensorX &tensor)
    {
        // a = z/(1 + e^-z) = z * sigmoid(z)
        return tensor * tensor.sigmoid();
    }


    /**
     * Compute the activation gradients of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        auto one = static_cast<Scalar>(1.0);
        return tensor.sigmoid() * (tensor * (one - tensor.sigmoid()) + one);
        /*auto swish = tensor * tensor.sigmoid();
        return swish + tensor.sigmoid() * (one - swish);*/
    }
};

}

#endif //ORION_SWISH_HPP
