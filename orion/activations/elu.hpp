//
// Created by macross on 9/23/22.
//

#ifndef ORION_ELU_HPP
#define ORION_ELU_HPP

#include "orion/orion_types.hpp"

namespace orion
{

/**
 * The exponential linear unit (ELU) activation function with alpha=1.0
 * Reference: https://arxiv.org/abs/1511.07289
 */

class ELU
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
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (tensor > zero).select(
                tensor,
                tensor.exp() - one);
    }


    /**
     * Compute the activation gradients of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (tensor > zero).select(
                tensor.constant(one),
                ELU::Activate(tensor) + one);
    }
};

} // namespace orion

#endif //ORION_ELU_HPP
