//
// Created by macross on 9/23/22.
//

#ifndef FLARE_ELU_HPP
#define FLARE_ELU_HPP

#include "flare/fl_types.hpp"

namespace fl
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

} // namespace fl

#endif //FLARE_ELU_HPP
