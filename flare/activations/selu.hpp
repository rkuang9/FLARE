//
// Created by macross on 9/23/22.
//

#ifndef FLARE_SELU_HPP
#define FLARE_SELU_HPP

#include "flare/fl_types.hpp"

namespace fl
{

// https://arxiv.org/abs/1706.02515
class SELU
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
        // values from the paper, but tensorflow uses alpha=1.7580993408473768599402175208123
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (tensor > zero).select(
                tensor,
                alpha * (tensor.exp() - tensor.constant(one)));
    }


    /**
     * Compute the activation gradients of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (tensor > zero).select(
                tensor.constant(one),
                alpha * tensor.exp());
    }
};

} // namespace fl

#endif //FLARE_SELU_HPP
