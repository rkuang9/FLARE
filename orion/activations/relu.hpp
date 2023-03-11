//
// Created by macross on 8/8/22.
//

#ifndef ORION_RELU_HPP
#define ORION_RELU_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class ReLU
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
        return tensor.cwiseMax(zero);
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
        auto zero = static_cast<Scalar>(0.0);

        return (tensor >= zero).select(tensor.constant(one),
                                       tensor.constant(zero));
    }
};

} // namespace orion

#endif //ORION_RELU_HPP
