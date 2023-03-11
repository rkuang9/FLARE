//
// Created by macross on 9/23/22.
//

#ifndef ORION_SOFTPLUS_HPP
#define ORION_SOFTPLUS_HPP

#include "orion/orion_types.hpp"

namespace orion
{

// doesn't work with binary cross entropy
class Softplus
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
        auto one = static_cast<Scalar>(1.0);
        return (one + tensor.exp()).log();
    }


    /**
     * Compute the activation gradients of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        // gradient of ln(1 + e^x) = e^x / (1 + e^x) = 1 / (e^-x + 1) = sigmoid(x)
        return tensor.sigmoid();
    }
};

} // namespace orion

#endif //ORION_SOFTPLUS_HPP
