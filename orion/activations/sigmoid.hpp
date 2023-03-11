//
// Created by macross on 8/7/22.
//

#ifndef ORION_SIGMOID_HPP
#define ORION_SIGMOID_HPP

#include "orion/orion_types.hpp"

namespace orion
{

// Tensorflow automatically swaps sigmoid with softmax if used with
// categorical cross entropy(except for tf.nn.sigmoid), that is not done here
class Sigmoid
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
        return tensor.sigmoid();
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
        return tensor.sigmoid() * (one - tensor.sigmoid());
    }
};

} // namespace orion

#endif //ORION_SIGMOID_HPP
