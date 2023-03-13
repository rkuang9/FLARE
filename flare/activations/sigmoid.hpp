//
// Created by macross on 8/7/22.
//

#ifndef FLARE_SIGMOID_HPP
#define FLARE_SIGMOID_HPP

#include "flare/fl_types.hpp"

namespace fl
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

} // namespace fl

#endif //FLARE_SIGMOID_HPP
