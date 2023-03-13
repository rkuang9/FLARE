//
// Created by macross on 8/8/22.
//

#ifndef FLARE_TANH_HPP
#define FLARE_TANH_HPP

#include "flare/fl_types.hpp"

namespace fl
{

class TanH
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
        return tensor.tanh();
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
        return one - tensor.tanh().square();
    }
};

} // namespace fl

#endif //FLARE_TANH_HPP
