//
// Created by macross on 9/23/22.
//

#ifndef FLARE_SOFTSIGN_HPP
#define FLARE_SOFTSIGN_HPP

#include "flare/fl_types.hpp"

namespace fl
{

class Softsign
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
        return tensor / (one + tensor.abs());
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
        return one / (one + tensor.abs()).square();
    }
};

} // namespace fl

#endif //FLARE_SOFTSIGN_HPP
