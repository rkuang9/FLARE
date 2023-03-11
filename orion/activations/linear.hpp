//
// Created by macross on 8/17/22.
//

#ifndef ORION_LINEAR_HPP
#define ORION_LINEAR_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Linear
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
        return tensor;
    }


    /**
     * Compute the activation gradients of a tensor
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        return tensor.constant(1.0);
    }
};

}

#endif //ORION_LINEAR_HPP