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
    template<int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        return features;
    }


    /*template<int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        return Tensor<TensorRank>().template device(Eigen::DefaultDevice()) = features.constant(1.0);
        auto one = static_cast<Scalar>(1.0);
        return Tensor<TensorRank>(features.dimensions()).setConstant(one);
    }*/


    /**
     * Compute the activation function gradients
     * @param tensor   Eigen::Tensor or Eigen::Tensor Op
     * @param output   Eigen::Tensor or Eigen::Tensor Op to assign output to
     * @param device   device such as Eigen::ThreadPoolDevice or Eigen::DefaultDevice
     * @return         Eigen::TensorO Op
     */
    template<typename TensorX>
    static auto Gradients(const TensorX &tensor)
    {
        return tensor.constant(1.0);
        /*
        auto one = static_cast<Scalar>(1.0);
        output. template device(device) = tensor.constant(one);*/
    }
};

}

#endif //ORION_LINEAR_HPP