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
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        return features.cwiseMax(0.0);
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        return features.cwiseMax(0.0);
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        return (features >= 0).select(features.constant(1.0),
                                      features.constant(0.0));
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        return (features >= 0).select(features.constant(1.0),
                                      features.constant(0.0));
    }
};

} // namespace orion

#endif //ORION_RELU_HPP
