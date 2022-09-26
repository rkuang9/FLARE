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
        auto zero = static_cast<Scalar>(0.0);
        return features.cwiseMax(zero);
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        return features.cwiseMax(zero);
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        auto zero = static_cast<Scalar>(0.0);

        return (features >= zero).select(features.constant(one),
                                         features.constant(zero));
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        auto zero = static_cast<Scalar>(0.0);

        return (features >= zero).select(features.constant(one),
                                         features.constant(zero));
    }
};

} // namespace orion

#endif //ORION_RELU_HPP
