//
// Created by macross on 8/8/22.
//

#ifndef ORION_SOFTMAX_HPP
#define ORION_SOFTMAX_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Softmax
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return features.exp() / features.exp().sum();
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return features.exp() / features.exp().sum();
    }


    static Tensor<3> Gradients(const Tensor<2> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return (features >= 0).select(features.constant(1.0), features.constant(0.0));
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return (features >= 0).select(features.constant(1.0), features.constant(0.0));
    }
};

} // namespace orion

#endif //ORION_SOFTMAX_HPP
