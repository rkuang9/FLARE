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
    template <int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return features.exp() / features.exp().sum();
    }


    template <int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        throw std::logic_error("softmax has issues, not working yet");
        return (features >= 0).select(features.constant(1.0), features.constant(0.0));
    }
};

} // namespace orion

#endif //ORION_SOFTMAX_HPP
