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
    template<int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        return features.cwiseMax(zero);
    }


    template<int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        auto zero = static_cast<Scalar>(0.0);

        return (features >= zero).select(features.constant(one),
                                         features.constant(zero));
    }
};

} // namespace orion

#endif //ORION_RELU_HPP
