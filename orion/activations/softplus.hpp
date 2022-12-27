//
// Created by macross on 9/23/22.
//

#ifndef ORION_SOFTPLUS_HPP
#define ORION_SOFTPLUS_HPP

#include "orion/orion_types.hpp"

namespace orion
{

// doesn't work with binary cross entropy
class Softplus
{
public:
    template <int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return (one + features.exp()).log();
    }


    template <int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        // gradient of ln(1 + e^x) = e^x / (1 + e^x) = 1 / (e^-x + 1) = sigmoid(x)
        return features.sigmoid();
    }
};

} // namespace orion

#endif //ORION_SOFTPLUS_HPP
