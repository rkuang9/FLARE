//
// Created by RKuang on 9/22/2022.
//

#ifndef ORION_SWISH_HPP
#define ORION_SWISH_HPP

#include "orion/orion_types.hpp"
#include "sigmoid.hpp"

namespace orion
{
// TODO: prone to exploding gradients, need to find a way to clip in optimizers
// https://arxiv.org/pdf/1710.05941.pdf
class Swish
{
public:
    template <int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        // a = z/(1 + e^-z) = z * sigmoid(z)
        return features * features.sigmoid();
    }


    template <int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);

        Tensor<TensorRank> swish = features * features.sigmoid();
        return swish + features.sigmoid() * (one - swish);
    }
};

}

#endif //ORION_SWISH_HPP
