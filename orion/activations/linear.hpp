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


    template<int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return Tensor<TensorRank>(features.dimensions()).setConstant(one);
    }
};

}

#endif //ORION_LINEAR_HPP