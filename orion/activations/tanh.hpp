//
// Created by macross on 8/8/22.
//

#ifndef ORION_TANH_HPP
#define ORION_TANH_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class TanH
{
public:
    template <int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        return features.tanh();
    }


    template <int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return one - features.tanh().square();
    }
};

} // namespace orion

#endif //ORION_TANH_HPP
