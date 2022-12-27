//
// Created by macross on 9/23/22.
//

#ifndef ORION_SOFTSIGN_HPP
#define ORION_SOFTSIGN_HPP

#include "orion/orion_types.hpp"

namespace orion {

class Softsign
{
public:
    template <int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return features / (one + features.abs());
    }


    template <int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return one / (one + features.abs()).square();
    }
};

} // namespace orion

#endif //ORION_SOFTSIGN_HPP
