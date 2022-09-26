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
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return features / (one + features.abs());
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return features / (one + features.abs());
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return one / (one + features.abs()).square();
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return one / (one + features.abs()).square();
    }
};

} // namespace orion

#endif //ORION_SOFTSIGN_HPP
