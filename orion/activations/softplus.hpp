//
// Created by macross on 9/23/22.
//

#ifndef ORION_SOFTPLUS_HPP
#define ORION_SOFTPLUS_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Softplus
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return (one + features.exp()).log();
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return (one + features.exp()).log();
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        return features.sigmoid();
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        return features.sigmoid();
    }
};

} // namespace orion

#endif //ORION_SOFTPLUS_HPP
