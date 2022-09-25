//
// Created by macross on 9/23/22.
//

#ifndef ORION_ELU_HPP
#define ORION_ELU_HPP

#include "orion/orion_types.hpp"

namespace orion
{

/**
 * The exponential linear unit (ELU) activation function with alpha=1.0
 * Reference: https://arxiv.org/abs/1511.07289
 */
class ELU
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (features > zero).select(
                features,
                features.exp() - one);
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (features > zero).select(
                features,
                features.exp() - one);
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (features > zero).select(
                features.constant(one),
                ELU::Activate(features) + one);
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return (features > zero).select(
                features.constant(one),
                ELU::Activate(features) + one);
    }
};

} // namespace orion

#endif //ORION_ELU_HPP
