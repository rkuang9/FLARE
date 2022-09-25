//
// Created by macross on 9/23/22.
//

#ifndef ORION_SELU_HPP
#define ORION_SELU_HPP

#include "orion/orion_types.hpp"

namespace orion
{

// https://arxiv.org/abs/1706.02515
class SELU
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        // values from the paper, but tensorflow uses alpha=1.7580993408473768599402175208123
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (features > zero).select(
                features,
                alpha * (features.exp() - features.constant(one)));
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (features > zero).select(
                features,
                alpha * (features.exp() - features.constant(one)));
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (features > zero).select(
                features.constant(one),
                alpha * features.exp());
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        auto alpha = static_cast<Scalar>(1.6732632423543772848170429916717);
        auto scale = static_cast<Scalar>(1.0507009873554804934193349852946);
        auto zero = static_cast<Scalar>(0.0);
        auto one = static_cast<Scalar>(1.0);

        return scale * (features > zero).select(
                features.constant(one),
                alpha * features.exp());
    }
};

} // namespace orion

#endif //ORION_SELU_HPP
