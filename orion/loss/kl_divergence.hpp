//
// Created by Macross on 12/14/22.
//

#ifndef ORION_KL_DIVERGENCE_HPP
#define ORION_KL_DIVERGENCE_HPP

#include "loss_function.hpp"

namespace orion
{

template<int TensorRank>
class KLDivergence : public LossFunction<TensorRank>
{
public:
    KLDivergence() = default;


    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        return Tensor<0>(
                (label * ((label + this->epsilon) / (predict + this->epsilon))
                        .log()).mean()).coeff();
    }


    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        auto zero = static_cast<Scalar>(0.0);

        return (predict != zero).select(-label / (predict), predict.constant(zero)) /
               static_cast<Scalar>(predict.size());
    }
};

}

#endif //ORION_KL_DIVERGENCE_HPP
