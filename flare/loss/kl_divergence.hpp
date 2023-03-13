//
// Created by Macross on 12/14/22.
//

#ifndef FLARE_KL_DIVERGENCE_HPP
#define FLARE_KL_DIVERGENCE_HPP

#include "loss_function.hpp"

namespace fl
{

template<int TensorRank>
class KLDivergence : public LossFunction<TensorRank>
{
public:
    KLDivergence() = default;


    KLDivergence &operator+(LossFunction<TensorRank> &other) override
    {
        this->loss += other.GetLoss();
        this->gradients += other.GetGradients();
        return *this;
    }


    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label) override
    {
        fl_assert(predict.dimensions() == label.dimensions(),
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
        fl_assert(predict.dimensions() == label.dimensions(),
                  "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        auto zero = static_cast<Scalar>(0.0);

        return (predict != zero).select(-label / (predict), predict.constant(zero)) /
               static_cast<Scalar>(predict.size());
    }
};

}

#endif //FLARE_KL_DIVERGENCE_HPP
