//
// Created by Macross on 12/13/22.
//

#ifndef ORION_MEAN_ABSOLUTE_ERROR_HPP
#define ORION_MEAN_ABSOLUTE_ERROR_HPP

#include "loss_function.hpp"

namespace orion
{

template<int TensorRank>
class MeanAbsoluteError : public LossFunction<TensorRank>
{
public:
    MeanAbsoluteError() = default;


    MeanAbsoluteError &operator+(LossFunction<TensorRank> &other) override
    {
        this->loss += other.GetLoss();
        this->gradients += other.GetGradients();
        return *this;
    }


    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        return Tensor<0>((label - predict).abs().mean()).coeff();
    }


    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        return (predict != label)
                       .select((predict > label)
                                       .select(predict.constant(1),
                                               predict.constant(-1)),
                               predict.constant(0)) /
               static_cast<Scalar>(predict.dimensions().TotalSize() /
                                   predict.dimension(0));
    }
};

}

#endif //ORION_MEAN_ABSOLUTE_ERROR_HPP
