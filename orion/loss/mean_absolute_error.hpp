//
// Created by Macross on 12/13/22.
//

#ifndef ORION_MEAN_ABSOLUTE_ERROR_HPP
#define ORION_MEAN_ABSOLUTE_ERROR_HPP

#include "loss_function.hpp"

namespace orion
{

class MeanAbsoluteError : public LossFunction
{
public:
    explicit MeanAbsoluteError(int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};

template<int TensorRank>
Scalar MeanAbsoluteError::Loss(const Tensor<TensorRank> &predict,
                               const Tensor<TensorRank> &label)
{
    return Tensor<0>((label - predict).abs().mean()).coeff();
}


template<int TensorRank>
Tensor<TensorRank> MeanAbsoluteError::Gradient(const Tensor<TensorRank> &predict,
                                               const Tensor<TensorRank> &label)
{
    return (predict != label)
                   .select((predict > label)
                                   .select(predict.constant(1),
                                           predict.constant(-1)),
                           predict.constant(0)) /
           Scalar(predict.dimensions().TotalSize() / predict.dimension(0));
}

}

#endif //ORION_MEAN_ABSOLUTE_ERROR_HPP
