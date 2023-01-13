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

    void CalculateLoss(const Tensor<3> &predict, const Tensor<3> &label) override;

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
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    return Tensor<0>((label - predict).abs().mean()).coeff();
}

}

#endif //ORION_MEAN_ABSOLUTE_ERROR_HPP
