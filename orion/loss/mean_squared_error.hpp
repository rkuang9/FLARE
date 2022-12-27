//
// Created by macross on 9/5/22.
//

#ifndef ORION_MEAN_SQUARED_ERROR_HPP
#define ORION_MEAN_SQUARED_ERROR_HPP

#include "loss_function.hpp"

namespace orion
{

class MeanSquaredError : public LossFunction
{
public:
    explicit MeanSquaredError(int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};


template<int TensorRank>
Scalar MeanSquaredError::Loss(const Tensor<TensorRank> &predict,
                              const Tensor<TensorRank> &label)
{
    return Tensor<0>((label - predict).square().mean())(0);
}

template<int TensorRank>
Tensor<TensorRank> MeanSquaredError::Gradient(const Tensor<TensorRank> &predict,
                                              const Tensor<TensorRank> &label)
{
    return 2 * (predict - label) /
           (Scalar(predict.dimensions().TotalSize() / predict.dimension(0)));
}

} // orion

#endif //ORION_MEAN_SQUARED_ERROR_HPP
