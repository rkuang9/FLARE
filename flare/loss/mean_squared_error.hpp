//
// Created by macross on 9/5/22.
//

#ifndef FLARE_MEAN_SQUARED_ERROR_HPP
#define FLARE_MEAN_SQUARED_ERROR_HPP

#include "loss_function.hpp"

namespace fl
{

template<int TensorRank>
class MeanSquaredError : public LossFunction<TensorRank>
{
public:
    MeanSquaredError() = default;


    MeanSquaredError &operator+(LossFunction<TensorRank> &other) override
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
                                           " don't match label dimensions " <<
                                           label.dimensions());
        return Tensor<0>((label - predict).square().mean())(0);
    }


    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label) override
    {
        fl_assert(predict.dimensions() == label.dimensions(),
                  "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());
        return 2 * (predict - label) / static_cast<Scalar>(predict.size());
    }
};

} // namespace fl

#endif //FLARE_MEAN_SQUARED_ERROR_HPP
