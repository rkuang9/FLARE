//
// Created by macross on 9/5/22.
//

#include "mean_squared_error.hpp"

namespace orion
{


MeanSquaredError::MeanSquaredError(int history_size)
{
    this->gradient_history2D.reserve(history_size);
    this->gradient_history3D.reserve(history_size);
    this->gradient_history3D.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void MeanSquaredError::CalculateLoss(const Tensor<2> &predict,
                                     const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.emplace_back(this->Gradient(predict, label));
}


void MeanSquaredError::CalculateLoss(const Tensor<3> &predict,
                                     const Tensor<3> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history3D.emplace_back(this->Gradient(predict, label));
}


void MeanSquaredError::CalculateLoss(const Tensor<4> &predict,
                                     const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.emplace_back(this->Gradient(predict, label));
}


template<int TensorRank>
Scalar MeanSquaredError::Loss(const Tensor<TensorRank> &predict,
                              const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions " <<
                                       label.dimensions());
    return Tensor<0>((label - predict).square().mean())(0);
}


template<int TensorRank>
Tensor<TensorRank> MeanSquaredError::Gradient(const Tensor<TensorRank> &predict,
                                              const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());
    return 2 * (predict - label) / static_cast<Scalar>(predict.size());
}

} // namespace orion