//
// Created by macross on 8/27/22.
//

#include "binary_cross_entropy.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace orion
{


BinaryCrossEntropy::BinaryCrossEntropy(Scalar epsilon, int history_size)
        : LossFunction(epsilon)
{
    this->gradient_history2D.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<2> &predict,
                                       const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.emplace_back(std::move(this->Gradient(predict, label)));
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<3> &predict,
                                       const Tensor<3> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history3D.emplace_back(std::move(this->Gradient(predict, label)));
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<4> &predict,
                                       const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.emplace_back(std::move(this->Gradient(predict, label)));
}


template<int TensorRank>
Scalar BinaryCrossEntropy::Loss(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    Tensor<TensorRank> predict_clip = predict.clip(this->clip_min, this->clip_max);
    return -Tensor<0>((label * (predict_clip + this->epsilon).log() +
                       (1.0 - label) * (1.0 - predict_clip + this->epsilon).log())
                              .mean()).coeff();
}


template<int TensorRank>
Tensor<TensorRank> BinaryCrossEntropy::Gradient(const Tensor<TensorRank> &predict,
                                                const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    return (-label / (predict + this->epsilon) +
            (1 - label) / (1 - predict + this->epsilon)) /
           static_cast<Scalar>(predict.size());
}

} // namespace orion