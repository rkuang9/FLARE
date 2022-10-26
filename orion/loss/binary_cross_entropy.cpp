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
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                         " don't match label dimensions " << label.dimensions());

    // there must be one feature per batch and values in range [0, 1]
    orion_assert(predict.dimension(0) == 1,
                 "BinaryCrossEntropy expects 1 output feature");

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.emplace_back(std::move(this->Gradient(predict, label)));
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<4> &predict,
                                       const Tensor<4> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                         " don't match label dimensions " << label.dimensions());

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.emplace_back(std::move(this->Gradient(predict, label)));
}


template<int TensorRank>
Scalar BinaryCrossEntropy::Loss(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label)
{
    // TODO: replace cwiseMax, cwiseMin with clip(min, max)
    auto predict_clip = predict.cwiseMax(this->clip_min).cwiseMin(this->clip_max);

    Tensor<0> mean = (label * (predict_clip + this->epsilon).log() +
                      (1 - label) * (1 - predict_clip + this->epsilon).log()).mean();
    return -mean(0);
}


template<int TensorRank>
Tensor<TensorRank> BinaryCrossEntropy::Gradient(const Tensor<TensorRank> &predict,
                                                const Tensor<TensorRank> &label)
{
    Tensor<TensorRank> gradients =
            ((-label / ((predict + this->epsilon))) +
             (1 - label) / ((1 - predict + this->epsilon)));
    return gradients;
}

} // namespace orion