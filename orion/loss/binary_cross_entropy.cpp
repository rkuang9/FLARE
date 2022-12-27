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
                                       " don't match label dimensions "
                                       << label.dimensions());

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.emplace_back(std::move(this->Gradient(predict, label)));
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<4> &predict,
                                       const Tensor<4> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions()
                                       << " don't match label dimensions "
                                       << label.dimensions());

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.emplace_back(std::move(this->Gradient(predict, label)));
}

} // namespace orion