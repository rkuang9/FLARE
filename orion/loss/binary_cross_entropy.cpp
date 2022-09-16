//
// Created by macross on 8/27/22.
//

#include "binary_cross_entropy.hpp"
#include <iostream>
#include <cmath>

namespace orion
{


BinaryCrossEntropy::BinaryCrossEntropy(Scalar epsilon, int history_size)
        : Loss(epsilon)
{
    this->gradient_history.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void BinaryCrossEntropy::CalculateLoss(const Tensor<2> &predict,
                                       const Tensor<2> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                         " don't match label dimensions " <<
                         label.dimensions());

    this->loss_history.push_back((*this)(predict, label));

    this->gradient_history.emplace_back(
            (-label / ((predict  + this->epsilon))) +
            (1 - label) / ((1 - predict + this->epsilon)));
}


Scalar BinaryCrossEntropy::operator()(const Tensor<2> &predict,
                                      const Tensor<2> &label) const
{
    auto predict_clip = predict.cwiseMax(this->clip_min).cwiseMin(this->clip_max);

    Tensor<0> mean = (label * (predict_clip + this->epsilon).log() +
                      (1 - label) * (1 - predict_clip + this->epsilon).log()).mean();

    return -mean(0);
}

} // namespace orion

/*
def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)
 */