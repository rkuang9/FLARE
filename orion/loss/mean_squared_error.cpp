//
// Created by macross on 9/5/22.
//

#include "mean_squared_error.hpp"

namespace orion
{


MeanSquaredError::MeanSquaredError(int history_size)
{
    this->gradient_history.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void MeanSquaredError::CalculateLoss(const Tensor2D &predict,
                                     const Tensor2D &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                         " don't match label dimensions " <<
                         label.dimensions());

    this->loss_history.push_back((*this)(predict, label));

    this->gradient_history.emplace_back(2 * (predict - label));
}


Scalar MeanSquaredError::operator()(const Tensor<2> &predict,
                                    const Tensor<2> &label) const
{
    Tensor<0> mean = (label - predict).square().mean();
    return mean(0);
}

} // namespaceorion