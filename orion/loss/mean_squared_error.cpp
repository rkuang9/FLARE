//
// Created by macross on 9/5/22.
//

#include "mean_squared_error.hpp"

namespace orion
{


MeanSquaredError::MeanSquaredError(int history_size)
{
    this->gradient_history2D.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void MeanSquaredError::CalculateLoss(const Tensor<2> &predict,
                                     const Tensor<2> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions " <<
                                       label.dimensions());

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.push_back(this->Gradient(predict, label));
}


void MeanSquaredError::CalculateLoss(const Tensor<4> &predict,
                                     const Tensor<4> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions " <<
                                       label.dimensions());

    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}

} // namespace orion