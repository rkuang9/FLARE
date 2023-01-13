//
// Created by Macross on 12/13/22.
//

#include "mean_absolute_error.hpp"

namespace orion
{

MeanAbsoluteError::MeanAbsoluteError(int history_size)
{
    this->gradient_history2D.reserve(history_size);
    this->gradient_history4D.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void MeanAbsoluteError::CalculateLoss(const Tensor<2> &predict,
                                      const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.push_back(this->Gradient(predict, label));
}


void MeanAbsoluteError::CalculateLoss(const Tensor<3> &predict,
                                      const Tensor<3> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history3D.push_back(this->Gradient(predict, label));
}


void MeanAbsoluteError::CalculateLoss(const Tensor<4> &predict,
                                      const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}


template<int TensorRank>
Tensor<TensorRank> MeanAbsoluteError::Gradient(const Tensor<TensorRank> &predict,
                                               const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    return (predict != label)
                   .select((predict > label)
                                   .select(predict.constant(1),
                                           predict.constant(-1)),
                           predict.constant(0)) /
           Scalar(predict.dimensions().TotalSize() / predict.dimension(0));
}

}