//
// Created by Macross on 12/14/22.
//

#include "kl_divergence.hpp"

namespace orion
{

KLDivergence::KLDivergence(int history_size)
{
    this->gradient_history2D.reserve(history_size);
    this->loss_history.reserve(history_size);
}


void KLDivergence::CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.push_back(this->Gradient(predict, label));
}


void KLDivergence::CalculateLoss(const Tensor<3> &predict, const Tensor<3> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history3D.push_back(this->Gradient(predict, label));
}


void KLDivergence::CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}


template<int TensorRank>
Scalar KLDivergence::Loss(const Tensor<TensorRank> &predict,
                          const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    return Tensor<0>((label * ((label + this->epsilon) / (predict + this->epsilon))
            .log()).mean()).coeff();
}


template<int TensorRank>
Tensor<TensorRank> KLDivergence::Gradient(const Tensor<TensorRank> &predict,
                                          const Tensor<TensorRank> &label)
{
    orion_assert(predict.dimensions() == label.dimensions(),
                 "predict dimensions " << predict.dimensions() <<
                                       " don't match label dimensions "
                                       << label.dimensions());

    auto zero = static_cast<Scalar>(0.0);

    return (predict != zero).select(-label / (predict), predict.constant(zero)) /
           static_cast<Scalar>(predict.size());
}

}