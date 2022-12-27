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

void KLDivergence::CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}

}