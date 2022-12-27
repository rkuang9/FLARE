//
// Created by Macross on 11/25/22.
//

#include "categorical_cross_entropy.hpp"

namespace orion
{

CategoricalCrossEntropy::CategoricalCrossEntropy(Scalar epsilon, int history_size)
        : LossFunction(epsilon)
{

}


void CategoricalCrossEntropy::CalculateLoss(const Tensor<2> &predict,
                                            const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.push_back(this->Gradient(predict, label));
}


void CategoricalCrossEntropy::CalculateLoss(const Tensor<4> &predict,
                                            const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}

} // namespace orion