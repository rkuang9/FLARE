//
// Created by Macross on 12/14/22.
//

#ifndef ORION_KL_DIVERGENCE_HPP
#define ORION_KL_DIVERGENCE_HPP

#include "loss_function.hpp"

namespace orion
{

class KLDivergence : public LossFunction
{
public:
    explicit KLDivergence(int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    void CalculateLoss(const Tensor<3> &predict, const Tensor<3> &label) override;

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};

}

#endif //ORION_KL_DIVERGENCE_HPP
