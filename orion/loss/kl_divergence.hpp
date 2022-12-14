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

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};

template<int TensorRank>
Scalar KLDivergence::Loss(const Tensor<TensorRank> &predict,
                          const Tensor<TensorRank> &label)
{
    return Tensor<0>((label * (label / (predict + 1e-7)).log()).mean()).coeff();
}


template<int TensorRank>
Tensor<TensorRank> KLDivergence::Gradient(const Tensor<TensorRank> &predict,
                                          const Tensor<TensorRank> &label)
{
    auto zero = static_cast<Scalar>(0.0);

    return (predict != zero).select(-label / (predict), predict.constant(zero)) /
           Scalar(predict.dimensions().TotalSize() / predict.dimension(0));
}

}

#endif //ORION_KL_DIVERGENCE_HPP
