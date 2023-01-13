//
// Created by Macross on 11/25/22.
//

#ifndef ORION_CATEGORICAL_CROSS_ENTROPY_H
#define ORION_CATEGORICAL_CROSS_ENTROPY_H

#include "loss_function.hpp"

namespace orion
{

class CategoricalCrossEntropy : public LossFunction
{
public:
    CategoricalCrossEntropy() = default;

    explicit CategoricalCrossEntropy(Scalar epsilon, int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);

};

using CrossEntropy = CategoricalCrossEntropy; // alias, since both are the same

} // namespace orion

#endif //ORION_CATEGORICAL_CROSS_ENTROPY_H
