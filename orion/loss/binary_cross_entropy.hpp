//
// Created by macross on 8/27/22.
//

#ifndef ORION_BINARY_CROSS_ENTROPY_HPP
#define ORION_BINARY_CROSS_ENTROPY_HPP

#include "loss_function.hpp"

namespace orion
{

class BinaryCrossEntropy : public LossFunction
{
public:
    BinaryCrossEntropy() = default;

    explicit BinaryCrossEntropy(Scalar epsilon, int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    void CalculateLoss(const Tensor<4> &predict, const Tensor<4> &label) override;

private:
    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label);


    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};

} // namespace orion

#endif //ORION_BINARY_CROSS_ENTROPY_HPP
