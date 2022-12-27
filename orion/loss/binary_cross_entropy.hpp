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

    template<int TensorRank>
    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label);

    template<int TensorRank>
    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label);
};


template<int TensorRank>
Scalar BinaryCrossEntropy::Loss(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label)
{
    Tensor<TensorRank> predict_clip = predict.clip(this->clip_min, this->clip_max);
    return -Tensor<0>((label * (predict_clip + this->epsilon).log() +
                       (1.0 - label) * (1.0 - predict_clip + this->epsilon).log())
                              .mean()).coeff();
}


template<int TensorRank>
Tensor<TensorRank> BinaryCrossEntropy::Gradient(const Tensor<TensorRank> &predict,
                                                const Tensor<TensorRank> &label)
{
    return (-label / (predict + this->epsilon) +
            (1 - label) / (1 - predict + this->epsilon)) /
           (Scalar(predict.dimensions().TotalSize() / predict.dimension(0)));
}

} // namespace orion

#endif //ORION_BINARY_CROSS_ENTROPY_HPP
