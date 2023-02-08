//
// Created by macross on 8/27/22.
//

#ifndef ORION_BINARY_CROSS_ENTROPY_HPP
#define ORION_BINARY_CROSS_ENTROPY_HPP

#include "loss_function.hpp"

namespace orion
{

template<int TensorRank>
class BinaryCrossEntropy : public LossFunction<TensorRank>
{
public:
    BinaryCrossEntropy() = default;


    explicit BinaryCrossEntropy(Scalar epsilon) : LossFunction<TensorRank>(epsilon)
    {
        // nothing to do
    }


    BinaryCrossEntropy(const Tensor<TensorRank> &predict,
                       const Tensor<TensorRank> &label)
    {
        (*this)(predict, label);
    }


    BinaryCrossEntropy &operator+(LossFunction<TensorRank> &other) override
    {
        this->loss += other.GetLoss();
        this->gradients += other.GetGradients();
        return *this;
    }


    Scalar Loss(const Tensor<TensorRank> &predict,
                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        Tensor<TensorRank> predict_clip = predict.clip(this->clip_min,
                                                       this->clip_max);
        return -Tensor<0>((label * (predict_clip + this->epsilon).log() +
                           (1.0 - label) *
                           (1.0 - predict_clip + this->epsilon).log())
                                  .mean()).coeff();
    };


    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label) override
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());

        return (-label / (predict + this->epsilon) +
                (1 - label) / (1 - predict + this->epsilon)) /
               static_cast<Scalar>(predict.size());
    };
};

} // namespace orion

#endif //ORION_BINARY_CROSS_ENTROPY_HPP
