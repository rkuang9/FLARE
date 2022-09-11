//
// Created by macross on 8/27/22.
//

#ifndef ORION_BINARY_CROSS_ENTROPY_HPP
#define ORION_BINARY_CROSS_ENTROPY_HPP

#include "loss.hpp"

namespace orion
{

class BinaryCrossEntropy : public Loss
{
public:
    BinaryCrossEntropy() = default;

    explicit BinaryCrossEntropy(Scalar epsilon, int history_size = 1000);

    void CalculateLoss(const Tensor2D &predict, const Tensor2D &label) override;

    Scalar operator()(const Tensor<2> &predict,
                      const Tensor<2> &label) const override;
};

} // namespace orion

#endif //ORION_BINARY_CROSS_ENTROPY_HPP
