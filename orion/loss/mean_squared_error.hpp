//
// Created by macross on 9/5/22.
//

#ifndef ORION_MEAN_SQUARED_ERROR_HPP
#define ORION_MEAN_SQUARED_ERROR_HPP

#include "loss.hpp"

namespace orion
{

class MeanSquaredError : public Loss
{
public:
    explicit MeanSquaredError(int history_size = 1000);

    void CalculateLoss(const Tensor2D &predict, const Tensor2D &label) override;

    Scalar operator()(const Tensor<2> &predict,
                      const Tensor<2> &label) const override;
};

} // orion

#endif //ORION_MEAN_SQUARED_ERROR_HPP
