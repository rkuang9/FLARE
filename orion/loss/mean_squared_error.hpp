//
// Created by macross on 9/5/22.
//

#ifndef ORION_MEAN_SQUARED_ERROR_HPP
#define ORION_MEAN_SQUARED_ERROR_HPP

#include "loss_function.hpp"

namespace orion
{

class MeanSquaredError : public LossFunction
{
public:
    explicit MeanSquaredError(int history_size = 1000);

    void CalculateLoss(const Tensor<2> &predict, const Tensor<2> &label) override;

    Scalar operator()(const Tensor<2> &predict,
                      const Tensor<2> &label) const override;
};

} // orion

#endif //ORION_MEAN_SQUARED_ERROR_HPP
