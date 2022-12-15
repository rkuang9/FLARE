//
// Created by macross on 8/17/22.
//

#ifndef ORION_RMSPROP_HPP
#define ORION_RMSPROP_HPP

#include "optimizer.hpp"

namespace orion
{

class RMSprop : public Optimizer
{
public:
    explicit RMSprop(Scalar learning_rate = 0.001, Scalar momentum = 0.9);

    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override;

    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override;

    void Step() override;

private:
    Scalar momentum = 0.9;
    Scalar epsilon = 1e-7; // prevent divison by zero

    std::map<const Scalar *, Tensor<2>> s_dw;
    std::map<const Scalar *, Tensor<4>> s_dk;
};

} // orion

#endif //ORION_RMSPROP_HPP
