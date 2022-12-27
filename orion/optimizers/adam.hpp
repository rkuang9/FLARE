//
// Created by macross on 8/8/22.
//

#ifndef ORION_ADAM_HPP
#define ORION_ADAM_HPP

#include "optimizer.hpp"

namespace orion
{

// https://arxiv.org/abs/1412.6980
class Adam : public Optimizer
{
public:
    explicit Adam(Scalar learning_rate = 0.001, Scalar beta1 = 0.9,
                  Scalar beta2 = 0.999);

    void Step() override;

    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override;

    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override;

private:
    Scalar beta1; // momentum
    Scalar beta2; // RMSprop

    Scalar beta1_t; // beta1 at time step t
    Scalar beta2_t; // beta2 at time step t
    Scalar lr_t; // learning rate at time step t

    Scalar epsilon = 1e-7; // numeric stability constant

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    // is unordered_map faster?
    std::map<const Scalar *, Tensor<2>> momentum_dw;
    std::map<const Scalar *, Tensor<2>> rmsprop_dw;

    std::map<const Scalar *, Tensor<4>> momentum_dk;
    std::map<const Scalar *, Tensor<4>> rmsprop_dk;
};

} // namespace orion

#endif //ORION_ADAM_HPP
