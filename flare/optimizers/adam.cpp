//
// Created by macross on 8/8/22.
//

#include "adam.hpp"
#include <iostream>
namespace fl
{

Adam::Adam(Scalar learning_rate, Scalar beta1, Scalar beta2)
        : Optimizer(learning_rate),
          beta1(beta1), beta2(beta2),
          beta1_t(beta1), beta2_t(beta2),
          lr_t(this->learning_rate * std::sqrt(1 - this->beta2_t) /
               (1 - this->beta1_t))
{
}


void Adam::Step()
{
    this->beta1_t *= this->beta1;
    this->beta2_t *= this->beta2;

    this->lr_t = this->learning_rate * std::sqrt(1 - this->beta2_t) /
                 (1 - this->beta1_t);
}


void Adam::Minimize(Tensor<1> &weights, const Tensor<1> &gradients)
{
    Tensor<1> &momentum = this->momentum_db[weights.data()];
    Tensor<1> &rmsprop = this->rmsprop_db[weights.data()];

    this->Update(weights, gradients, momentum, rmsprop);
}


void Adam::Minimize(Tensor<2> &weights, const Tensor<2> &gradients)
{
    Tensor<2> &momentum = this->momentum_dw[weights.data()];
    Tensor<2> &rmsprop = this->rmsprop_dw[weights.data()];

    this->Update(weights, gradients, momentum, rmsprop);
}


void Adam::Minimize(Tensor<3> &weights, const Tensor<3> &gradients)
{
    Tensor<3> &momentum = this->momentum_dw3[weights.data()];
    Tensor<3> &rmsprop = this->rmsprop_dw3[weights.data()];

    this->Update(weights, gradients, momentum, rmsprop);
}


void Adam::Minimize(Tensor<4> &kernels, const Tensor<4> &gradients)
{
    Tensor<4> &momentum = this->momentum_dk[kernels.data()];
    Tensor<4> &rmsprop = this->rmsprop_dk[kernels.data()];

    this->Update(kernels, gradients, momentum, rmsprop);
}


template<int TensorRank>
void Adam::Update(
        Tensor<TensorRank> &weights, const Tensor<TensorRank> &gradients,
        Tensor<TensorRank> &momentum, Tensor<TensorRank> &rmsprop)
{
// on first run, initialize zero matrix with same shape as weights
    if (momentum.size() == 0) {
        momentum.resize(weights.dimensions());
        momentum.setZero();
    }

    if (rmsprop.size() == 0) {
        rmsprop.resize(weights.dimensions());
        rmsprop.setZero();
    }

    momentum = this->beta1 * momentum + (1 - this->beta1) * gradients; // momentum
    rmsprop = this->beta2 * rmsprop +
              (1 - this->beta2) * gradients.square(); // RMSprop

    weights -= this->lr_t * momentum / (rmsprop.sqrt() + this->epsilon);
}

} // namespace fl