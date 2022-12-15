//
// Created by macross on 8/17/22.
//

#include "rmsprop.hpp"

namespace orion
{

RMSprop::RMSprop(Scalar learning_rate, Scalar momentum) :
        Optimizer(learning_rate), momentum(momentum)
{

}


void RMSprop::Minimize(Tensor<2> &weights, const Tensor<2> &gradients)
{
    Tensor<2> &velocity = this->s_dw[weights.data()];

    if (velocity.size() == 0) {
        velocity.resize(weights.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity +
               (1 - this->momentum) * gradients * gradients;

    weights -= this->learning_rate * gradients / (velocity.sqrt() + this->epsilon);
}


void RMSprop::Minimize(Tensor<4> &kernels, const Tensor<4> &gradients)
{
    Tensor<4> &velocity = this->s_dk[kernels.data()];

    if (velocity.size() == 0) {
        velocity.resize(kernels.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + (1 - this->momentum) * gradients * gradients;

    kernels -= this->learning_rate * gradients / (velocity.sqrt() + this->epsilon);
}

void RMSprop::Step()
{

}

} // namespace orion