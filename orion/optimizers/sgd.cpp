//
// Created by macross on 8/8/22.
//

#include "sgd.hpp"

namespace orion
{

SGD::SGD(Scalar learning_rate, Scalar momentum) : Optimizer(learning_rate),
                                                  momentum(momentum)
{
    // nothing to do
}


void SGD::Minimize(Tensor<2> &weights, const Tensor<2> &gradients)
{
    Tensor<2> &velocity = this->v_dw[weights.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as weights
        velocity.resize(weights.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + gradients; // * (1 - this->momentum)

    weights -= this->learning_rate * velocity;
}


void SGD::Minimize(Tensor<4> &kernels, const Tensor<4> &gradients)
{
    Tensor<4> &velocity = this->v_dk[kernels.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as bias
        velocity.resize(kernels.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + gradients; // * (1 - this->momentum)

    kernels -= this->learning_rate * velocity;
}


void SGD::Step()
{
    // nothing to do
}

} // namespace orion