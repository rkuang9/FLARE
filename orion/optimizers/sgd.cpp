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


void SGD::Minimize(Tensor<1> &weights, const Tensor<1> &gradients)
{
    Tensor<1> &velocity = this->v_db[weights.data()];
    this->Update(weights, gradients, velocity);
}


void SGD::Minimize(Tensor<2> &weights, const Tensor<2> &gradients)
{
    Tensor<2> &velocity = this->v_dw[weights.data()];
    this->Update(weights, gradients, velocity);
}


void SGD::Minimize(Tensor<3> &weights, const Tensor<3> &gradients)
{
    Tensor<3> &velocity = this->v_dy[weights.data()];
    this->Update(weights, gradients, velocity);
}


void SGD::Minimize(Tensor<4> &kernels, const Tensor<4> &gradients)
{
    Tensor<4> &velocity = this->v_dk[kernels.data()];
    this->Update(kernels, gradients, velocity);
}


void SGD::Step()
{
    // nothing to do
}


template<int TensorRank>
void SGD::Update(Tensor<TensorRank> &weights,
                           const Tensor<TensorRank> &gradients,
                           Tensor<TensorRank> &velocity)
{
    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as bias
        velocity.resize(weights.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + gradients; // * (1 - this->momentum)

    weights -= this->learning_rate * velocity;
}

} // namespace orion