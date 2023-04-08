//
// Created by macross on 8/17/22.
//

#include "rmsprop.hpp"

namespace fl
{

RMSprop::RMSprop(Scalar learning_rate, Scalar momentum) :
        Optimizer(learning_rate), momentum(momentum)
{

}


void RMSprop::Minimize(Tensor<1> &weights, const Tensor<1> &gradients)
{
    this->Update(weights, gradients, this->s_db[weights.data()]);
}


void RMSprop::Minimize(Tensor<2> &weights, const Tensor<2> &gradients)
{
    this->Update(weights, gradients, this->s_dw[weights.data()]);
}


void RMSprop::Minimize(Tensor<3> &weights, const Tensor<3> &gradients)
{
    this->Update(weights, gradients, this->s_dw3[weights.data()]);
}


void RMSprop::Minimize(Tensor<4> &kernels, const Tensor<4> &gradients)
{
    this->Update(kernels, gradients, this->s_dk[kernels.data()]);
}


void RMSprop::Step()
{

}


template<int TensorRank>
void RMSprop::Update(Tensor<TensorRank> &weights,
                     const Tensor<TensorRank> &gradients,
                     Tensor<TensorRank> &velocity)
{
    if (velocity.size() == 0) {
        velocity.resize(weights.dimensions());
        velocity.setZero();
    }

    velocity.device(this->device) =
            this->momentum * velocity + (1 - this->momentum) * gradients * gradients;

    weights.device(this->device) -=
            this->learning_rate * gradients / (velocity.sqrt() + this->epsilon);
}

} // namespace fl